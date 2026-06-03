from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from maskrcnn_pipeline.dataset import RockCocoDataset, collate_fn
from maskrcnn_pipeline.models import build_maskrcnn
from maskrcnn_pipeline.utils import seed_everything


# Recommended debug run:
# python3 -u -m maskrcnn_pipeline.train_maskrcnn \
#   --epochs 1 \
#   --batch-size 2 \
#   --lr 1e-4 \
#   --log-interval 20 \
#   --debug-first-batch \
#   --profile-data-time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on the rock COCO dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("maskrcnn_pipeline/dataset"))
    parser.add_argument("--image-root", type=Path, default=Path("/home/sanghwon/capstone/datasets/rock_det"))
    parser.add_argument("--output-dir", type=Path, default=Path("maskrcnn_pipeline/runs/maskrcnn_rock"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--weights", default="DEFAULT", help="DEFAULT or none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20, help="Print training status every N steps.")
    parser.add_argument("--debug-first-batch", action="store_true", help="Print image/target info for the first batch.")
    parser.add_argument("--profile-data-time", action="store_true", help="Print DataLoader and compute timings separately.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = RockCocoDataset(args.data_dir / "train.json", args.image_root)
    val_ds = RockCocoDataset(args.data_dir / "val.json", args.image_root)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = build_maskrcnn(num_classes=2, weights=args.weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print_start_info(args, device, train_ds, val_ds, train_loader)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, args, epoch)
        val_metrics = evaluate_loss(model, val_loader, device, args, epoch)
        val_loss = val_metrics["total_loss"]

        print(format_metrics(epoch, "train", train_metrics), flush=True)
        print(format_metrics(epoch, "val", val_metrics), flush=True)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "args": vars(args),
        }
        last_path = args.output_dir / "last.pth"
        best_path = args.output_dir / "best.pth"
        torch.save(checkpoint, last_path)
        print(f"[CKPT] saved last checkpoint: {last_path}", flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, best_path)
            print(f"[CKPT] saved best checkpoint: {best_path}", flush=True)


def train_one_epoch(model, loader, optimizer, scaler, device, args, epoch: int):
    model.train()
    running = defaultdict(float)
    steps = 0
    total_steps = len(loader)
    epoch_start = time.perf_counter()
    step_time_sum = 0.0
    data_start = time.perf_counter()
    iterator = iter(loader)

    print(f"[EPOCH] start epoch {epoch}/{args.epochs}", flush=True)
    while steps < total_steps:
        step = steps + 1
        images = targets = None
        try:
            images, targets = next(iterator)
            data_time = time.perf_counter() - data_start
        except StopIteration:
            break
        except Exception as exc:
            log_exception(epoch, step, exc, targets, device)
            raise SystemExit(1) from exc

        if args.debug_first_batch and epoch == 1 and step == 1:
            print_first_batch_debug(images, targets)

        step_start = time.perf_counter()
        try:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]

            optimizer.zero_grad(set_to_none=True)
            sync_cuda(device)
            compute_start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss_dict = model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sync_cuda(device)
            compute_time = time.perf_counter() - compute_start

            update_running(running, loss_dict, total_loss)
            steps += 1
            step_time = time.perf_counter() - step_start
            step_time_sum += step_time

            if should_log(step, total_steps, args.log_interval):
                print_train_step_log(epoch, args.epochs, step, total_steps, loss_dict, total_loss, step_time, step_time_sum / steps, device)
                if args.profile_data_time:
                    print_profile_log(epoch, args.epochs, step, total_steps, data_time, compute_time)
        except Exception as exc:
            log_exception(epoch, step, exc, targets, device)
            raise SystemExit(1) from exc
        finally:
            data_start = time.perf_counter()

    elapsed_hours = (time.perf_counter() - epoch_start) / 3600.0
    metrics = average_metrics(running, steps)
    print(f"[EPOCH] end epoch {epoch}/{args.epochs} | train_loss_avg={metrics.get('total_loss', 0.0):.4f} | elapsed={elapsed_hours:.2f}h", flush=True)
    return metrics


@torch.no_grad()
def evaluate_loss(model, loader, device, args, epoch: int):
    was_training = model.training
    model.train()
    running = defaultdict(float)
    steps = 0
    total_steps = len(loader)
    epoch_start = time.perf_counter()
    iterator = iter(loader)

    print(f"[VAL] start epoch {epoch}/{args.epochs}", flush=True)
    while steps < total_steps:
        step = steps + 1
        images = targets = None
        try:
            images, targets = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            log_exception(epoch, step, exc, targets, device)
            raise SystemExit(1) from exc

        try:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss_dict = model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())
            update_running(running, loss_dict, total_loss)
            steps += 1
            if should_log(step, total_steps, args.log_interval):
                print(f"[VAL] step {step}/{total_steps} | loss={float(total_loss.detach().cpu()):.4f}", flush=True)
        except Exception as exc:
            log_exception(epoch, step, exc, targets, device)
            raise SystemExit(1) from exc
    model.train(was_training)
    elapsed_hours = (time.perf_counter() - epoch_start) / 3600.0
    metrics = average_metrics(running, steps)
    print(f"[VAL] end epoch {epoch}/{args.epochs} | val_loss_avg={metrics.get('total_loss', 0.0):.4f} | elapsed={elapsed_hours:.2f}h", flush=True)
    return metrics


def update_running(running, loss_dict, total_loss):
    running["total_loss"] += float(total_loss.detach().cpu())
    for key, value in loss_dict.items():
        running[key] += float(value.detach().cpu())


def average_metrics(running, steps: int):
    divisor = max(1, steps)
    return {key: value / divisor for key, value in running.items()}


def format_metrics(epoch: int, split: str, metrics: dict[str, float]) -> str:
    keys = ["loss_classifier", "loss_box_reg", "loss_mask", "total_loss"]
    joined = " ".join(f"{key}={metrics.get(key, 0.0):.4f}" for key in keys)
    return f"epoch={epoch:03d} split={split} {joined}"


def print_start_info(args, device, train_ds, val_ds, train_loader) -> None:
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"
    last_path = args.output_dir / "last.pth"
    best_path = args.output_dir / "best.pth"
    print(f"[INFO] device: {device}", flush=True)
    print(f"[INFO] torch: {torch.__version__}", flush=True)
    print(f"[INFO] cuda available: {torch.cuda.is_available()}", flush=True)
    print(f"[INFO] gpu: {gpu_name}", flush=True)
    print(f"[INFO] train images: {len(train_ds)}", flush=True)
    print(f"[INFO] val images: {len(val_ds)}", flush=True)
    print(f"[INFO] batch size: {args.batch_size}", flush=True)
    print(f"[INFO] num_workers: {args.num_workers}", flush=True)
    print(f"[INFO] epochs: {args.epochs}", flush=True)
    print(f"[INFO] lr: {args.lr}", flush=True)
    print(f"[INFO] steps/epoch: {len(train_loader)}", flush=True)
    print(f"[INFO] output directory: {args.output_dir}", flush=True)
    print(f"[INFO] checkpoint last: {last_path}", flush=True)
    print(f"[INFO] checkpoint best: {best_path}", flush=True)


def print_first_batch_debug(images, targets) -> None:
    print("[DEBUG] first batch loaded", flush=True)
    print(f"[DEBUG] batch image count: {len(images)}", flush=True)
    for index, (image, target) in enumerate(zip(images, targets)):
        boxes = target.get("boxes")
        masks = target.get("masks")
        labels = target.get("labels")
        image_id = target.get("image_id")
        print(f"[DEBUG] image[{index}] shape: {image.shape}", flush=True)
        print(f"[DEBUG] target[{index}] boxes: {boxes.shape if boxes is not None else 'N/A'}", flush=True)
        print(f"[DEBUG] target[{index}] masks: {masks.shape if masks is not None else 'N/A'}", flush=True)
        print(f"[DEBUG] target[{index}] labels unique: {labels.unique().tolist() if labels is not None else 'N/A'}", flush=True)
        print(f"[DEBUG] target[{index}] image_id: {tensor_to_list(image_id)}", flush=True)
        print(f"[DEBUG] target[{index}] boxes min/max: {tensor_min_max(boxes)}", flush=True)
        print(f"[DEBUG] target[{index}] masks dtype: {masks.dtype if masks is not None else 'N/A'}", flush=True)
        print(f"[DEBUG] target[{index}] masks device: {masks.device if masks is not None else 'N/A'}", flush=True)


def print_train_step_log(epoch, epochs, step, total_steps, loss_dict, total_loss, step_time, avg_time, device) -> None:
    loss_values = loss_dict_to_float(loss_dict, total_loss)
    eta_hours = ((total_steps - step) * avg_time) / 3600.0
    gpu_alloc, gpu_reserved = gpu_memory(device)
    print(
        f"[TRAIN] epoch {epoch}/{epochs} | step {step}/{total_steps} | "
        f"loss={loss_values['total_loss']:.4f} | cls={loss_values['loss_classifier']:.4f} | "
        f"box={loss_values['loss_box_reg']:.4f} | mask={loss_values['loss_mask']:.4f} | "
        f"obj={loss_values['loss_objectness']:.4f} | rpn={loss_values['loss_rpn_box_reg']:.4f} | "
        f"step={step_time:.2f}s | avg={avg_time:.2f}s | eta={eta_hours:.1f}h | "
        f"gpu_alloc={gpu_alloc} | gpu_reserved={gpu_reserved}",
        flush=True,
    )


def print_profile_log(epoch, epochs, step, total_steps, data_time, compute_time) -> None:
    print(f"[PROFILE] epoch {epoch}/{epochs} | step {step}/{total_steps} | data={data_time:.2f}s | compute={compute_time:.2f}s", flush=True)


def loss_dict_to_float(loss_dict, total_loss) -> dict[str, float]:
    values = {
        "total_loss": float(total_loss.detach().cpu()),
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_mask": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }
    for key in values:
        if key == "total_loss":
            continue
        value = loss_dict.get(key)
        if value is not None:
            values[key] = float(value.detach().cpu())
    return values


def should_log(step: int, total_steps: int, interval: int) -> bool:
    interval = max(1, interval)
    return step == 1 or step % interval == 0 or step == total_steps


def gpu_memory(device) -> tuple[str, str]:
    if device.type != "cuda":
        return "N/A", "N/A"
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    return f"{allocated:.1f}GB", f"{reserved:.1f}GB"


def sync_cuda(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def log_exception(epoch: int, step: int, exc: Exception, targets, device) -> None:
    print(f"[ERROR] failed at epoch {epoch} step {step}", flush=True)
    image_ids = extract_image_ids(targets)
    if image_ids:
        print(f"[ERROR] image_ids: {image_ids}", flush=True)
    print(f"[ERROR] {type(exc).__name__}: {exc}", flush=True)
    if device.type == "cuda":
        try:
            print(torch.cuda.memory_summary(), flush=True)
        except Exception as summary_exc:
            print(f"[ERROR] CUDA memory summary unavailable: {summary_exc}", flush=True)


def extract_image_ids(targets) -> list[int]:
    if not targets:
        return []
    image_ids = []
    for target in targets:
        image_id = target.get("image_id")
        if image_id is None:
            continue
        image_ids.extend(tensor_to_list(image_id))
    return image_ids


def tensor_to_list(value) -> list[int]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def tensor_min_max(value) -> str:
    if value is None or value.numel() == 0:
        return "N/A"
    return f"{float(value.min()):.2f}/{float(value.max()):.2f}"


if __name__ == "__main__":
    main()
