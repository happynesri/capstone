#!/usr/bin/env python3
"""
여러 preset 결과를 비교 분석하는 도구
"""
import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def compare_preset_results(image_name, output_dir="../result"):
    """
    동일 이미지에 대한 여러 preset 결과를 비교 분석합니다.
    """
    # 해당 이미지의 모든 결과 JSON 파일 찾기
    result_files = sorted(
        Path(output_dir).glob(f"{image_name}_results_*.json")
    )
    
    if not result_files:
        print(f"결과를 찾을 수 없습니다: {image_name}")
        return
    
    # 각 preset별 파일 분류
    preset_results = defaultdict(list)
    known_presets = {"baseline", "wide", "tight", "binary_low", "binary_high"}
    
    for json_file in result_files:
        # 파일명에서 preset 추출
        fname = json_file.stem
        # 구조: {image_name}_results_{preset}_{timestamp}
        # 예: 101-d-001_results_wide_20260429_214006
        parts = fname.replace(f"{image_name}_results_", "").split('_')
        
        # 마지막 8자리를 제거하면 timestamp (YYYYMMDD_HHMMSS)
        if len(parts) >= 2:
            # parts[-2:]가 timestamp (YYYYMMDD, HHMMSS)
            potential_preset = '_'.join(parts[:-2])
            
            # 알려진 preset인지 확인
            if potential_preset in known_presets:
                preset = potential_preset
            else:
                # 구식 format은 무시
                continue
        else:
            continue
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        preset_results[preset].append({
            'file': json_file,
            'data': data,
            'timestamp': data['timestamp']
        })
    
    # 결과 출력
    print(f"\n{'='*80}")
    print(f"이미지: {image_name}")
    print(f"{'='*80}\n")
    
    summary = []
    for preset_name in sorted(preset_results.keys()):
        results = preset_results[preset_name]
        latest = results[-1]  # 가장 최신 결과만 사용
        data = latest['data']
        
        total = data['total_stones']
        passed = data['pass_count']
        failed = data['fail_count']
        
        summary.append({
            'preset': preset_name,
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': (passed / total * 100) if total > 0 else 0,
            'stones': data['stones']
        })
    
    # 테이블 출력
    print(f"{'Preset':<15} {'Total':>6} {'Pass':>6} {'Fail':>6} {'Pass Rate':>12} {'Stones Info':<30}")
    print("-" * 80)
    
    for s in summary:
        stones_info = ', '.join([
            f"L:{st['long_axis_px']:.0f}" for st in s['stones'][:3]
        ])
        if len(s['stones']) > 3:
            stones_info += f", ... ({len(s['stones'])} total)"
        
        print(
            f"{s['preset']:<15} {s['total']:>6} {s['passed']:>6} {s['failed']:>6} "
            f"{s['pass_rate']:>10.1f}% {stones_info:<30}"
        )
    
    # 결론
    print(f"\n{'='*80}")
    best = max(summary, key=lambda x: x['pass_rate'])
    print(f"✓ 최고 성능 preset: {best['preset']} "
          f"({best['pass_rate']:.1f}% pass rate, {best['passed']}/{best['total']})")
    print(f"{'='*80}\n")
    
    return summary


def compare_multiple_images(image_names, output_dir="../result"):
    """
    여러 이미지에 대해 모든 preset 결과를 비교합니다.
    """
    all_summaries = {}
    for image_name in image_names:
        summary = compare_preset_results(image_name, output_dir)
        if summary:
            all_summaries[image_name] = summary
    
    # 전체 통계
    if all_summaries:
        print(f"\n{'='*80}")
        print("전체 통계")
        print(f"{'='*80}\n")
        
        preset_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0})
        for image_summaries in all_summaries.values():
            for s in image_summaries:
                preset_stats[s['preset']]['total'] += s['total']
                preset_stats[s['preset']]['passed'] += s['passed']
                preset_stats[s['preset']]['failed'] += s['failed']
        
        print(f"{'Preset':<15} {'Total':>6} {'Pass':>6} {'Fail':>6} {'Overall Pass Rate':>20}")
        print("-" * 80)
        
        for preset_name in sorted(preset_stats.keys()):
            stats = preset_stats[preset_name]
            total = stats['total']
            passed = stats['passed']
            overall_rate = (passed / total * 100) if total > 0 else 0
            
            print(
                f"{preset_name:<15} {total:>6} {passed:>6} {stats['failed']:>6} "
                f"{overall_rate:>18.1f}%"
            )
        
        print(f"{'='*80}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        help="분석할 이미지 파일명 (확장자 제외, 예: 101-d-001)"
    )
    parser.add_argument(
        "--images",
        type=str,
        help="분석할 이미지 파일명 목록 (쉼표로 구분, 예: 101-d-001,101-d-002)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../result",
        help="결과 폴더 경로"
    )
    args = parser.parse_args()
    
    if args.image:
        compare_preset_results(args.image, args.output_dir)
    elif args.images:
        image_names = [name.strip() for name in args.images.split(',')]
        compare_multiple_images(image_names, args.output_dir)
    else:
        print("사용 예시:")
        print("  python3 compare_presets.py --image 101-d-001")
        print("  python3 compare_presets.py --images 101-d-001,101-d-002,101-d-004")


if __name__ == "__main__":
    main()
