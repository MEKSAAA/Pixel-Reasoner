#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计评估指标的准确率脚本
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def calculate_accuracy(metrics_file):
    """
    从metrics文件中计算准确率
    
    Args:
        metrics_file: metrics JSON文件路径
    
    Returns:
        dict: 包含各种统计信息的字典
    """
    with open(metrics_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 总体统计
    total_correct = 0
    total_samples = 0
    
    # 按类别统计（从benchname中提取）
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for item in data:
        benchname = item['benchname']
        ncorrect = item['ncorrect']
        ntotal = item['ntotal']
        pass1 = item['pass1']
        
        # 累加总体统计
        total_correct += ncorrect
        total_samples += ntotal
        
        # 提取类别（如果benchname包含类别信息）
        # 例如: mvtec_test_302 -> mvtec_test
        category = '_'.join(benchname.split('_')[:-1])
        category_stats[category]['correct'] += ncorrect
        category_stats[category]['total'] += ntotal
    
    # 计算总体准确率
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # 计算各类别准确率
    category_accuracies = {}
    for category, stats in category_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        category_accuracies[category] = {
            'accuracy': accuracy,
            'correct': stats['correct'],
            'total': stats['total']
        }
    
    return {
        'overall': {
            'accuracy': overall_accuracy,
            'correct': total_correct,
            'total': total_samples,
            'percentage': overall_accuracy * 100
        },
        'categories': category_accuracies,
        'num_samples': len(data)
    }


def print_results(results):
    """打印统计结果"""
    print("=" * 60)
    print("评估结果统计")
    print("=" * 60)
    
    # 总体结果
    overall = results['overall']
    print(f"\n【总体准确率】")
    print(f"  正确数: {overall['correct']}/{overall['total']}")
    print(f"  准确率: {overall['accuracy']:.4f} ({overall['percentage']:.2f}%)")
    print(f"  样本数: {results['num_samples']}")
    
    # 按类别统计
    if results['categories']:
        print(f"\n【按类别统计】")
        for category, stats in sorted(results['categories'].items()):
            print(f"  {category}:")
            print(f"    正确数: {stats['correct']}/{stats['total']}")
            print(f"    准确率: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")
    
    print("=" * 60)


def save_results(results, output_file):
    """保存结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='统计评估指标的准确率')
    parser.add_argument('metrics_file', type=str, help='metrics JSON文件路径')
    parser.add_argument('--output', '-o', type=str, help='输出结果文件路径（可选）')
    parser.add_argument('--quiet', '-q', action='store_true', help='安静模式，不打印详细信息')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.metrics_file).exists():
        print(f"错误: 文件不存在: {args.metrics_file}")
        return 1
    
    # 计算准确率
    results = calculate_accuracy(args.metrics_file)
    
    # 打印结果
    if not args.quiet:
        print_results(results)
    else:
        print(f"准确率: {results['overall']['percentage']:.2f}%")
    
    # 保存结果
    if args.output:
        save_results(results, args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())

