#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 QA 问题准确率脚本（二分类：A=有异常，B=正常）
功能：匹配 logs 目录下的预测结果与真值文件，计算准确率、F1 及 AUROC 等指标
"""

import json
import argparse
import re
import sys
from pathlib import Path

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


# 从回答中解析 <answer>X</answer>（X 为纯文本，如 A/B/C/D）
ANSWER_PATTERN = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.DOTALL)

# 工具调用统计
TOOL_NAMES = ("crop_image_normalized", "search", "query_image")
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_NAME_PATTERNS = {
    name: re.compile(
        r'(?:\\"|")name(?:\\"|")\s*:\s*(?:\\"|")'
        + re.escape(name)
        + r'(?:\\"|")'
    )
    for name in TOOL_NAMES
}


def _get_answer_from_messages(messages: list) -> str | None:
    """从最后一条含 <answer> 的 assistant 消息中解析答案文本。"""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role is not None and role != "assistant":
            continue
        content = msg.get("content") if "content" in msg else msg.get("value")
        if not content:
            continue

        text_parts = []
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str):
                    text_parts.append(c["text"])
                elif isinstance(c, str):
                    text_parts.append(c)
        elif isinstance(content, str):
            text_parts.append(content)

        text = " ".join(text_parts)
        m = ANSWER_PATTERN.search(text)
        if not m:
            continue
        return m.group(1).strip().upper()
    return None


def count_tools_in_text(text: str) -> dict:
    """返回文本中各工具名称的出现次数。"""
    counts = {name: 0 for name in TOOL_NAMES}
    blocks = TOOL_CALL_PATTERN.findall(text)
    if not blocks:
        return counts
    for block in blocks:
        for name, pattern in TOOL_NAME_PATTERNS.items():
            counts[name] += len(pattern.findall(block))
    return counts


def _merge_tool_counts(accumulator: dict, delta: dict) -> None:
    for name in TOOL_NAMES:
        accumulator[name] += delta[name]


def count_tools_in_data(data) -> dict:
    """递归遍历 data（conv 结构），统计 assistant 消息中的工具调用。"""
    counts = {name: 0 for name in TOOL_NAMES}

    def walk(node, role_context=None):
        current_role = role_context
        if isinstance(node, dict):
            if "role" in node and isinstance(node["role"], str):
                current_role = node["role"]
            for key, value in node.items():
                if key == "role":
                    continue
                walk(value, current_role)
        elif isinstance(node, list):
            for item in node:
                walk(item, current_role)
        elif isinstance(node, str):
            if current_role == "assistant" and "<tool_call>" in node:
                _merge_tool_counts(counts, count_tools_in_text(node))

    walk(data)
    return counts


def count_tools_in_conv_file(conv_path: Path) -> dict:
    """读取 conv.json，统计该对话中各类工具调用次数。"""
    try:
        with open(conv_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {name: 0 for name in TOOL_NAMES}
    if not isinstance(data, list) or len(data) == 0:
        return {name: 0 for name in TOOL_NAMES}
    messages = data[0] if isinstance(data[0], list) else data
    if not isinstance(messages, list):
        return {name: 0 for name in TOOL_NAMES}
    return count_tools_in_data(messages)
    

def extract_answer(conv_file):
    """从 conv.json 中提取 <answer>X</answer> 里的 X。"""
    try:
        with open(conv_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            return None
        messages = data[0] if isinstance(data[0], list) else data
        if not isinstance(messages, list):
            return None
        return _get_answer_from_messages(messages)
    except Exception:
        pass
    return None


def calculate_qa_accuracy(gt_file, dumps_dir):
    """
    计算 QA 准确率
    """
    # 1. 加载真值数据
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    gt_dict = {item['qid']: item for item in gt_data}
    
    # 统计变量：二分类 A=异常(正类) B=正常(负类)
    stats = {
        'total': {'correct': 0, 'count': 0},
        'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
        'tool_counts': {name: 0 for name in TOOL_NAMES},
        'y_true': [],
        'y_score': []
    }

    # 2. 遍历预测文件夹
    dumps_path = Path(dumps_dir)
    if not dumps_path.exists():
        print(f"错误: 文件夹不存在 {dumps_dir}")
        return None

    for subfolder in dumps_path.iterdir():
        if not subfolder.is_dir():
            continue
            
        qid = subfolder.name
        if qid not in gt_dict:
            continue
            
        conv_json = subfolder / "conv.json"
        if not conv_json.exists():
            continue

        # 提取预测值和真值（A=有异常，B=正常）
        pred_ans = extract_answer(conv_json)
        gt_item = gt_dict[qid]
        gt_ans = gt_item['question_item']['Answer'].strip().upper()

        # 核心修改：如果无法解析出预测值，则不计入分母（对齐 Acc 和 F1/AUROC）
        if pred_ans is None:
            continue

        is_correct = (pred_ans == gt_ans)

        # 总体准确率
        stats['total']['count'] += 1
        if is_correct:
            stats['total']['correct'] += 1

        # 二分类混淆矩阵（正类=A=有异常）
        if pred_ans in ('A', 'B') and gt_ans in ('A', 'B'):
            # 收集 AUROC 数据
            stats['y_true'].append(1 if gt_ans == 'A' else 0)
            stats['y_score'].append(1 if pred_ans == 'A' else 0)

            if gt_ans == 'A' and pred_ans == 'A':
                stats['tp'] += 1
            elif gt_ans == 'B' and pred_ans == 'A':
                stats['fp'] += 1
            elif gt_ans == 'A' and pred_ans == 'B':
                stats['fn'] += 1
            else:
                stats['tn'] += 1

        # 工具调用数量
        one_tool = count_tools_in_conv_file(conv_json)
        for name in TOOL_NAMES:
            stats['tool_counts'][name] += one_tool.get(name, 0)

    return stats


def _compute_f1(tp, fp, fn):
    """计算正类（A=有异常）的 Precision、Recall、F1。"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def print_report(stats):
    """打印格式化报告（二分类：A=有异常，B=正常）"""
    if not stats or stats['total']['count'] == 0:
        print("没有找到有效的评估数据。")
        return

    total = stats['total']
    overall_acc = total['correct'] / total['count'] if total['count'] > 0 else 0
    tp, fp, fn, tn = stats['tp'], stats['fp'], stats['fn'], stats['tn']
    precision, recall, f1 = _compute_f1(tp, fp, fn)

    # 计算 AUROC
    auroc = "N/A"
    if roc_auc_score and len(set(stats['y_true'])) > 1:
        auroc = f"{roc_auc_score(stats['y_true'], stats['y_score']):.4f}"
    elif not roc_auc_score:
        auroc = "N/A (sklearn not installed)"

    print("=" * 70)
    print(f"{'QA 评估指标统计报告 (二分类 A=异常 B=正常)':^60}")
    print("=" * 70)
    print(f"【总体统计】")
    print(f"  样本总数 (已排除无法解析): {total['count']}")
    print(f"  正确数量: {total['correct']}")
    print(f"  总体准确率: {overall_acc:.2%}")
    print("-" * 70)
    print(f"【二分类指标（正类=有异常 A）】")
    print(f"  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUROC:     {auroc}")
    print("-" * 70)
    print(f"【工具调用统计 (总计)】")
    tc = stats.get('tool_counts') or {}
    for name in TOOL_NAMES:
        print(f"  {name}: {tc.get(name, 0)}")
    print(f"  合计: {sum(tc.get(name, 0) for name in TOOL_NAMES)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='统计 MMAD QA 任务准确率')
    parser.add_argument('--gt', type=str, required=True, help='真值 JSON 文件路径')
    parser.add_argument('--log_dir', type=str, default='logs/dumps_iter0', help='预测结果根目录')
    parser.add_argument('--output', '-o', type=str, help='结果保存为 JSON 文件')

    args = parser.parse_args()

    results = calculate_qa_accuracy(args.gt, args.log_dir)
    
    if results:
        print_report(results)
        
        if args.output:
            tp, fp, fn, tn = results['tp'], results['fp'], results['fn'], results['tn']
            p, r, f1 = _compute_f1(tp, fp, fn)
            
            # 计算 AUROC 用于保存
            auroc_val = None
            if roc_auc_score and len(set(results['y_true'])) > 1:
                auroc_val = float(roc_auc_score(results['y_true'], results['y_score']))

            save_data = {
                'overall': results['total'],
                'confusion': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
                'precision': p, 'recall': r, 'f1': f1,
                'auroc': auroc_val,
                'tool_counts': results.get('tool_counts', {}),
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"结果已保存至: {args.output}")

if __name__ == '__main__':
    main()