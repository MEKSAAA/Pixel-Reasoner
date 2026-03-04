#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 Anomaly Detection QA 准确率脚本
功能：仅统计「Anomaly Detection」类型问题，使用 normal/abnormal 两类准确率的平均值
      (average accuracy of normal and abnormal categories)
"""

import json
import argparse
import re
from pathlib import Path


# 从回答中解析 <answer>X</answer>（X 为纯文本，如 A/B/C/D）
ANSWER_PATTERN = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.DOTALL)

# 工具调用统计（与 count_tool_usage.py 一致）
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
        # 兼容 role/content 或 value 两种结构
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
    """从 conv.json 中提取 <answer>X</answer> 里的 X，逻辑仿照 count_tool_usage 的消息遍历。"""
    try:
        with open(conv_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            return None
        # 兼容：顶层是 [messages] 或直接 messages 列表
        messages = data[0] if isinstance(data[0], list) else data
        if not isinstance(messages, list):
            return None
        return _get_answer_from_messages(messages)
    except Exception:
        pass
    return None


# 仅统计 Anomaly Detection 类型
ANOMALY_DETECTION_TYPE = "Anomaly Detection"
# 正常/异常类别用 gt_answer（bool）：False=正常，True=异常；对错用 question_item['Answer'] 判定


def calculate_qa_accuracy(gt_file, dumps_dir):
    """
    仅统计 Anomaly Detection 问题，计算 normal/abnormal 两类准确率的平均值。
    """
    # 1. 加载真值数据
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    gt_dict = {item['qid']: item for item in gt_data}

    # 只统计 AD：normal / abnormal 两类的 correct & count，以及工具调用
    stats = {
        'normal': {'correct': 0, 'count': 0},
        'abnormal': {'correct': 0, 'count': 0},
        'tool_counts': {name: 0 for name in TOOL_NAMES},
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

        gt_item = gt_dict[qid]
        q_type = gt_item.get('question_item', {}).get('type', "")
        if q_type != ANOMALY_DETECTION_TYPE:
            continue

        conv_json = subfolder / "conv.json"
        if not conv_json.exists():
            continue

        pred_ans = extract_answer(conv_json)
        gt_ans = gt_item['question_item']['Answer'].strip().upper()
        # 对错：与 question_item['Answer'] 比较
        if pred_ans is None:
            is_correct = False
        else:
            is_correct = (pred_ans == gt_ans)

        # 正常/异常类别：用 gt_answer（bool）划分，False=正常，True=异常
        gt_answer = bool(gt_item.get('gt_answer', False))
        if gt_answer:
            stats['abnormal']['count'] += 1
            if is_correct:
                stats['abnormal']['correct'] += 1
        else:
            stats['normal']['count'] += 1
            if is_correct:
                stats['normal']['correct'] += 1

        one_tool = count_tools_in_conv_file(conv_json)
        for name in TOOL_NAMES:
            stats['tool_counts'][name] += one_tool.get(name, 0)

    return stats


def print_report(stats):
    """打印 Anomaly Detection 报告：normal/abnormal 准确率及二者平均值"""
    if not stats:
        print("没有找到有效的评估数据。")
        return

    n = stats['normal']
    a = stats['abnormal']
    total_count = n['count'] + a['count']
    if total_count == 0:
        print("没有找到 Anomaly Detection 样本。")
        return

    acc_normal = n['correct'] / n['count'] if n['count'] > 0 else 0.0
    acc_abnormal = a['correct'] / a['count'] if a['count'] > 0 else 0.0
    avg_accuracy = (acc_normal + acc_abnormal) / 2.0

    print("=" * 70)
    print(f"{'Anomaly Detection QA 统计 (仅 AD，平均准确率)':^60}")
    print("=" * 70)
    print(f"【仅统计: Anomaly Detection】")
    print(f"  样本总数: {total_count} (normal: {n['count']}, abnormal: {a['count']})")
    print("-" * 70)
    print(f"{'Category':<28} | {'Correct/Total':<15} | {'Accuracy':<10}")
    print("-" * 70)
    print(f"{'Normal (gt_answer=False)':<28} | {n['correct']}/{n['count']:<14} | {acc_normal:.2%}")
    print(f"{'Abnormal (gt_answer=True)':<28} | {a['correct']}/{a['count']:<14} | {acc_abnormal:.2%}")
    print("-" * 70)
    print(f"  Average accuracy (normal & abnormal): {avg_accuracy:.2%}")
    print("-" * 70)

    tc = stats.get('tool_counts') or {}
    print(f"【工具调用统计 (仅 AD 样本)】")
    for name in TOOL_NAMES:
        print(f"  {name}: {tc.get(name, 0)}")
    print(f"  合计: {sum(tc.get(name, 0) for name in TOOL_NAMES)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='统计 Anomaly Detection QA 准确率 (normal/abnormal 平均)')
    parser.add_argument('--gt', type=str, default='test_mmad_qa.json', help='真值 JSON 文件路径')
    parser.add_argument('--log_dir', type=str, default='logs/dumps_iter0', help='预测结果根目录')
    parser.add_argument('--output', '-o', type=str, help='结果保存为 JSON 文件')

    args = parser.parse_args()

    results = calculate_qa_accuracy(args.gt, args.log_dir)

    if results:
        print_report(results)

        if args.output:
            n, a = results['normal'], results['abnormal']
            acc_n = n['correct'] / n['count'] if n['count'] > 0 else 0.0
            acc_a = a['correct'] / a['count'] if a['count'] > 0 else 0.0
            save_data = {
                'anomaly_detection_only': True,
                'normal': n,
                'abnormal': a,
                'accuracy_normal': acc_n,
                'accuracy_abnormal': acc_a,
                'average_accuracy': (acc_n + acc_a) / 2.0,
                'tool_counts': results.get('tool_counts', {}),
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"结果已保存至: {args.output}")

if __name__ == '__main__':
    main()