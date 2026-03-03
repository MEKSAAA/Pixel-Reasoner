#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""统计指定目录下 JSON 日志中的工具调用次数，并从 testjson 的 gt_answer 计算准确率。支持三种统计口径，并包含详细的子目录统计（含解析错误标注）。
口径二：有 conv.json 即统计；但凡有 conv 却未解析出 <answer> 中 anomaly_present 的均算解析失败，按错计入。"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from sklearn.metrics import roc_auc_score, f1_score
except ImportError:
    roc_auc_score = None
    f1_score = None

TOOL_NAMES = ("crop_image_normalized", "search","query_image")
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_NAME_PATTERNS = {
    name: re.compile(
        r"(?:\\\"|\")name(?:\\\"|\")\s*:\s*(?:\\\"|\")"
        + re.escape(name)
        + r"(?:\\\"|\")"
    )
    for name in TOOL_NAMES
}

# 从回答中解析 <answer>{"anomaly_present": true/false, ...}</answer>
# 用 (.*?) 取完整标签内容再 json.loads，避免 \{.*?\} 在第一个 } 处截断多字段 JSON
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

# 子文件夹命名：test_<类别>_<id>，例如 test_brain_8, test_chest_15631
SAMPLE_DIR_PATTERN = re.compile(r"^test_(.+)_\d+$")

# 总体统计两大组：brain 单独一大类，其余四类（mvtec/visa/loco/goodsad）为另一大类
CATEGORY_GROUP_BRAIN = ("brain",)
CATEGORY_GROUP_OTHER_FOUR = ("mvtec", "visa", "loco", "goodsad")

def source_from_item(item: dict) -> str:
    """根据 test json 条目的 id 或 image 路径返回数据集来源。"""
    path = (item.get("id") or item.get("image") or "") if isinstance(item, dict) else ""
    if not isinstance(path, str):
        return "unknown"
    path_lower = path.replace("\\", "/").lower()
    if "brain" in path_lower:
        return "brain"
    if "mvtec-loco" in path_lower or "/loco/" in path_lower:
        return "loco"
    if "goodsad" in path_lower:
        return "goodsad"
    if "ds-mvtec" in path_lower or "mvtec" in path_lower:
        return "mvtec"
    if "visa" in path_lower:
        return "visa"
    return "unknown"

def count_tools_in_text(text: str) -> Dict[str, int]:
    counts = {name: 0 for name in TOOL_NAMES}
    blocks = TOOL_CALL_PATTERN.findall(text)
    if not blocks:
        return counts
    for block in blocks:
        for name, pattern in TOOL_NAME_PATTERNS.items():
            counts[name] += len(pattern.findall(block))
    return counts

def merge_counts(accumulator: Dict[str, int], delta: Dict[str, int]) -> None:
    for name in TOOL_NAMES:
        accumulator[name] += delta[name]

def count_tools_in_data(data) -> Dict[str, int]:
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
                merge_counts(counts, count_tools_in_text(node))
    walk(data)
    return counts

def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()

def scan_json_file(path: str) -> Dict[str, int]:
    try:
        text = read_text(path)
    except OSError:
        return {name: 0 for name in TOOL_NAMES}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return count_tools_in_text(text)
    counts = count_tools_in_data(data)
    if sum(counts.values()) == 0 and "<tool_call>" in text:
        return count_tools_in_text(text)
    return counts

def iter_json_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(".json"):
                yield os.path.join(dirpath, filename)

def category_from_dirname(dirname: str) -> Optional[str]:
    m = SAMPLE_DIR_PATTERN.match(dirname)
    return m.group(1) if m else None

def _get_pred_anomaly_from_messages(messages: list) -> Optional[bool]:
    pred = None
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
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
        try:
            ans = json.loads(m.group(1))
            if "anomaly_present" in ans:
                pred = bool(ans["anomaly_present"])
                break
        except (json.JSONDecodeError, TypeError):
            continue
    return pred

def parse_conv_for_pred(conv_path: str) -> Optional[bool]:
    try:
        text = read_text(conv_path)
        data = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None
    try:
        messages = data[0] if isinstance(data, list) and len(data) > 0 else data
    except (TypeError, IndexError, KeyError):
        # data 为 None、空列表、或 dict 等非常规结构时避免抛错，视为解析失败
        return None
    if not isinstance(messages, list):
        return None
    try:
        return _get_pred_anomaly_from_messages(messages)
    except (TypeError, KeyError, IndexError, json.JSONDecodeError):
        # 内容结构异常或 JSON 片段解析失败时仍算“解析失败”，不抛错
        return None

def load_test_items(test_json_path: str) -> Dict[str, dict]:
    text = read_text(test_json_path)
    data = json.loads(text)
    items: Dict[str, dict] = {}
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict) and isinstance(obj.get("qid"), str):
                items[obj["qid"]] = obj
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(v.get("qid"), str):
                items[v["qid"]] = v
            elif isinstance(k, str) and isinstance(v, dict):
                items[k] = v if "qid" in v else {**v, "qid": k}
    else:
        raise ValueError(f"Unsupported test json format: {type(data)}")
    return items

def qid_from_sample_dirname(dirname: str) -> Optional[str]:
    return dirname if SAMPLE_DIR_PATTERN.match(dirname) else None

def compute_metrics(y_true, y_pred):
    """计算 Acc, F1, AUROC"""
    if not y_true:
        return 0.0, 0.0, 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / len(y_true)
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if f1_score else 0.0
    auroc = 0.0
    if roc_auc_score and len(set(y_true)) > 1:
        try:
            auroc = float(roc_auc_score(y_true, y_pred))
        except:
            auroc = 0.0
    return acc, f1, auroc

def format_counts(counts: Dict[str, int]) -> str:
    return ", ".join(f"{name}={counts[name]}" for name in TOOL_NAMES)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="eval 目录")
    parser.add_argument("--test-json", required=True, help="test json 路径")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--per-file", action="store_true", help="输出每个样本的详细结果")
    args = parser.parse_args()

    if not os.path.isdir(args.root) or not os.path.isfile(args.test_json):
        print("路径错误")
        return

    test_items = load_test_items(args.test_json)
    root_abs = os.path.abspath(args.root)
    
    # 收集所有 conv.json 路径
    qid_to_conv: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(args.root):
        sample_dir = os.path.basename(dirpath)
        qid_here = qid_from_sample_dirname(sample_dir)
        if qid_here and "conv.json" in filenames:
            qid_to_conv[qid_here] = os.path.join(dirpath, "conv.json")

    # 三种口径的数据收集
    modes_data = {
        "mode1": defaultdict(lambda: ([], [])),
        "mode2": defaultdict(lambda: ([], [])),
        "mode3": defaultdict(lambda: ([], []))
    }
    
    # 用于详细子目录统计
    sample_results = {} # qid -> {counts, status_str, top_dir}
    per_dir_summary = defaultdict(lambda: {"counts": {name: 0 for name in TOOL_NAMES}, "correct": 0, "total": 0, "parse_error": 0})

    for qid, item in test_items.items():
        if not isinstance(item, dict) or "gt_answer" not in item:
            continue
        gt = int(bool(item.get("gt_answer")))
        category = category_from_dirname(qid) or source_from_item(item)
        conv_path = qid_to_conv.get(qid)
        try:
            pred = parse_conv_for_pred(conv_path) if conv_path else None
            counts = scan_json_file(conv_path) if conv_path else {name: 0 for name in TOOL_NAMES}
        except Exception:
            # conv 存在但读取/解析过程抛错时，仍按“有文件、解析失败”处理，保证进入口径二
            pred = None
            counts = {name: 0 for name in TOOL_NAMES}

        # Mode 1: 解析成功
        if pred is not None:
            modes_data["mode1"][category][0].append(gt)
            modes_data["mode1"][category][1].append(int(pred))

        # Mode 2: 有 conv.json 即统计；但凡有 conv 却未解析出 <answer> 中 anomaly_present 的均算解析失败，按错算
        if conv_path:
            modes_data["mode2"][category][0].append(gt)
            modes_data["mode2"][category][1].append(int(pred) if pred is not None else 1 - gt)

        # Mode 3: 全部样本
        modes_data["mode3"][category][0].append(gt)
        modes_data["mode3"][category][1].append(int(pred) if pred is not None else 1 - gt)

        # 详细统计逻辑
        if conv_path:
            rel_dir = os.path.relpath(os.path.dirname(conv_path), root_abs)
            top_dir = rel_dir.split(os.sep)[0] if os.sep in rel_dir else rel_dir

            # 有 conv 但未解析出 anomaly_present 的均标为解析失败，仍计入口径二
            status_str = ""
            if pred is None:
                status_str = "解析错误"
                per_dir_summary[top_dir]["parse_error"] += 1
            elif pred == bool(gt):
                status_str = "正确"
                per_dir_summary[top_dir]["correct"] += 1
            else:
                status_str = "错误"

            sample_results[qid] = {"counts": counts, "status_str": status_str, "top_dir": top_dir}

            # 按一级子目录汇总
            for name in TOOL_NAMES:
                per_dir_summary[top_dir]["counts"][name] += counts[name]
            per_dir_summary[top_dir]["total"] += 1

    # 生成报告
    report_lines = []
    
    def add_mode_report(mode_title, mode_key):
        report_lines.append(f"【口径：{mode_title}】")
        data = modes_data[mode_key]
        all_cats = sorted(data.keys())
        total_y_t, total_y_p = [], []
        for cat in all_cats:
            total_y_t.extend(data[cat][0])
            total_y_p.extend(data[cat][1])
        t_acc, t_f1, t_auroc = compute_metrics(total_y_t, total_y_p)
        report_lines.append(f"  总体: 样本={len(total_y_t)}, Acc={t_acc:.4f}, F1={t_f1:.4f}, AUROC={t_auroc:.4f}")
        for cat in all_cats:
            acc, f1, auroc = compute_metrics(data[cat][0], data[cat][1])
            report_lines.append(f"  - {cat}: 样本={len(data[cat][0])}, Acc={acc:.4f}, F1={f1:.4f}, AUROC={auroc:.4f}")
        
        brain_metrics = [compute_metrics(data[c][0], data[c][1]) for c in CATEGORY_GROUP_BRAIN if c in data]
        other_metrics = [compute_metrics(data[c][0], data[c][1]) for c in CATEGORY_GROUP_OTHER_FOUR if c in data]
        if brain_metrics:
            report_lines.append(f"  Brain 大类平均 Acc: {sum(m[0] for m in brain_metrics)/len(brain_metrics):.4f}")
        if other_metrics:
            report_lines.append(f"  其他四类平均 Acc: {sum(m[0] for m in other_metrics)/len(other_metrics):.4f}")
        report_lines.append("")

    add_mode_report("1. 解析成功 (能解析出 <answer>)", "mode1")
    add_mode_report("2. 文件存在 (有 conv.json 即统计；未解析出 <answer> 中 anomaly_present 的算解析失败、按错)", "mode2")
    add_mode_report("3. 全部样本 (包含未生成文件的样本，算错)", "mode3")

    # 工具统计
    total_tool_counts = {name: 0 for name in TOOL_NAMES}
    for qid in sample_results:
        for name in TOOL_NAMES:
            total_tool_counts[name] += sample_results[qid]["counts"][name]
    
    report_lines.append("扫描目录: " + args.root)
    report_lines.append(f"发现 JSON 文件数量: {len(qid_to_conv)}")
    report_lines.append("总调用次数:")
    for name in TOOL_NAMES:
        report_lines.append(f"  {name}: {total_tool_counts[name]}")
    report_lines.append(f"  总计: {sum(total_tool_counts.values())}")
    report_lines.append("")

    # 按一级子目录统计
    report_lines.append("按一级子目录统计:")
    for top_dir in sorted(per_dir_summary.keys()):
        s = per_dir_summary[top_dir]
        acc_str = f"{s['correct']}/{s['total']} ({s['correct']/s['total']*100:.2f}%)" if s['total'] > 0 else "0/0"
        parse_err_str = f" | 解析错误: {s['parse_error']}" if s['parse_error'] > 0 else ""
        report_lines.append(f"- {top_dir}: {format_counts(s['counts'])} | 判断正确: {acc_str}{parse_err_str}")
        
        # 如果开启了 --per-file，输出该目录下每个样本的详细情况
        if args.per_file:
            for qid, res in sorted(sample_results.items()):
                if res["top_dir"] == top_dir:
                    report_lines.append(f"  - {qid}: {format_counts(res['counts'])} | {res['status_str']}")

    report = "\n".join(report_lines)
    print(report)

    output_path = args.output or os.path.join(os.path.dirname(os.path.abspath(args.root)), "multi_mode_summary.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n结果已写入: {output_path}")

if __name__ == "__main__":
    main()