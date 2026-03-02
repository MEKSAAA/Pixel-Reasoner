#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""统计指定目录下 JSON 日志中的工具调用次数，并从 testjson 的 gt_answer 计算准确率。"""

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
ANSWER_PATTERN = re.compile(r"<answer>\s*(\{.*?\})\s*</answer>", re.DOTALL)

# 子文件夹命名：test_<类别>_<id>，例如 test_brain_8, test_chest_15631
SAMPLE_DIR_PATTERN = re.compile(r"^test_(.+)_\d+$")

# 总体统计两大组：brain 单独一大类，其余四类（mvtec/visa/loco/goodsad）为另一大类
CATEGORY_GROUP_BRAIN = ("brain",)
CATEGORY_GROUP_OTHER_FOUR = ("mvtec", "visa", "loco", "goodsad")

# 从 qid 对应条目的 id/image 路径推断数据集：brain / mvtec / visa / loco / goodsad
def source_from_item(item: dict) -> str:
    """根据 test json 条目的 id 或 image 路径返回数据集来源。"""
    path = (item.get("id") or item.get("image") or "") if isinstance(item, dict) else ""
    if not isinstance(path, str):
        return "unknown"
    path_lower = path.replace("\\", "/").lower()
    # 顺序敏感：先匹配 brain，再匹配 MVTec-LOCO/LOCO、GoodsAD、DS-MVTec、VisA
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


# -----------------------------
# 工具调用统计（保留你的原逻辑）
# -----------------------------
def count_tools_in_text(text: str) -> Dict[str, int]:
    """返回文本中各工具名称的出现次数。"""
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
    """读取文件内容，优先使用 UTF-8，失败则回退到 Latin-1。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def scan_json_file(path: str) -> Dict[str, int]:
    """扫描单个 JSON 文件，统计工具调用次数。"""
    try:
        text = read_text(path)
    except OSError as err:
        print(f"无法读取文件 {path}: {err}", file=sys.stderr)
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
    """遍历目录下的所有 JSON 文件路径。"""
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(".json"):
                yield os.path.join(dirpath, filename)


def category_from_dirname(dirname: str) -> Optional[str]:
    """从样本目录名提取类别，如 test_brain_8 -> brain。"""
    m = SAMPLE_DIR_PATTERN.match(dirname)
    return m.group(1) if m else None


# -----------------------------
# 从 conv.json 解析预测 pred
# -----------------------------
def _get_pred_anomaly_from_messages(messages: list) -> Optional[bool]:
    """从最后一条含 <answer> 的 assistant 消息中解析 anomaly_present。"""
    pred = None
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not content:
            continue

        text_parts = []
        # 兼容 content 为 list[{"type":"text","text":...}] 或直接字符串
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
    """解析 conv.json，返回 pred_anomaly；无法解析返回 None。"""
    try:
        text = read_text(conv_path)
        data = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None

    # 兼容：data 可能是 [messages] 或直接 messages
    messages = data[0] if isinstance(data, list) and len(data) > 0 else data
    if not isinstance(messages, list):
        return None

    return _get_pred_anomaly_from_messages(messages)


# -----------------------------
# testjson 读取 & 缺失导出
# -----------------------------
def load_test_items(test_json_path: str) -> Dict[str, dict]:
    """
    读取 test json，返回 {qid: item}。
    兼容两种常见格式：
      1) list[{"qid": "...", ...}, ...]
      2) dict[qid] = {...} 或 dict[...]= {"qid": "...", ...}
    """
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
    """test_<cat>_<id> 这种目录名，qid 就等于目录名。"""
    return dirname if SAMPLE_DIR_PATTERN.match(dirname) else None


def compute_accuracy_from_testjson(
    eval_root: str,
    test_json_path: str,
) -> Tuple[
    Dict[str, Dict],
    Dict[str, Dict],
    int,
    int,
    Dict[str, Tuple[List[int], List[int]]],
    Optional[float],
]:
    """
    用 testjson 的 gt_answer 作为 GT，解析 eval_root 下 conv.json 得到 pred，统计准确率。

    返回：
      (category_stats, topdir_stats, total_correct, total_evaluated, category_auroc_data, overall_f1)
      category_auroc_data: {cat: (y_true_list, y_score_list)} 用于计算 AUROC
      overall_f1: 总体 F1（无 sklearn 或无可评测样本时为 None）
    """
    test_items = load_test_items(test_json_path)

    category_correct: Dict[str, int] = defaultdict(int)
    category_total: Dict[str, int] = defaultdict(int)
    topdir_correct: Dict[str, int] = defaultdict(int)
    topdir_total: Dict[str, int] = defaultdict(int)
    category_auroc_data: Dict[str, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))

    total_correct = 0
    total_evaluated = 0

    root_abs = os.path.abspath(eval_root)

    for dirpath, dirnames, filenames in os.walk(eval_root):
        sample_dir = os.path.basename(dirpath)
        qid_here = qid_from_sample_dirname(sample_dir)
        if qid_here is None:
            continue

        if "conv.json" not in filenames:
            continue

        conv_path = os.path.join(dirpath, "conv.json")
        pred = parse_conv_for_pred(conv_path)
        if pred is None:
            continue

        item = test_items.get(qid_here)
        if item is None:
            continue

        gt = bool(item.get("gt_answer"))
        total_evaluated += 1

        # 类别：优先从样本目录名取（如 test_brain_8 -> brain），否则从条目 id 路径取（mvtec/visa/loco/goodsad）
        category = category_from_dirname(sample_dir)
        if category is None:
            category = source_from_item(item)
        category_total[category] += 1
        if gt == pred:
            total_correct += 1
            category_correct[category] += 1

        # 收集 AUROC 数据：(y_true, y_score)，pred 作为 score（0/1）
        category_auroc_data[category][0].append(int(gt))
        category_auroc_data[category][1].append(int(pred))

        # topdir（一级子目录）
        rel = os.path.relpath(os.path.abspath(conv_path), root_abs)
        parts = rel.split(os.sep)
        top_dir = parts[0] if len(parts) > 1 else "."
        topdir_total[top_dir] += 1
        if gt == pred:
            topdir_correct[top_dir] += 1

    category_stats: Dict[str, Dict] = {}
    for cat in sorted(category_total.keys()):
        c, t = category_correct[cat], category_total[cat]
        category_stats[cat] = {"correct": c, "total": t, "accuracy": c / t if t else 0.0}
        # 按数据集(mvtec/visa/loco/goodsad)计算 F1（二分类 anomaly_present）
        y_true, y_pred = category_auroc_data.get(cat, ([], []))
        if f1_score is not None and len(y_true) > 0:
            category_stats[cat]["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        else:
            category_stats[cat]["f1"] = None

    topdir_stats: Dict[str, Dict] = {}
    for d in sorted(topdir_total.keys()):
        c, t = topdir_correct[d], topdir_total[d]
        topdir_stats[d] = {"correct": c, "total": t, "accuracy": c / t if t else 0.0}

    # 转为普通 dict，方便调用方使用
    auroc_data = {k: (list(v0), list(v1)) for k, (v0, v1) in category_auroc_data.items()}

    # 总体 F1
    all_y_true: List[int] = []
    all_y_pred: List[int] = []
    for v0, v1 in auroc_data.values():
        all_y_true.extend(v0)
        all_y_pred.extend(v1)
    overall_f1: Optional[float] = None
    if f1_score is not None and len(all_y_true) > 0:
        overall_f1 = float(f1_score(all_y_true, all_y_pred, zero_division=0))

    return (
        category_stats,
        topdir_stats,
        total_correct,
        total_evaluated,
        auroc_data,
        overall_f1,
    )


def format_counts(counts: Dict[str, int]) -> str:
    return ", ".join(f"{name}={counts[name]}" for name in TOOL_NAMES)


def judge_conv_correctness(
    conv_path: str,
    test_items: Dict[str, dict],
) -> Tuple[str, Optional[bool], Optional[bool]]:
    """
    对单个 conv.json 输出判断结果：
      status: "CORRECT" / "WRONG" / "PRED_NA" / "GT_NA" / "NOT_SAMPLE"
      gt/pred: 可选返回
    """
    # conv.json 所在目录名就是 qid：test_xxx_yyy
    qid = qid_from_sample_dirname(os.path.basename(os.path.dirname(conv_path)))
    if qid is None:
        return "NOT_SAMPLE", None, None

    item = test_items.get(qid)
    if item is None or "gt_answer" not in item:
        return "GT_NA", None, None

    gt = bool(item["gt_answer"])
    pred = parse_conv_for_pred(conv_path)
    if pred is None:
        return "PRED_NA", gt, None

    return ("CORRECT" if gt == pred else "WRONG"), gt, pred



# -----------------------------
# main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="统计目录下 JSON 文件中的工具调用次数，并用 testjson 计算 accuracy。")
    parser.add_argument("root", help="包含日志的根目录（eval目录）")
    parser.add_argument("--per-file", action="store_true", help="输出每个 JSON 文件的统计信息")
    parser.add_argument("--output", help="将统计结果写入指定 txt 文件；默认写入 parent/tool_usage_summary.txt")
    parser.add_argument("--test-json", required=True, help="test json 路径（必须包含 qid 与 gt_answer），用于统计准确率")

    args = parser.parse_args()

    test_items: Dict[str, dict] = {}
    if args.test_json and os.path.isfile(args.test_json):
        test_items = load_test_items(args.test_json)


    if not os.path.isdir(args.root):
        parser.error(f"路径不存在或不是目录: {args.root}")

    json_files = sorted(iter_json_files(args.root))
    if not json_files:
        print(f"未在目录 {args.root} 下找到任何 JSON 文件。")
        return

    # 统计工具调用
    totals = {name: 0 for name in TOOL_NAMES}
    per_file = []
    per_dir: Dict[str, Dict[str, int]] = defaultdict(lambda: {name: 0 for name in TOOL_NAMES})

    for path in json_files:
        counts = scan_json_file(path)
        for name in TOOL_NAMES:
            totals[name] += counts[name]

        rel_path = os.path.relpath(path, args.root)
        per_file.append((rel_path, counts))

        top_dir = rel_path.split(os.sep)[0] if os.sep in rel_path else "."
        dir_counts = per_dir[top_dir]
        for name in TOOL_NAMES:
            dir_counts[name] += counts[name]

    total_calls = sum(totals.values())

    lines = [
        f"扫描目录: {args.root}",
        f"发现 JSON 文件数量: {len(json_files)}",
        "总调用次数:",
    ]
    for name in TOOL_NAMES:
        lines.append(f"  {name}: {totals[name]}")
    lines.append(f"  总计: {total_calls}")

    if args.per_file:
        lines.append("按一级子目录统计（各样本）:")
        # 按 topdir 分组
        by_topdir: Dict[str, List[Tuple[str, Dict[str, int]]]] = defaultdict(list)
        for rel_path, counts in per_file:
            top_dir = rel_path.split(os.sep)[0] if os.sep in rel_path else "."
            by_topdir[top_dir].append((rel_path, counts))
        for top_dir in sorted(by_topdir.keys()):
            lines.append(f"- {top_dir}:")
            for rel_path, counts in sorted(by_topdir[top_dir]):
                suffix = ""
                # 只对样本的 conv.json 做 correctness 标注
                if rel_path.endswith(os.sep + "conv.json") and test_items:
                    abs_path = os.path.join(args.root, rel_path)
                    status, gt, pred = judge_conv_correctness(abs_path, test_items)
                    if status == "CORRECT":
                        suffix = " | 正确"
                    elif status == "WRONG":
                        suffix = " | 错误"
                    elif status == "PRED_NA":
                        suffix = " | 无法解析"
                    elif status == "GT_NA":
                        suffix = " | 无GT"
                lines.append(f"  - {rel_path}: {format_counts(counts)}{suffix}")


    # 计算准确率（必须提供 --test-json）
    topdir_stats: Dict[str, Dict] = {}
    if not os.path.isfile(args.test_json):
        lines.append("")
        lines.append(f"【准确率】跳过：--test-json 不存在: {args.test_json}")
    else:
        (
            category_stats,
            topdir_stats,
            total_correct,
            total_evaluated,
            category_auroc_data,
            overall_f1,
        ) = compute_accuracy_from_testjson(args.root, args.test_json)

        lines.append("")
        lines.append("【按数据集(brain/mvtec/visa/loco/goodsad)准确率】（GT= testjson 的 gt_answer；Pred= conv.json 的 <answer>）")
        if total_evaluated > 0:
            overall_acc = total_correct / total_evaluated
            lines.append(f"  总体: {total_correct}/{total_evaluated} = {overall_acc:.4f} ({overall_acc*100:.2f}%)")
        else:
            lines.append("  总体: 0/0（没有成功解析出可评测样本）")

        for cat in sorted(category_stats.keys()):
            s = category_stats[cat]
            lines.append(f"  {cat}: {s['correct']}/{s['total']} = {s['accuracy']:.4f} ({s['accuracy']*100:.2f}%)")
        # 两大类平均准确率
        brain_acc_list = [category_stats[c]["accuracy"] for c in CATEGORY_GROUP_BRAIN if c in category_stats and category_stats[c]["total"] > 0]
        other_acc_list = [category_stats[c]["accuracy"] for c in CATEGORY_GROUP_OTHER_FOUR if c in category_stats and category_stats[c]["total"] > 0]
        lines.append("")
        lines.append("【两大类平均准确率】")
        if brain_acc_list:
            lines.append(f"  Brain 大类平均: {sum(brain_acc_list) / len(brain_acc_list):.4f}")
        else:
            lines.append("  Brain 大类平均: N/A")
        if other_acc_list:
            lines.append(f"  其他四类(mvtec/visa/loco/goodsad)平均: {sum(other_acc_list) / len(other_acc_list):.4f}")
        else:
            lines.append("  其他四类(mvtec/visa/loco/goodsad)平均: N/A")

        # 按数据集及总体输出 F1
        if f1_score is None:
            lines.append("")
            lines.append("【按数据集(brain/mvtec/visa/loco/goodsad) / 总体 F1】跳过：未安装 sklearn（pip install scikit-learn）")
        else:
            lines.append("")
            lines.append("【按数据集(brain/mvtec/visa/loco/goodsad) F1】")
            for cat in sorted(category_stats.keys()):
                s = category_stats[cat]
                f1_val = s.get("f1")
                if f1_val is not None:
                    lines.append(f"  {cat}: {f1_val:.4f}")
                else:
                    lines.append(f"  {cat}: N/A")
            if overall_f1 is not None:
                lines.append(f"  总体: {overall_f1:.4f}")
            else:
                lines.append("  总体: N/A")
            # 两大类平均：brain 一大类，其余四类(mvtec/visa/loco/goodsad)一大类
            brain_f1_list = [category_stats[c]["f1"] for c in CATEGORY_GROUP_BRAIN if c in category_stats and category_stats[c].get("f1") is not None]
            other_f1_list = [category_stats[c]["f1"] for c in CATEGORY_GROUP_OTHER_FOUR if c in category_stats and category_stats[c].get("f1") is not None]
            lines.append("")
            lines.append("【两大类平均 F1】")
            if brain_f1_list:
                lines.append(f"  Brain 大类平均: {sum(brain_f1_list) / len(brain_f1_list):.4f}")
            else:
                lines.append("  Brain 大类平均: N/A")
            if other_f1_list:
                lines.append(f"  其他四类(mvtec/visa/loco/goodsad)平均: {sum(other_f1_list) / len(other_f1_list):.4f}")
            else:
                lines.append("  其他四类(mvtec/visa/loco/goodsad)平均: N/A")

        # 计算并按数据集输出 AUROC（需要 sklearn）
        if roc_auc_score is None:
            lines.append("")
            lines.append("【按数据集(brain/mvtec/visa/loco/goodsad) AUROC】跳过：未安装 sklearn（pip install scikit-learn）")
        elif category_auroc_data:
            lines.append("")
            lines.append("【按数据集(brain/mvtec/visa/loco/goodsad) AUROC】")
            cat_auroc_values: Dict[str, Optional[float]] = {}
            for cat in sorted(category_auroc_data.keys()):
                y_true, y_score = category_auroc_data[cat]
                if len(y_true) < 2:
                    lines.append(f"  {cat}: N/A（样本数不足）")
                    cat_auroc_values[cat] = None
                elif len(set(y_true)) < 2:
                    lines.append(f"  {cat}: N/A（仅单一类别）")
                    cat_auroc_values[cat] = None
                else:
                    try:
                        auroc = roc_auc_score(y_true, y_score)
                        lines.append(f"  {cat}: {auroc:.4f}")
                        cat_auroc_values[cat] = auroc
                    except ValueError:
                        lines.append(f"  {cat}: N/A")
                        cat_auroc_values[cat] = None
            # 总体 AUROC
            all_y_true = []
            all_y_score = []
            for yt, ys in category_auroc_data.values():
                all_y_true.extend(yt)
                all_y_score.extend(ys)
            if len(all_y_true) >= 2 and len(set(all_y_true)) >= 2:
                try:
                    overall_auroc = roc_auc_score(all_y_true, all_y_score)
                    lines.append(f"  总体: {overall_auroc:.4f}")
                except ValueError:
                    pass
            # 两大类平均 AUROC
            brain_auroc_list = [cat_auroc_values[c] for c in CATEGORY_GROUP_BRAIN if c in cat_auroc_values and cat_auroc_values[c] is not None]
            other_auroc_list = [cat_auroc_values[c] for c in CATEGORY_GROUP_OTHER_FOUR if c in cat_auroc_values and cat_auroc_values[c] is not None]
            lines.append("")
            lines.append("【两大类平均 AUROC】")
            if brain_auroc_list:
                lines.append(f"  Brain 大类平均: {sum(brain_auroc_list) / len(brain_auroc_list):.4f}")
            else:
                lines.append("  Brain 大类平均: N/A")
            if other_auroc_list:
                lines.append(f"  其他四类(mvtec/visa/loco/goodsad)平均: {sum(other_auroc_list) / len(other_auroc_list):.4f}")
            else:
                lines.append("  其他四类(mvtec/visa/loco/goodsad)平均: N/A")

    # 一级子目录统计：工具调用次数 + 是否判断正确：工具调用次数 + 是否判断正确
    lines.append("")
    lines.append("按一级子目录统计:")
    for dirname in sorted(per_dir):
        line = f"- {dirname}: {format_counts(per_dir[dirname])}"
        if dirname in topdir_stats:
            s = topdir_stats[dirname]
            line += f" | 判断正确: {s['correct']}/{s['total']} ({s['accuracy']*100:.2f}%)"
        lines.append(line)

    report = "\n".join(lines)
    print(report)

    # 输出报告 txt
    if args.output:
        output_path = args.output
    else:
        root_abs = os.path.abspath(args.root)
        parent = os.path.dirname(root_abs) or root_abs
        output_path = os.path.join(parent, "tool_usage_summary.txt")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
    except OSError as err:
        print(f"写入文件 {output_path} 失败: {err}", file=sys.stderr)
    else:
        print(f"结果已写入: {output_path}")


if __name__ == "__main__":
    main()
