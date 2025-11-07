#!/usr/bin/env python3

"""统计指定目录下 JSON 日志中的工具调用次数。"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Iterable


TOOL_NAMES = ("crop_image_normalized", "query_image")
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_NAME_PATTERNS = {
    name: re.compile(
        r"(?:\\\"|\")name(?:\\\"|\")\s*:\s*(?:\\\"|\")"
        + re.escape(name)
        + r"(?:\\\"|\")"
    )
    for name in TOOL_NAMES
}


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


def format_counts(counts: Dict[str, int]) -> str:
    return ", ".join(f"{name}={counts[name]}" for name in TOOL_NAMES)


def main() -> None:
    parser = argparse.ArgumentParser(description="统计目录下 JSON 文件中的工具调用次数。")
    parser.add_argument("root", help="包含日志的根目录")
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="输出每个 JSON 文件的统计信息",
    )
    parser.add_argument(
        "--output",
        help="将统计结果写入指定 txt 文件；默认写入根目录下的 tool_usage_summary.txt",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        parser.error(f"路径不存在或不是目录: {args.root}")

    json_files = sorted(iter_json_files(args.root))

    if not json_files:
        print(f"未在目录 {args.root} 下找到任何 JSON 文件。")
        return

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

    lines.append("按一级子目录统计:")
    for dirname in sorted(per_dir):
        lines.append(f"- {dirname}: {format_counts(per_dir[dirname])}")

    if args.per_file:
        lines.append("按文件统计:")
        for rel_path, counts in per_file:
            lines.append(f"- {rel_path}: {format_counts(counts)}")

    report = "\n".join(lines)
    print(report)

    if args.output:
        output_path = args.output
    else:
        root_abs = os.path.abspath(args.root)
        parent = os.path.dirname(root_abs)
        grandparent = os.path.dirname(parent)
        if not grandparent or grandparent == parent:
            grandparent = parent or root_abs
        output_path = os.path.join(grandparent, "tool_usage_summary.txt")
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


