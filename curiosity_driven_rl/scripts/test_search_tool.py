#!/usr/bin/env python3
"""
只测 search 工具：多模态检索 + 格式化，不跑完整 eval。
仅使用 curiosity9 当前环境（不加载 biomedical-mm 的 faiss_lib）；需已安装: pip install faiss-cpu open_clip_torch

用法（在 curiosity_driven_rl 目录下）:
  python scripts/test_search_tool.py --config /path/to/retriever_config.json [--query "brain tumor MRI"] [--top_k 3]
"""
import argparse
import os
import sys

# 只使用 curiosity9 当前环境：不加载 biomedical-mm 的 faiss_lib（避免 NumPy 1.x/2.x 冲突）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RL_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
# 从 sys.path 中移除任何包含 biomedical-mm 或 faiss_lib 的路径，确保 import faiss 用当前环境的 faiss-cpu/faiss-gpu
sys.path = [p for p in sys.path if "biomedical-mm" not in p and "faiss_lib" not in p]
# 只把 ppo_utils 加入 path，直接导入 multimodal_retriever，避免拉取 openrlhf.trainer 整条依赖链
_PPO_UTILS = os.path.abspath(os.path.join(_RL_ROOT, "openrlhf", "trainer", "ppo_utils"))
if _PPO_UTILS not in sys.path:
    sys.path.insert(0, _PPO_UTILS)

from multimodal_retriever import create_retriever, search_with_retriever


def main():
    parser = argparse.ArgumentParser(description="Test search tool (multimodal retriever only)")
    parser.add_argument("--config", type=str, required=True, help="检索配置文件路径 (与 test_retrieval / biomedical-mm 同款)")
    parser.add_argument("--query", type=str, default="brain tumor glioblastoma MRI", help="测试查询文本")
    parser.add_argument("--top_k", type=int, default=3, help="返回条数")
    parser.add_argument("--use_text_index", type=lambda x: x.lower() == "true", default=True, help="True=文本索引, False=图像索引")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)

    print("加载检索器...")
    try:
        retriever = create_retriever(args.config)
    except Exception as e:
        print(f"错误: 创建检索器时异常: {e}")
        print("\n若使用 biomedical-mm 的 faiss_lib（为 NumPy 1.x 编译），当前环境 NumPy 2.x 会不兼容。")
        print("建议: 在 biomedical-mm 目录下用其环境运行 test_retrieval.py 做检索测试；")
        print("      或在本环境临时安装 numpy<2:  pip install 'numpy<2'")
        sys.exit(1)
    if retriever is None:
        print("错误: 无法创建检索器，请检查 open_clip / faiss 等依赖及 config 内容")
        print("若报错与 NumPy / faiss 相关，请用 numpy<2 或在与 biomedical-mm 相同环境中运行。")
        sys.exit(1)
    print("检索器加载成功.\n")

    print(f"查询: {args.query!r}")
    print(f"top_k={args.top_k}, use_text_index={args.use_text_index}\n")

    # 只返回文本
    text_only = search_with_retriever(
        retriever, args.query, top_k=args.top_k, use_text_index=args.use_text_index, return_results=False
    )
    print("--- 仅文本 (return_results=False) ---")
    print(text_only[:500] + "..." if len(text_only) > 500 else text_only)
    print()

    # 返回文本 + 原始 results（含 preview_path）
    text, results = search_with_retriever(
        retriever, args.query, top_k=args.top_k, use_text_index=args.use_text_index, return_results=True
    )
    print("--- 文本 + results (return_results=True) ---")
    print("格式化文本长度:", len(text))
    print("结果条数:", len(results))
    for i, r in enumerate(results):
        meta = r.get("metadata") or {}
        path = meta.get("preview_path") or meta.get("image_path")
        print(f"  [{i+1}] id={meta.get('id')}, domain={meta.get('domain')}, path_exists={os.path.isfile(path) if path else False}")
    print("\n测试通过: search 工具（多模态检索 + 格式化）工作正常。")


if __name__ == "__main__":
    main()
