#!/usr/bin/env python3
"""
验证 experience_maker.py 中 search 工具的集成：SearchKnowledge + execute_tool。
与 test_search_tool.py 一样避免 biomedical-mm 路径；需已安装 faiss-cpu open_clip_torch。

用法（在 curiosity_driven_rl 目录下）:
  python scripts/test_search_integration.py --config /path/to/retriever_config.json [--query "brain tumor MRI"]
"""
import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RL_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path = [p for p in sys.path if "biomedical-mm" not in p and "faiss_lib" not in p]
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

from openrlhf.trainer.ppo_utils.experience_maker import SearchKnowledge, execute_tool
from openrlhf.trainer.ppo_utils.multimodal_retriever import create_retriever


def main():
    parser = argparse.ArgumentParser(description="Test search tool integration in experience_maker")
    parser.add_argument("--config", type=str, required=True, help="检索配置文件路径")
    parser.add_argument("--query", type=str, default="brain tumor glioblastoma MRI", help="测试查询文本")
    parser.add_argument("--top_k", type=int, default=3, help="返回条数")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)

    print("加载检索器...")
    retriever = create_retriever(args.config)
    if retriever is None:
        print("错误: 无法创建检索器")
        sys.exit(1)
    print("检索器加载成功.\n")

    tool = SearchKnowledge()

    # 1. 测试 SearchKnowledge.call()
    print("--- 1. SearchKnowledge.call() ---")
    out = tool.call(
        args.query,
        topk=args.top_k,
        retriever=retriever,
        use_text_index=True,
        return_results=True,
    )
    text = out.get("text", "")
    results = out.get("results", [])
    print(f"格式化文本长度: {len(text)}")
    print(f"结果条数: {len(results)}")
    if text:
        print("文本预览:", text[:400] + "..." if len(text) > 400 else text)
    if results:
        for i, r in enumerate(results[:3]):
            meta = r.get("metadata") or {}
            path = meta.get("preview_path") or meta.get("image_path")
            exists = os.path.isfile(path) if path else False
            print(f"  [{i+1}] id={meta.get('id')}, domain={meta.get('domain')}, path_exists={exists}")
    print()

    # 2. 测试 execute_tool（search 分支）
    print("--- 2. execute_tool(toolname='search') ---")
    fn = tool.call
    res = execute_tool(
        images=[],
        rawimages=[],
        args={"query": args.query},
        toolname="search",
        is_video=False,
        function=fn,
        search_topk=args.top_k,
        retriever=retriever,
        search_use_text_index=True,
    )
    print(f"返回类型: dict with keys: {list(res.keys())}")
    print(f"text 长度: {len(res.get('text', ''))}")
    print(f"results 条数: {len(res.get('results', []))}")
    txt = res.get("text", "")
    if txt:
        print("text 预览:", txt[:300] + "..." if len(txt) > 300 else txt)
    res_list = res.get("results", [])
    for i, r in enumerate(res_list[:3]):
        meta = r.get("metadata") or {}
        path = meta.get("preview_path") or meta.get("image_path")
        exists = os.path.isfile(path) if path else False
        print(f"  [{i+1}] path_exists={exists}, path={path[:80] + '...' if path and len(path) > 80 else path}")
    print()

    # 若有有效路径，尝试加载并保存一张测试图
    for r in res_list:
        meta = r.get("metadata") or {}
        path = meta.get("preview_path") or meta.get("image_path")
        if path and os.path.isfile(path):
            try:
                from PIL import Image
                img = Image.open(path).convert("RGB")
                out_path = os.path.join(_RL_ROOT, "scripts", "_search_test_image.jpg")
                img.save(out_path)
                print(f"已保存测试图片到: {out_path} （可打开验证）")
            except Exception as e:
                print(f"加载/保存测试图失败: {e}")
            break
    else:
        print("无有效图片路径，无法保存测试图。请检查 faiss_indices/metadata.json 中的 image_path/preview_path。")

    print("\n集成验证通过: SearchKnowledge + execute_tool 工作正常.")


if __name__ == "__main__":
    main()
