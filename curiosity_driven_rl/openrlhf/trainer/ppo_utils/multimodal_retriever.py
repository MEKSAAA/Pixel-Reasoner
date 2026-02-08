"""
多模态检索器 (CLIP + FAISS)，与 biomedical-mm/test_retrieval.py 同款接口。

用于 Pixel-Reasoner 的 search 工具：按文本查询检索相似样本，返回 metadata（含 text/domain/id 等）。
"""

import json
from typing import List, Dict, Any, Optional


def _format_multimodal_results(results: List[Dict[str, Any]]) -> str:
    """将 MultimodalRetriever 的返回格式化为 <information> 内使用的 Doc i(Title: ...) 文本。"""
    out = []
    for r in results:
        meta = r.get("metadata") or {}
        rank = r.get("rank", len(out) + 1)
        domain = meta.get("domain", "")
        id_ = meta.get("id", "")
        text = meta.get("text", "")
        title = f"{id_} | {domain}".strip(" |") or "Result"
        out.append(f"Doc {rank}(Title: {title}) {text}")
    return "\n".join(out) + "\n" if out else ""


def create_retriever(config_path: str):
    """
    根据配置文件创建 MultimodalRetriever 实例（懒加载 open_clip/faiss）。
    若依赖不可用则返回 None。
    """
    try:
        import numpy as np
        import torch
        import faiss
        from PIL import Image
        import open_clip
    except ImportError as e:
        print(f"[multimodal_retriever] Optional deps not available: {e}")
        return None

    with open(config_path, "r") as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        config["clip_model"],
        pretrained=config["clip_pretrained"],
    )
    model = model.to(device)
    model.eval()

    image_index = faiss.read_index(config["image_index_path"])
    text_index = faiss.read_index(config["text_index_path"])

    with open(config["metadata_path"], "r") as f:
        metadata = json.load(f)

    class _Retriever:
        def __init__(self):
            self.model = model
            self.preprocess = preprocess
            self.device = device
            self.image_index = image_index
            self.text_index = text_index
            self.metadata = metadata

        def retrieve_by_text(self, query_text: str, top_k: int = 5) -> List[Dict]:
            text_tokens = open_clip.tokenize([query_text]).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(text_tokens)
                features /= features.norm(dim=-1, keepdim=True)
                features = features.cpu().numpy()
            distances, indices = self.image_index.search(features, top_k)
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                results.append({
                    "rank": i + 1,
                    "score": float(dist),
                    "metadata": self.metadata[idx],
                })
            return results

        def retrieve_images_by_text_description(self, text_query: str, top_k: int = 5) -> List[Dict]:
            text_tokens = open_clip.tokenize([text_query]).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(text_tokens)
                features /= features.norm(dim=-1, keepdim=True)
                features = features.cpu().numpy()
            distances, indices = self.text_index.search(features, top_k)
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                results.append({
                    "rank": i + 1,
                    "score": float(dist),
                    "metadata": self.metadata[idx],
                })
            return results

    return _Retriever()


def search_with_retriever(
    retriever,
    query: str,
    top_k: int = 5,
    use_text_index: bool = True,
    return_results: bool = False,
):
    """
    使用 MultimodalRetriever 做一次检索。
    return_results=False: 仅返回格式化字符串（供 <information> 使用）。
    return_results=True: 返回 (formatted_str, results)，results 含 metadata.preview_path 等，用于附带图片。
    """
    if not retriever or not (query or "").strip():
        return ("", []) if return_results else ""
    try:
        if use_text_index:
            results = retriever.retrieve_images_by_text_description((query or "").strip(), top_k=top_k)
        else:
            results = retriever.retrieve_by_text((query or "").strip(), top_k=top_k)
        text = _format_multimodal_results(results)
        if return_results:
            return (text, results)
        return text
    except Exception as e:
        err = f"[Search error: {e}]"
        return (err, []) if return_results else err
