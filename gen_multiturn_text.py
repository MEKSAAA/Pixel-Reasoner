import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ======================
# Config
# ======================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-CR8zGpArpDAF1IkyE55181D2285f47Ee867244548262977d")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.laozhang.ai/v1")
MODEL_NAME = os.environ.get("VLM_MODEL", "gpt-4o-mini")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "8"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))
WORKERS = int(os.environ.get("VLM_WORKERS", "30"))

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var. Please export OPENAI_API_KEY before running.")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ======================
# Prompts (system + user)
# ======================
SYSTEM_PROMPT = """You are a vision expert specialized in industrial anomaly detection.
You will evaluate whether the given object image is normal or abnormal. If abnormal, select the most fitting anomaly label from the candidate types provided by the user.

Output format:
<think>Explain your detailed visual reasoning.</think><answer>{"anomaly_present": true/false, "top_anomaly": "<label or 'none'>", "visual_descriptions": ["..."]}</answer>
If normal → anomaly_present=false, top_anomaly="none", visual_descriptions=[].
If abnormal → include concise visual phrases for visible cues.

IMPORTANT (about ground truth):
- The user will provide a PRIVATE_GROUND_TRUTH hint. You MUST use it only to ensure the final <answer> is correct.
- You MUST NOT mention or quote that hint in <think>.
- Do NOT use words like: ground truth, GT, label, provided hint, answer key in <think>.
- Give detailed reasoning in every step!!!
- Give detailed visual reasoning when you call tool functions.
- Write detailed visual-only reasoning based on the images (texture, shape, edges, continuity, surface quality, etc.)

# Tools
You should call tool functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

<tools>
[
  {
    "type": "function",
    "function": {
      "name": "crop_image_normalized",
      "description": "Zoom in on the image based on the bounding box coordinates to examine local regions in higher detail.",
      "parameters": {
        "type": "object",
        "properties": {
          "bbox_2d": {
            "type": "array",
            "description": "Normalized coordinates [x1, y1, x2, y2] of the region to crop (values between 0.0 and 1.0).",
            "items": {"type": "number"}
          },
          "target_image": {
            "type": "number",
            "description": "The index of the image to crop (1 for the original target image)."
          }
        },
        "required": ["bbox_2d", "target_image"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "query_image",
      "description": "Retrieve a normal reference image of the same class for comparison. This function does not require any arguments.",
      "parameters": {
        "type": "object",
        "properties": {},
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "search",
      "description": "Search for visual information and generate an extremely detailed visual description for industrial inspection (geometric morphology, contour, texture, color, edges, etc.). Use when you need dense visual characteristics.",
      "parameters": {
        "type": "object",
        "properties": {
          "description": {
            "type": "string",
            "description": "A brief description of a normal or anomalous object to generate detailed visual description for."
          }
        },
        "required": ["description"]
      }
    }
  }
]
</tools>

For each function call, return a JSON object with the function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Tool usage rules in this task:
- If you want to crop, call crop_image_normalized.
- If you want a normal reference, call query_image.
- If you need dense visual description (morphology, texture, edges, etc.), call search with a brief description.
- You may call no tool, one tool, or multiple tools in any order.
- After finishing tool usage, output the final answer in the required XML format.

Policy:
- You are expected to use at least ONE tool before giving the final <answer>, unless the object is perfectly unambiguous from the full image.
- When you decide to use a tool, output ONLY a short rationale + the <tool_call> block (do NOT output <answer> yet).
- You are recommended to call crop_image_normalized and/or query_image before producing the final <answer>. You may also call search for visual details. Order is your choice.
- You should not include two <tool_call> blocks in one message.
- If you want to call two tools, do it in two messages.
- YOU ARE VERY ENCOURAGED TO CALL DIFFERENT TOOLS!!!

"""


def build_user_prompt(sample: Dict) -> str:
    cls = sample.get("class_name", "object")
    anoms = sample.get("anomaly_list", []) or []
    anoms_text = "".join([f"- {a}\n" for a in anoms]) if anoms else "- (none specified)\n"

    gt_answer = bool(sample.get("gt_answer"))
    gt_anom = sample.get("anomaly_type", "none") or "none"
    gt_line = (
        f'PRIVATE_GROUND_TRUTH: anomaly_present=true; top_anomaly="{gt_anom}"'
        if gt_answer
        else 'PRIVATE_GROUND_TRUTH: anomaly_present=false; top_anomaly="none"'
    )

    return (
        f'Evaluate the following image from the class "{cls}".\n\n'
        f"Candidate anomaly types:\n{anoms_text}\n"
        f"{gt_line}\n\n"
        "Decide whether you need tools:\n"
        "- crop_image_normalized: to zoom into the ROI\n"
        "- query_image: to get a normal reference image\n"
        "- search: to get a detailed visual description (morphology, texture, edges, etc.)\n\n"
        "You may use one or more tools in any order. "
        "You should call tools to assist with the user query. "
        "When ready, output exactly:\n"
        "<think>...</think><answer>{...}</answer>"
    )


# ======================
# Tool-call parsing
# ======================
# 支持缺少闭合 </tool_call> 的块（模型有时会漏掉），避免 search 等工具无法被识别
RE_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)(?:\s*</tool_call>|\Z)", re.DOTALL)
RE_ANSWER_TAG = re.compile(r"<answer>\s*\{.*?\}\s*</answer>", re.DOTALL | re.IGNORECASE)


def _fix_tool_call_trailing_braces(s: str) -> str:
    """模型有时多打一个 }，如 search 结尾变成 }}} 应为 }}。去掉末尾多余的 } 再解析。"""
    s = (s or "").strip()
    while s.endswith("}") and not s.endswith("{}"):
        candidate = s[:-1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
        s = candidate
    return s


def _parse_tool_call_json(inner: str) -> Optional[Dict[str, Any]]:
    """从 tool_call 块内部文本解析 JSON。会尝试修正常见的 }}} 多写为 }}。"""
    s = (inner or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s = _fix_tool_call_trailing_braces(s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None


def _extract_search_description_fallback(inner: str) -> str:
    """JSON 解析失败时用正则从块内提取 search 的 description。"""
    if not inner or "search" not in inner or "description" not in inner:
        return ""
    m = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', inner)
    if m:
        return m.group(1).replace('\\"', '"').strip()
    return ""


def _get_bbox_list(sample: Dict) -> List[List[float]]:
    """从 sample 得到 bbox 列表，每个元素为 [x1,y1,x2,y2]。"""
    bbox = sample.get("bbox")
    if not bbox or not isinstance(bbox, list):
        return []
    if len(bbox) > 0 and isinstance(bbox[0], (list, tuple)):
        return [list(b) for b in bbox]
    if len(bbox) == 4:
        return [list(bbox)]
    return []


def _get_crop_image_list(sample: Dict) -> List[str]:
    """从 sample 得到 crop_image 路径列表。"""
    crop = sample.get("crop_image")
    if not crop:
        return []
    if isinstance(crop, list):
        return [str(p) for p in crop]
    return [str(crop)]


def extract_tool_calls(text: str, sample: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    通过字符串匹配检测 tool_call。crop_image_normalized 用 sample 的 bbox 替换，若有多个 bbox 则展开为多次调用。
    """
    sample = sample or {}
    bbox_list = _get_bbox_list(sample)
    crop_list = _get_crop_image_list(sample)
    n_crops = min(len(bbox_list), len(crop_list)) if (bbox_list and crop_list) else (len(bbox_list) or 0)
    if n_crops == 0 and bbox_list:
        n_crops = len(bbox_list)  # 仍展开，apply 里可能用 bbox 占位

    calls = []
    first_crop_expanded = False
    for m in RE_TOOL_CALL_BLOCK.finditer(text):
        inner = (m.group(1) or "").strip()  # 块内 JSON 文本，支持缺 </tool_call>
        parsed = _parse_tool_call_json(inner)
        if "crop_image_normalized" in inner:
            if not first_crop_expanded and n_crops >= 1:
                first_crop_expanded = True
                for i in range(n_crops):
                    bbox_i = bbox_list[i] if i < len(bbox_list) else [0.0, 0.0, 1.0, 1.0]
                    calls.append({
                        "name": "crop_image_normalized",
                        "arguments": {"bbox_2d": bbox_i, "target_image": 1},
                        "_crop_index": i,
                        "_total_crops": n_crops,
                    })
            else:
                bbox_i = bbox_list[0] if bbox_list else [0.0, 0.0, 1.0, 1.0]
                calls.append({"name": "crop_image_normalized", "arguments": {"bbox_2d": bbox_i, "target_image": 1}})
        elif "query_image" in inner:
            calls.append({"name": "query_image", "arguments": {}})
        elif "search" in inner:
            args = (parsed.get("arguments") or {}) if parsed else {}
            desc = args.get("description") if isinstance(args, dict) else None
            if not isinstance(desc, str):
                desc = _extract_search_description_fallback(inner)
            calls.append({"name": "search", "arguments": {"description": desc or ""}})
    return calls


def rewrite_assistant_text_with_multi_crop(assistant_text: str, sample: Dict) -> str:
    """若 sample 有多个 bbox，将 assistant 中第一个 crop tool_call 替换为连续多个 tool_call（用我们的 bbox）。"""
    bbox_list = _get_bbox_list(sample)
    n = min(len(bbox_list), len(_get_crop_image_list(sample))) if bbox_list else 0
    if n <= 1:
        return assistant_text
    first = RE_TOOL_CALL_BLOCK.search(assistant_text)
    if not first or "crop_image_normalized" not in first.group(0):
        return assistant_text
    new_blocks = "".join(
        f'<tool_call>{{"name":"crop_image_normalized","arguments":{{"bbox_2d":{json.dumps(bbox_list[i])},"target_image":1}}}}</tool_call>'
        for i in range(n)
    )
    return assistant_text[: first.start()] + new_blocks + assistant_text[first.end() :]

def has_final_answer(text: str) -> bool:
    return RE_ANSWER_TAG.search(text) is not None


# ======================
# Message format helpers
# ======================
def image_path_to_data_url(image_path: str) -> str:
    """
    将本地图片路径转换为 base64 data URL
    """
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # 根据文件扩展名确定 MIME 类型
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    mime_type = mime_types.get(ext, 'image/jpeg')
    
    # 读取文件并编码为 base64
    with open(image_path, 'rb') as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data).decode('utf-8')
    
    return f"data:{mime_type};base64,{base64_data}"

def make_system_msg() -> Dict:
    return {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}

def make_user_with_image(text: str, image_path: str) -> Dict:
    data_url = image_path_to_data_url(image_path)
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }
    # 保存原始路径，用于后续 JSON 序列化
    msg["_image_path"] = image_path
    return msg

def make_user_text_only(text: str) -> Dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}

def make_user_with_image_only(prefix_text: str, image_path: str) -> Dict:
    data_url = image_path_to_data_url(image_path)
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": prefix_text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }
    msg["_image_path"] = image_path
    return msg


def make_user_with_multiple_images(prefix_text: str, image_paths: List[str]) -> Dict:
    """一条 user 消息包含多张图片（用于多 bbox 时一次性返回全部裁剪图）。"""
    content: List[Dict] = [{"type": "text", "text": prefix_text}]
    for p in image_paths:
        if p and os.path.exists(p):
            content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(p)}})
    msg = {"role": "user", "content": content}
    msg["_image_paths"] = image_paths  # 供 convert_messages_to_paths 写回路径
    return msg

def make_assistant_text(text: str) -> Dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def normalize_tool_calls_in_assistant_text(text: str) -> str:
    """
    保证保存的 assistant 文本中工具调用格式正确：
    1) 若存在未闭合的 <tool_call>，补上 </tool_call>；
    2) 模型有时多打一个 }，如 }}} 应为 }}，统一修正。
    """
    if not text:
        return text
    # 修正多余的 }}} -> }}（如 search 结尾应为两个 }）
    text = re.sub(r"\}\}\}</tool_call>", "}}</tool_call>", text, flags=re.IGNORECASE)
    text = re.sub(r"\}\}\}\s*$", "}}", text)
    if "<tool_call>" not in text:
        return text
    last_open = text.rfind("<tool_call>")
    if last_open < 0:
        return text
    after_last_open = text[last_open:]
    if "</tool_call>" in after_last_open:
        return text
    return text + "</tool_call>"


def convert_messages_to_paths(messages: List[Dict]) -> List[Dict]:
    """
    将消息列表中的 base64 data URL 替换为原始图片路径，并规范化 assistant 中的 tool_call，用于 JSON 序列化。
    """
    converted = []
    for msg in messages:
        msg_copy = dict(msg)
        # 保证 assistant 消息里的 tool_call 格式正确（补全缺失的 </tool_call>）
        if msg_copy.get("role") == "assistant" and "content" in msg_copy:
            new_content = []
            for item in msg_copy["content"]:
                if item.get("type") == "text" and "text" in item:
                    new_content.append({
                        **item,
                        "text": normalize_tool_calls_in_assistant_text(item["text"]),
                    })
                else:
                    new_content.append(item)
            msg_copy["content"] = new_content
        if "_image_paths" in msg_copy:
            image_paths = msg_copy.pop("_image_paths")
            if "content" in msg_copy and isinstance(image_paths, list):
                new_content = []
                idx = 0
                for item in msg_copy["content"]:
                    if item.get("type") == "image_url" and "image_url" in item:
                        url = image_paths[idx] if idx < len(image_paths) else ""
                        new_content.append({"type": "image_url", "image_url": {"url": url}})
                        idx += 1
                    else:
                        new_content.append(item)
                msg_copy["content"] = new_content
        elif "_image_path" in msg_copy:
            image_path = msg_copy.pop("_image_path")
            if "content" in msg_copy:
                new_content = []
                for item in msg_copy["content"]:
                    if item.get("type") == "image_url" and "image_url" in item:
                        new_content.append({"type": "image_url", "image_url": {"url": image_path}})
                    else:
                        new_content.append(item)
                msg_copy["content"] = new_content
        converted.append(msg_copy)
    return converted


# ======================
# Search tool (visual description API)
# ======================
SEARCH_SYSTEM_PROMPT = """
You are a professional web search agent for industrial visual inspection.

Before producing the final output, you MUST internally search the web to retrieve relevant visual information about the described object or anomaly type.

Then generate a concise visual description (2-4 sentences, 80-150 words maximum).

Strict rules:
- Output only one short paragraph. Keep it concise: 2-4 sentences or 80-150 words max.
- No reasoning.
- No interpretation.
- No speculation.
- No explanation.
- No mention of search.
- No conclusions.
- No safety notes.
- No bullet points.
- No formatting.
- No headings.

Mention only the most relevant: morphology, contour, texture, color, edges, or contrast. Focus purely on observable visual characteristics.
""".strip()


def call_search(description: str) -> str:
    """Call API to generate concise visual description from brief description."""
    if not description or not description.strip():
        return "No description provided; please provide a brief description of the object or anomaly type."
    user_prompt = f"""
The following is a short inspection description:

"{description.strip()}"

Search for relevant visual information about this object or anomaly type.
Then produce a concise visual description (2-4 sentences, 80-150 words max) based purely on visual characteristics.
""".strip()
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Search tool failed: {e}"


# ======================
# Rollout engine
# ======================
def apply_tool_call(sample: Dict, call: Dict[str, Any]) -> Optional[Dict]:
    """
    根据 model tool_call 返回要插入的 user message。
    crop 已用 sample 的 bbox 替换；多 bbox 时由 rollout 合并为一条多图消息，此处仅处理单次调用。
    """
    name = call.get("name", "")
    args = call.get("arguments") or {}

    if name == "crop_image_normalized":
        # bbox 已在 extract_tool_calls 中替换为 sample 的 bbox
        crop_list = _get_crop_image_list(sample)
        if not crop_list:
            return make_user_text_only("Cropped view is unavailable because crop_image is missing.")
        idx = int(call.get("_crop_index", 0))
        idx = max(0, min(idx, len(crop_list) - 1))  # 防止越界
        crop_path = crop_list[idx]
        if not crop_path or not os.path.exists(crop_path):
            return make_user_text_only("Cropped view is unavailable because crop_image path is missing.")
        return make_user_with_image_only("Here is the cropped image:", crop_path)

    if name == "query_image":
        sims = sample.get("similar_templates") or []
        normal_path = sims[0] if sims else ""
        if not normal_path:
            return make_user_text_only("Normal reference image is unavailable because similar_templates is empty.")
        return make_user_with_image_only("Here is the normal reference image:", normal_path)

    if name == "search":
        desc = (args.get("description") or "").strip()
        result = call_search(desc)
        return make_user_text_only(f"Here is the detailed visual description:\n\n{result}")

    # 未知工具：忽略或提示
    return make_user_text_only(f"Tool '{name}' is not available.")


def chat_once(messages: List[Dict], temperature: float = 0.2) -> str:
    """
    单次调用模型，返回 assistant 文本。
    """
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def rollout_one_sample(sample: Dict) -> Tuple[Dict, Optional[str]]:
    """
    返回：带 message_list 的 sample（原字段不改） + error
    """
    msgs: List[Dict] = []
    msgs.append(make_system_msg())
    msgs.append(make_user_with_image(build_user_prompt(sample), sample.get("image", "")))

    final_text = None

    for step in range(1, MAX_STEPS + 1):
        # call model with retry
        last_err = None
        assistant_text = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                assistant_text = chat_once(msgs, temperature=0.2)
                break
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** (attempt - 1), 16))
        if last_err and not assistant_text:
            return {"rollout_error": str(last_err), "message_list": convert_messages_to_paths(msgs)}, str(last_err)

        # 多 bbox 时把 assistant 里第一个 crop 替换成连续多个 tool_call（用我们的 bbox）
        assistant_text = rewrite_assistant_text_with_multi_crop(assistant_text, sample)
        msgs.append(make_assistant_text(assistant_text))

        # if final answer produced, stop
        if has_final_answer(assistant_text):
            final_text = assistant_text
            break

        # parse tool calls（crop 已按 sample 的 bbox 展开为多次调用）
        calls = extract_tool_calls(assistant_text, sample)
        if not calls:
            # 没答案也没工具调用：强制给一个引导，避免卡死
            msgs.append(make_user_text_only(
                "If you need more evidence, you may call crop_image_normalized, query_image, or search. "
                "Otherwise, please provide the final <think>...</think><answer>...</answer>."
            ))
            continue

        # 若本轮全是 crop_image_normalized 且有多条，合并为一条 user 消息并给出全部裁剪图
        if (
            len(calls) > 1
            and all(c.get("name") == "crop_image_normalized" for c in calls)
        ):
            crop_list = _get_crop_image_list(sample)
            # 防止 _crop_index 或 calls 数量超过 crop_list 导致 list index out of range
            paths = []
            for i, c in enumerate(calls):
                idx = c.get("_crop_index", i)
                if idx < len(crop_list):
                    p = crop_list[idx]
                    if p and os.path.exists(p):
                        paths.append(p)
            if paths:
                msgs.append(make_user_with_multiple_images("Here are the cropped images:", paths))
            else:
                msgs.append(make_user_text_only("Cropped views are unavailable (crop_image missing or invalid)."))
        else:
            for call in calls:
                tool_user_msg = apply_tool_call(sample, call)
                if tool_user_msg is not None:
                    msgs.append(tool_user_msg)

    out = dict(sample)  # ✅ 保留原字段
    # 在保存前将消息中的 base64 data URL 转换为原始路径
    out["message_list"] = convert_messages_to_paths(msgs)
    # out["final_answer_text"] = final_text
    # if final_text is None:
    #     out["rollout_error"] = out.get("rollout_error", f"Max steps reached ({MAX_STEPS}) without final answer.")

    return out, out.get("rollout_error")


def process_idx(samples: List[Dict], idx: int) -> Tuple[int, Dict, Optional[str]]:
    out, err = rollout_one_sample(samples[idx])
    return idx, out, err


def main():
    input_path = Path("/NEW_EDS/miaojw/projects/agentiad-03/10.json")
    output_path = Path("/NEW_EDS/miaojw/projects/agentiad-03/10_multiturn.json")
    base_img_path = "/NEW_EDS/miaojw/projects/MMAD"

    raw = input_path.read_text(encoding="utf-8")
    raw = raw.replace("/NEW_EDS/miaojw/projects/MMAD", base_img_path)
    samples: List[Dict] = json.loads(raw)

    outputs: List[Optional[Dict]] = [None] * len(samples)
    failures: List[Dict] = []

    print(f"Start rollouts: N={len(samples)}, workers={WORKERS}, model={MODEL_NAME}, max_steps={MAX_STEPS}")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_idx, samples, i): i for i in range(len(samples))}
        done = 0
        for fut in as_completed(futs):
            done += 1
            idx = futs[fut]
            qid = samples[idx].get("qid", f"index_{idx}")
            try:
                _, out, err = fut.result()
                outputs[idx] = out
                if err:
                    failures.append({"qid": qid, "error": err})
                    print(f"❌ [{done}/{len(samples)}] {qid}: {err}")
                else:
                    print(f"✅ [{done}/{len(samples)}] {qid}")
            except Exception as e:
                failures.append({"qid": qid, "error": str(e)})
                outputs[idx] = dict(samples[idx])
                outputs[idx]["rollout_error"] = str(e)
                print(f"❌ [{done}/{len(samples)}] {qid}: {e}")

    # 写出
    output_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Wrote output to: {output_path}")
    print(f"Total: {len(samples)} | Failures: {len(failures)}")


if __name__ == "__main__":
    main()
