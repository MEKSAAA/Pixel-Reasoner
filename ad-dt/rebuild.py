
# Creates a reusable converter script that transforms your old two-round AD samples
# into the new multi-turn message_list format with region phrases + <tool_call>.
# It reads a JSON (list of items like your examples) and writes a JSONL file.
import json, re, os, sys
from typing import List, Dict, Any

SYSTEM_PROMPT_TMPL = """
    You are a vision expert specialized in industrial anomaly detection.

    You will complete TWO related rounds of visual reasoning for the SAME object image.

    ---
    ### ROUND 1 — Global Visual Understanding
    Goal: Build a visual understanding of the object's appearance and possible anomaly patterns for its class.

    Input:
    (1) The object class name
    (2) ONE image of that class (the SAME image will later be reused in Round 2)
    (3) A list of anomaly categories for this class

    Output:
    Return a STRICT JSON object in this schema:
    {
    "global": ["...phrases..."],
    "anomalies": {
        "<anomaly_1>": ["...phrases..."],
        "<anomaly_2>": ["...phrases..."]
    }
    }

    Guidelines:
    - Use short noun/adjective phrases only (no full sentences).
    - Focus on visible cues only: material, color/tone, edge/geometry, size, or location.
    - Each phrase should describe at least two aspects.
    - Avoid vague placeholders unless visually specified (e.g. “opaque residue on rim”).

    ---
    ### ROUND 2 — Instance Evaluation (SAME original image reused)
    Goal: Evaluate whether the SAME object shown in Round 1 is normal or abnormal.

    Input:
    (A) The SAME ORIGINAL image already seen in Round 1
    (B) The same candidate anomaly types from Round 1

    Task:
    1. Examine the original image for overall integrity and consistency.
    2. When necessary, you may use the tool `crop_image_normalized` to zoom in on any suspected region.
    - You decide autonomously where to crop and how many times.
    - For each tool call, the user will return the cropped image for inspection.
    3. Integrate global and local evidence to decide whether the object is normal or abnormal.
    4. If abnormal, select the most fitting anomaly label from the candidate types.

    Output format:
    <think>
    Explain your visual reasoning (how the full and ROI views support the decision).
    </think>
    <answer>{"anomaly_present": true/false, "top_anomaly": "<label or 'none'>", "visual_descriptions": ["..."]}</answer>

    Rules:
    - Keep reasoning visual and objective.
    - Do not mention ROI indices or coordinates explicitly in your reasoning.
    - If normal → anomaly_present=false, top_anomaly="none", visual_descriptions=[].
    - If abnormal → include concise visual phrases for visible cues.

    ---
    # Tool Usage Extension

    <tools>
    {"type": "function", "function": {"name": "crop_image_normalized", "description": "Zoom in on the image based on the bounding box coordinates that YOU determine from visual observation.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "Normalized coordinates [x_min, y_min, x_max, y_max] within [0.0, 1.0].", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "Index of the image to crop (1 for the original)."}}, "required": ["bbox_2d", "target_image"]}}}
    </tools>

    For each function call, return a json object within <tool_call></tool_call> XML tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call>
"""

def region_phrase_from_bbox(b: Dict[str, float]) -> str:
    # Use bbox center to map into 3x3 grid -> 7 words set
    cx = (b["x_min"] + b["x_max"]) / 2.0
    cy = (b["y_min"] + b["y_max"]) / 2.0
    # columns: left <0.33, center 0.33-0.67, right >0.67
    # rows: top <0.33, center 0.33-0.67, bottom >0.67
    col = "left" if cx < 0.33 else ("right" if cx > 0.67 else "center")
    row = "top" if cy < 0.33 else ("bottom" if cy > 0.67 else "center")
    if row == "center" and col == "center":
        return "center"
    if row == "center" and col == "left":
        return "left"
    if row == "center" and col == "right":
        return "right"
    if row == "top" and col == "left":
        return "top-left"
    if row == "top" and col == "right":
        return "top-right"
    if row == "bottom" and col == "left":
        return "bottom-left"
    if row == "bottom" and col == "right":
        return "bottom-right"
    # row is top/bottom and col is center → default to center w/ qualifier
    return "center"

def fmt_bbox_list(b):
    # Round to 3 decimals for readability/consistency
    return [round(float(b["x_min"]), 3), round(float(b["y_min"]), 3),
            round(float(b["x_max"]), 3), round(float(b["y_max"]), 3)]

def extract_round1_user(msg_list: List[Dict[str, Any]]):
    # Return the original Round1 user block (text+image) as-is
    for m in msg_list:
        if m.get("role") == "user":
            txts = [c.get("text","") for c in m.get("content",[]) if "text" in c]
            if any("ROUND 1 — Global Visual Understanding" in t for t in txts):
                return m
    return None

def extract_round1_assistant(msg_list: List[Dict[str, Any]]):
    for m in msg_list:
        if m.get("role") == "assistant":
            txts = [c.get("text","") for c in m.get("content",[]) if "text" in c]
            if txts and txts[0].strip().startswith("{") and "\"global\"" in txts[0]:
                return m
    return None

def extract_round2_candidates(msg_list: List[Dict[str, Any]]) -> str:
    # Extract the "Candidate anomaly types" block from Round2 user text
    for m in msg_list:
        if m.get("role") == "user":
            for c in m.get("content", []):
                if "text" in c and "ROUND 2 — Instance Evaluation" in c["text"]:
                    return c["text"]
    return "ROUND 2 — Instance Evaluation\n\nNow evaluate the SAME image you analyzed in Round 1.\nUse the same candidate anomaly types."

def extract_round2_original_image(msg_list: List[Dict[str, Any]], fallback_path: str) -> str:
    # Prefer the single path under round2 user; else fallback to original_image_path
    for m in msg_list:
        if m.get("role") == "user":
            txts = [c.get("text","") for c in m.get("content",[]) if "text" in c]
            if any("ROUND 2 — Instance Evaluation" in t for t in txts):
                # find first content with "image"
                for c in m.get("content", []):
                    if "image" in c:
                        img = c["image"]
                        if isinstance(img, str):
                            return img
                        if isinstance(img, list) and img:
                            # first should be original image
                            return img[0]
    return fallback_path

def build_system() -> Dict[str, Any]:
    return {"role": "system", "content": [{"text": SYSTEM_PROMPT_TMPL}]}

def build_round2_user(round2_text: str, orig_image: str) -> Dict[str, Any]:
    # Replace any lines that promise ROI crops/boxes with a generic instruction
    cleaned = re.sub(r"(?is)ROI.*?(types:|$)", "Candidate anomaly types:\n", round2_text)
    if "Candidate anomaly types" not in cleaned:
        cleaned += "\n\nCandidate anomaly types:\n"
    cleaned = re.sub(r"You will receive.*?(Class:|Use)", "Use the same candidate anomaly types.\n\n", cleaned, flags=re.I|re.S)
    return {"role": "user", "content": [{"text": cleaned.strip()}, {"image": orig_image}]}

def build_tool_turn(region_phrase: str, bbox: List[float]) -> Dict[str, Any]:
    text = f"I will inspect the **{region_phrase}** region for closer evidence.\n" \
           f"<tool_call>{{\"name\":\"crop_image_normalized\",\"arguments\":{{\"bbox_2d\":{bbox},\"target_image\":1}}}}</tool_call>"
    return {"role":"assistant","content":[{"text":text}]}

def build_user_cropped(seg_path: str) -> Dict[str, Any]:
    return {"role":"user","content":[{"text":"Here is the cropped image:"},{"image": seg_path}]}

def build_final_assistant(category: str) -> Dict[str, Any]:
    if category.strip().lower() == "good":
        think = ("The overall appearance of the object is consistent and intact across the full image "
                 "and inspected local regions. Surfaces remain smooth and continuous without visible residue, "
                 "cracks, or chips; color and texture are uniform.")
        ans = {"anomaly_present": False, "top_anomaly": "none", "visual_descriptions": []}
    else:
        # Generic abnormal fallback (you can post-edit per class if needed)
        think = ("Local inspections reveal visible irregularities inconsistent with the global pattern, "
                 "supporting the presence of an anomaly.")
        ans = {"anomaly_present": True, "top_anomaly": category, "visual_descriptions": ["local irregular texture","color/edge inconsistency"]}
    return {"role":"assistant","content":[{"text": f"<think>{think}</think><answer>{json.dumps(ans, ensure_ascii=False)}</answer>"}]}

def convert_item(item: Dict[str, Any]) -> Dict[str, Any]:
    # Keep top-level metadata as-is when possible
    out = {k: v for k, v in item.items() if k not in ("message_list",)}
    msg_old = item.get("message_list", [])

    system_new = build_system()
    r1_user = extract_round1_user(msg_old)
    r1_assistant = extract_round1_assistant(msg_old)

    # Round 2 user: rebuild from old text + single original image
    round2_text = extract_round2_candidates(msg_old)
    orig_image = extract_round2_original_image(msg_old, item.get("original_image_path",""))
    r2_user = build_round2_user(round2_text, orig_image)

    # Build assistant tool turns from provided regions (in order)
    regions = item.get("regions", [])
    tool_and_returns: List[Dict[str, Any]] = []
    for reg in regions:
        b = reg.get("bbox_normalized", {})
        # ensure x_max/y_max present: some inputs include width/height; prefer given x_max,y_max else compute
        x_min = float(b.get("x_min", 0.0))
        y_min = float(b.get("y_min", 0.0))
        x_max = float(b.get("x_max", x_min + float(b.get("width", 0.0))))
        y_max = float(b.get("y_max", y_min + float(b.get("height", 0.0))))
        bbox = [x_min, y_min, x_max, y_max]
        bbox = [round(v, 3) for v in bbox]
        phrase = region_phrase_from_bbox({"x_min":x_min,"y_min":y_min,"x_max":x_max,"y_max":y_max})
        tool_and_returns.append(build_tool_turn(phrase, bbox))
        seg_path = reg.get("segmentation_image_path", "")
        tool_and_returns.append(build_user_cropped(seg_path))

    final_assistant = build_final_assistant(item.get("category",""))

    out["message_list"] = [system_new, r1_user, r1_assistant, r2_user] + tool_and_returns + [final_assistant]
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Convert old two-round AD samples into tool_call multi-turn format.")
    ap.add_argument("--input", required=True, help="Path to input JSON (list of items).")
    ap.add_argument("--output", required=True, help="Path to output JSONL.")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = 0
    with open(args.output, "w", encoding="utf-8") as w:
        for item in data:
            try:
                new_item = convert_item(item)
                w.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                n += 1
            except Exception as e:
                sys.stderr.write(f"[WARN] skip item due to error: {e}\n")
                continue
    print(f"Converted {n} items -> {args.output}")


if __name__ == "__main__":
    main()
