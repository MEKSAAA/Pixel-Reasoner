import json
from pathlib import Path

SYSTEM_PROMPT = """You are a vision expert specialized in industrial anomaly detection.
You will evaluate whether the given object image is normal or abnormal. If abnormal, select the most fitting anomaly label from the candidate types provided by the user.
Output format:<think>Explain your visual reasoning.</think><answer>{"anomaly_present": true/false, "top_anomaly": "<label or 'none'>", "visual_descriptions": ["..."]}</answer>
If normal → anomaly_present=false, top_anomaly="none", visual_descriptions=[].
If abnormal → include concise visual phrases for visible cues.

# Tools

You may call functions to assist with the user query.

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
  }
]
</tools>

For each function call, return a JSON object with the function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""

def build_user_prompt(sample, fixed_bottle_prompt=False):
    """
    - 如果 fixed_bottle_prompt=True，则使用你给的 bottle 固定提示；
    - 否则，按样本字段动态生成（推荐，能泛化到其它类和候选标签）。
    """
    if fixed_bottle_prompt:
        return (
            'Evaluate the following image from the class "bottle".\n\n'
            "Candidate anomaly types:\n"
            "- A smooth, uniform indentation.\n"
            "- A scratched surface.\n"
            "- A large, round hole.\n"
            "- A jagged, irregular area.\n\n"
            "Determine if the object is normal or abnormal. Follow the instruction and we can look closer by `crop_image_normalized`. "
            "You may also call `query_image` if the crop is still insufficient. "
            "Reason with the visual information step by step, and output the final answer in the required XML format."
        )
    else:
        cls = sample.get("class_name", "object")
        anoms = sample.get("anomaly_list", [])
        anoms_text = "".join([f"- {a}\n" for a in anoms]) if anoms else "- (none specified)\n"
        return (
            f'Evaluate the following image from the class "{cls}".\n\n'
            f"Candidate anomaly types:\n{anoms_text}\n"
            "Determine if the object is normal or abnormal. Follow the instruction and we can look closer by `crop_image_normalized`. "
            "If, after inspecting the crop, the evidence is still insufficient, you may also call `query_image` to retrieve a normal reference image. "
            "Reason with the visual information step by step, and output the final answer in the required XML format."
        )

def make_message_list(sample, second_text_map, third_text_map=None, use_fixed_bottle_prompt=False):
    """
    sample: 一条来自 train1600_diff_answers_enriched.json 的样本
    second_text_map: {qid: second_assistant_text} 来自 train1600_diff_answers_enriched_with_query.json
    third_text_map: {qid: third_assistant_text} 来自 train1600_diff_answers_enriched_with_query.json
    """
    qid = sample["qid"]
    second_assistant_text = second_text_map.get(qid, sample.get("second_assistant_text", ""))
    
    # 优先从 third_text_map 获取，否则从 sample 获取，最后回退到 second_assistant_text
    if third_text_map:
        third_assistant_text = third_text_map.get(qid, sample.get("third_assistant_text", second_assistant_text))
    else:
        third_assistant_text = sample.get("third_assistant_text", second_assistant_text)



    similar_templates = sample.get("similar_templates", [])
    normal_img_path = similar_templates[0] if similar_templates else ""

    # 构造消息列表
    message_list = [
        {
            "role": "system",
            "content": [{"text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [{"text": build_user_prompt(sample, fixed_bottle_prompt=use_fixed_bottle_prompt)}]
        },
        {
            "role": "assistant",
            "content": [{"text": sample["first_assistant_text"]}]
        },
        {
            "role": "user",
            "content": [
                {"text": "Here is the cropped image:"},
                {"image": sample.get("crop_image", "")}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    # 第二轮 assistant 输出 = 模型 reasoning + tool 调用
                    "text": (
                        f"{second_assistant_text.strip()}\n"
                        '<tool_call>{"name": "query_image", "arguments": {} }</tool_call>'
                    )
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"text": "Here is the normal reference image."},
                {"image": normal_img_path}
            ]
        },
        {
            "role": "assistant",
            "content": [{"text": third_assistant_text}]
        }
    ]
    return message_list

def main():
    base_path = Path("/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt")
    file_first = base_path / "train1600_diff_answers_enriched.json"
    file_second = base_path / "train1600_diff_answers_enriched_with_query.json"

    with open(file_first, "r", encoding="utf-8") as f:
        data_first = json.load(f)

    with open(file_second, "r", encoding="utf-8") as f:
        data_second = json.load(f)

    # 建立 {qid: second_assistant_text} 和 {qid: third_assistant_text} 映射
    second_map = {s["qid"]: s["second_assistant_text"] for s in data_second}
    third_map = {s["qid"]: s["third_assistant_text"] for s in data_second if "third_assistant_text" in s}

    out_records = []
    for sample in data_first:
        msg_list = make_message_list(sample, second_map, third_map, use_fixed_bottle_prompt=False)
        out = {
            "qid": sample["qid"],
            "class_name": sample.get("class_name"),
            "image": sample.get("image"),
            "message_list": msg_list
        }
        out_records.append(out)

    # 保存为 JSON（或改为 JSONL）
    out_file = base_path / "train1600_with_messages.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_records, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Wrote {len(out_records)} samples to: {out_file}")

if __name__ == "__main__":
    main()
