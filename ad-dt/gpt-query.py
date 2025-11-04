import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

OPENAI_API_KEY="sk-CR8zGpArpDAF1IkyE55181D2285f47Ee867244548262977d"
OPENAI_BASE_URL="https://api.laozhang.ai/v1"


client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

SYSTEM_PROMPT = """You are a vision expert specialized in industrial anomaly detection.
You will evaluate whether the given object image is normal or abnormal.
If abnormal, select the most fitting anomaly label from the candidate types provided by the user.
Output format:
<think>Explain your visual reasoning.</think><answer>{"anomaly_present": true/false, "top_anomaly": "<label or 'none'>", "visual_descriptions": ["..."]}</answer>
If normal → anomaly_present=false, top_anomaly="none", visual_descriptions=[].
If abnormal → include concise visual phrases for visible cues.
"""

def extract_answer_from_text(text: str):
    """提取 <answer> JSON 并返回 anomaly_present 布尔值"""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(1).strip())
        return data.get("anomaly_present")
    except Exception:
        return None

def validate_format(answer_text: str) -> bool:
    """校验是否包含 <think>…</think> 和可解析的 <answer>{…}</answer>"""
    has_think = "<think>" in answer_text and "</think>" in answer_text
    predicted = extract_answer_from_text(answer_text)
    return has_think and (predicted is not None)

def validate_against_gt(answer_text: str, gt_answer: bool) -> bool:
    predicted = extract_answer_from_text(answer_text)
    return (predicted is not None) and (predicted == gt_answer)

def generate_with_reference(sample: dict, model: str = "gpt-4o", max_retries: int = 5):
    """
    两阶段生成：
      1) Round1：只看原图+crop → 模型自己声明证据不足，需要正常参考图（不含 <think>/<answer>，不暴露GT）
      2) Round2：给一张正常参考图，并明确告知 GT（gt_answer & anomaly_type）→ 输出 <think><answer>，并要求一致
    """
    # --------- Round 1：仅"需要参考图"的反思（不含 <think>/<answer>，不暴露GT）---------
    anomaly_list_str = "\n- " + "\n- ".join(sample.get("anomaly_list", []))
    user_prompt_1 = f"""
You are examining an object of class "{sample['class_name']}" for potential anomalies.
Candidate anomaly types are:{anomaly_list_str}

You have the target full image and a cropped ROI. Below is your observation text focusing on the crop:
{sample['first_assistant_text']}

Write a reflection (2-3 sentences maximum) explaining that the current evidence from the target image and its crop
is not sufficient to confidently decide normal vs. abnormal, and that you would like to check a normal
reference image of the same class before making a final decision.
Do NOT include any <think> or <answer> sections here. Keep your response SHORT and concise.
"""
    r1 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_1},
        ],
        temperature=0.7,
    )
    first_reply = r1.choices[0].message.content.strip()

    # --------- Round 2：提供正常参考图 + 明确告知GT → 要求 <think><answer> ----------
    similar_templates = sample.get("similar_templates", [])
    normal_ref = similar_templates[0] if similar_templates else "N/A"

    base_prompt_2 = f"""
You have now viewed a normal reference image of the same class.
Normal reference image path: {normal_ref}

GROUND TRUTH (for supervision in this stage):
- anomaly_present = {sample['gt_answer']}
- anomaly_type = {sample['anomaly_type']}

Task:
Compare the target image (and its cropped ROI) against the normal reference.
Then provide your reasoning and final decision in the required format:

<think>Explain your visual reasoning with concrete visual cues that differ between the target and the normal reference.</think>
<answer>{{"anomaly_present": true/false, "top_anomaly": "<label or 'none'>", "visual_descriptions": ["..."]}}</answer>

Requirements:
- Your <answer>.anomaly_present MUST be {str(sample['gt_answer']).lower()} to match the ground truth shown above.
- If anomaly_present is true, set top_anomaly to "{sample['anomaly_type']}" (or the closest match in candidate types).
- Use short visual phrases in "visual_descriptions" (e.g., "fine linear scuff marks", "matte patches on glossy coating").
- Output exactly one <think> block and one <answer> block. Do not include any tool calls.
"""

    second_reply = None
    for attempt in range(1, max_retries + 1):
        r2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": base_prompt_2},
            ],
            temperature=0.7,
        )
        candidate = r2.choices[0].message.content.strip()

        # 1) 先检查格式
        if not validate_format(candidate):
            base_prompt_2 += """
Format reminder: Include exactly one <think>...</think> and one <answer>{...}</answer> with valid JSON.
"""
            if attempt == max_retries:
                second_reply = candidate
                break
            continue

        # 2) 再检查与GT一致（这里GT已暴露，所以属于监督信号）
        if validate_against_gt(candidate, sample["gt_answer"]):
            second_reply = candidate
            break
        else:
            # 引导其与GT对齐（不改变GT，仅提示一致性与对比维度）
            base_prompt_2 += f"""
Consistency reminder: Your <answer>.anomaly_present must be {str(sample['gt_answer']).lower()} to match the ground truth.
Refine your rationale with coating gloss, micro-scuffs, and text-edge crispness versus the normal reference.
"""
            if attempt == max_retries:
                second_reply = candidate

    # --------- 汇总输出 ----------
    out = {
        "qid": sample["qid"],
        "class_name": sample["class_name"],
        "image": sample["image"],
        "crop_image": sample.get("crop_image"),
        "first_assistant_text": first_reply,    # 仅“需要参考图”的反思
        "second_assistant_text": second_reply,  # <think><answer>（已在第二轮暴露GT）
        "gt_answer": sample["gt_answer"],
        "anomaly_type": sample["anomaly_type"],
        "anomaly_list": sample["anomaly_list"],
        "similar_templates": sample["similar_templates"],
    }
    out["needs_manual_review"] = not validate_against_gt(second_reply, sample["gt_answer"])
    return out

def process_sample(sample, index, total):
    try:
        out = generate_with_reference(sample)
        return (True, out, sample["qid"], None)
    except Exception as e:
        return (False, None, sample.get("qid", "unknown"), str(e))

if __name__ == "__main__":
    input_file = "/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt/train1600_diff_answers_enriched.json"
    output_file = "/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt/train1600_diff_answers_enriched_with_query.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results, failed = [], []
    max_workers = 10
    print(f"开始并行处理 {len(data)} 条记录，线程数={max_workers} ...")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2smp = {ex.submit(process_sample, s, i, len(data)): (i, s) for i, s in enumerate(data)}
        done = 0
        for fut in as_completed(fut2smp):
            done += 1
            _, sample = fut2smp[fut]
            try:
                ok, out, qid, err = fut.result()
                if ok:
                    results.append(out)
                    print(f"✅ [{done}/{len(data)}] {qid}")
                else:
                    failed.append({"qid": qid, "error": err})
                    print(f"❌ [{done}/{len(data)}] {qid}: {err}")
            except Exception as e:
                qid = sample.get("qid", "unknown")
                failed.append({"qid": qid, "error": str(e)})
                print(f"❌ [{done}/{len(data)}] {qid}: {e}")

    # 按原始顺序排序结果
    idx = {s["qid"]: i for i, s in enumerate(data)}
    results.sort(key=lambda x: idx.get(x["qid"], len(data)))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n完成！")
    print(f"- 成功生成: {len(results)}/{len(data)}")
    print(f"- 失败: {len(failed)}")
    if failed:
        print("需关注的失败条目：")
        for it in failed:
            print(f"  - {it['qid']}: {it['error']}")