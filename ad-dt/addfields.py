import json, re, os, sys

tool_re = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
answer_re = re.compile(r"<answer>\s*(\{.*?\})\s*</answer>", re.DOTALL)
class_re = re.compile(r'class\s+[\"\\\']([^\"\\\']+)[\"\\\']', re.IGNORECASE)

def parse_candidates(text: str):
    lines = text.splitlines()
    out = []
    in_block = False
    for ln in lines:
        if "Candidate anomaly types" in ln:
            in_block = True
            continue
        if in_block:
            s = ln.strip()
            if not s:
                break
            if s.startswith("-") or s.startswith("–"):
                out.append(s[1:].strip().rstrip(".") + ".")
            else:
                # 不是列表项就结束
                break
    return out

def build_question(cls_name, cand_list):
    header = f'Evaluate the following image from the class "{cls_name}".\n\nCandidate anomaly types:'
    body = "\n" + "\n".join([f"- {x}" for x in cand_list]) + "\n"
    tail = "\nDetermine if the object is normal or abnormal. Follow the instruction and we can look closer by `crop_image_normalized`. Reason with the visual information step by step, and output the final answer in the required XML format."
    return header + body + tail

def infer_mask_path(image_path, cls_name, anomaly_type):
    # 仅 DS-MVTec 且有异常类型时尝试推断
    if ("DS-MVTec" not in image_path) or (not anomaly_type) or (anomaly_type.lower()=="none"):
        return None
    # 例：.../DS-MVTec/zipper/image/squeezed_teeth/015.png
    # ->  .../DS-MVTec/zipper/rgb_mask/squeezed_teeth/015_rgb_mask.png
    try:
        parts = image_path.split("/")
        idx = parts.index("DS-MVTec")
        base = parts[:idx+2]  # ..., DS-MVTec, <cls>
        # 找到文件名与异常子目录
        fname = os.path.basename(image_path)
        stem, ext = os.path.splitext(fname)
        # image/.../<anom>/<fname>
        anom = anomaly_type.strip().lower().replace(" ", "_")
        # 特例：去掉句号
        anom = anom.rstrip(".")
        mask = base + ["rgb_mask", anom, f"{stem}_rgb_mask.png"]
        return "/".join(mask)
    except Exception:
        return None

def parse_bbox(tool_text):
    try:
        m = tool_re.search(tool_text)
        if not m: return None
        obj = json.loads(m.group(1))
        args = obj.get("arguments", {})
        bbox = args.get("bbox_2d")
        if isinstance(bbox, list) and len(bbox)==4:
            return [float(x) for x in bbox]
    except Exception:
        return None
    return None

def main(in_path, out_path):
    data = json.load(open(in_path, "r"))
    for item in data:
        # 1) 拿到 user 段原图与描述文本
        msg = item.get("message_list", [])
        user_text = ""
        image_path = None
        for m in msg:
            if m.get("role")=="user":
                for c in m.get("content", []):
                    if "text" in c and not user_text:
                        user_text = c["text"]
                    if "image" in c and image_path is None:
                        image_path = c["image"]
                # 只取首个 user 段
                if user_text and image_path:
                    break

        # 2) 类名与候选类型
        cls_name = None
        mcls = class_re.search(user_text or "")
        if mcls: cls_name = mcls.group(1).strip()
        cand_list = parse_candidates(user_text or "")

        # 3) 解析 assistant 的 answer JSON 与 tool_call bbox
        anomaly_present = None
        top_anomaly = None
        bbox = None
        for m in msg:
            if m.get("role")=="assistant":
                for c in m.get("content", []):
                    t = c.get("text","")
                    # 答案
                    ma = answer_re.search(t)
                    if ma:
                        try:
                            ans = json.loads(ma.group(1))
                            anomaly_present = bool(ans.get("anomaly_present"))
                            top_anomaly = ans.get("top_anomaly","none")
                        except Exception:
                            pass
                    # bbox
                    if bbox is None:
                        bb = parse_bbox(t)
                        if bb is not None:
                            bbox = bb

        # 4) question 文本
        question = build_question(cls_name or "unknown", cand_list)

        # 5) gt_answer / anomaly_type
        gt_answer = bool(anomaly_present) if anomaly_present is not None else None
        anomaly_type = (top_anomaly or "none")
        if not gt_answer:
            anomaly_type = "none"

        # 6) mask_path
        mask_path = infer_mask_path(image_path or "", cls_name or "", anomaly_type)

        # 7) 写回新字段
        item["question"] = question
        item["image"] = image_path
        item["class_name"] = cls_name
        item["anomaly_list"] = cand_list
        item["gt_answer"] = gt_answer
        item["anomaly_type"] = anomaly_type
        item["mask_path"] = mask_path
        item["bbox"] = bbox

    json.dump(data, open(out_path, "w"), ensure_ascii=False, indent=2)

if __name__=="__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    main(in_path, out_path)