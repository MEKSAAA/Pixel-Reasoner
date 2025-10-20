# -*- coding: utf-8 -*-
"""
多轮对话推理脚本
复现训练数据中的两阶段推理流程：
1. 第一阶段：学习物体的全局和异常视觉描述
2. 中间处理：基于异常描述生成 ROI 区域
3. 第二阶段：基于 ROI 进行异常检测
"""
import os
import sys
import json
from PIL import Image
from typing import List, Dict, Any
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# 添加 CLIPtool 路径以便导入
CLIPTOOL_PATH = os.path.join(os.path.dirname(__file__), "CLIPtool")
if CLIPTOOL_PATH not in sys.path:
    sys.path.insert(0, CLIPTOOL_PATH)

# GPU 配置
# 提示：可以将 VLM 和 CLIP 放在不同的 GPU 上以避免显存竞争
# 如果只有一张 GPU，将两者都设为 "0"
VLM_GPU_ID = "0"      # VLM 模型使用的 GPU
CLIP_GPU_ID = "3"     # CLIP 模型使用的 GPU（如果只有一张GPU，设为 "0"）

# 图像分辨率配置（参考 curiosity_driven_rl 的配置）
MIN_PIXELS = 401408  # 256 * 28 * 28
MAX_PIXELS = 802816  # 5120 * 28 * 28

# 本地模型路径
MODEL_PATH = "/NEW_EDS/miaojw/projects/Pixel-Reasoner/output/ad_sft_qwen25vl7b_v2/checkpoint-924"


class AnomalyDetector:
    """异常检测推理类"""
    
    def __init__(self, model_path: str, gpu_id: str = "0"):
        """初始化模型和处理器"""
        # 设置环境变量
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["MIN_PIXELS"] = str(MIN_PIXELS)
        os.environ["MAX_PIXELS"] = str(MAX_PIXELS)
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"设备: {self.device}")
        print(f"图像分辨率范围: MIN_PIXELS={MIN_PIXELS}, MAX_PIXELS={MAX_PIXELS}")
        
        # 加载 Processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # 配置 tokenizer
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Workaround 1: 临时重命名 generation_config.json
        gen_config_path = os.path.join(model_path, "generation_config.json")
        gen_config_backup = os.path.join(model_path, "generation_config.json.backup")
        
        if os.path.exists(gen_config_path):
            os.rename(gen_config_path, gen_config_backup)
            print("已临时重命名 generation_config.json")
        
        # Workaround 2: 临时修复 config.json 中的 text_config 字段
        config_path = os.path.join(model_path, "config.json")
        config_backup = os.path.join(model_path, "config.json.backup")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        if 'text_config' in config_data:
            if not os.path.exists(config_backup):
                with open(config_backup, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                print("已备份 config.json")
            
            del config_data['text_config']
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            print("已临时删除 config.json 中的 text_config 字段")
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        
        # 恢复 config.json
        if os.path.exists(config_backup):
            os.rename(config_backup, config_path)
            print("已恢复 config.json")
        
        # 设置模型的 pad_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        print("模型加载完成\n")
    
    def generate_response(self, messages: List[Dict], max_new_tokens: int = 512, 
                         temperature: float = 0.0, do_sample: bool = False) -> str:
        """
        生成模型回复
        
        Args:
            messages: 对话消息列表
            max_new_tokens: 最大生成长度
            temperature: 温度参数
            do_sample: 是否采样
            
        Returns:
            模型生成的文本
        """
        # 处理输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # 生成回答
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 0.0,
        )
        
        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    
    def stage1_visual_description(self, class_name: str, image_path: str, 
                                 anomaly_categories: List[str]) -> Dict[str, Any]:
        """
        第一阶段：生成全局和异常视觉描述
        
        Args:
            class_name: 物体类别名称
            image_path: 图片路径
            anomaly_categories: 异常类别列表
            
        Returns:
            包含 global 和 anomalies 的字典
        """
        print("=" * 60)
        print("第一阶段：生成视觉描述")
        print("=" * 60)
        
        # 构建第一阶段的对话
        system_prompt = """You are a vision expert specialized in industrial inspection.

Goal:
Observe the given object image carefully and produce precise, concrete visual descriptions that capture what is visibly present.

Input:
(1) The object class name
(2) ONE image of that object
(3) A list of anomaly categories for this class

Output:
Return a single JSON object with two parts:
- "global" — phrases that objectively describe what this image visibly shows about the object's structure, material, color, texture, or shape.
- "anomalies" — for each anomaly category, phrases describing how that anomaly would visually appear on this class, limited to what is plausible from this image.

Language & style requirements:
- Use short noun/adjective phrases only (no full sentences, numbering, or punctuation).
- Focus on visible cues only — do not infer causes or unseen parts.
- Each phrase should describe at least two aspects among material/texture, color/contrast/tone, edge/shape/geometry, size/extent, or location/orientation.
- Avoid vague placeholders such as "residue" or "foreign object" unless qualified by appearance (e.g., "opaque residue on inner rim").
- Prefer diverse cues and avoid near duplicates.

Formatting rule:
Return JSON only (no additional text).
Example schema:
{
  "global": ["..."],
  "anomalies": {
    "<anomaly_1>": ["..."],
    "<anomaly_2>": ["..."]
  }
}"""
        
        # 构建异常类别文本
        anomaly_list_text = "\n".join([f"- \"{cat}\"" for cat in anomaly_categories])
        
        user_prompt = f"""You are now analyzing a new sample from the "{class_name}" class.

Class: "{class_name}"

Anomaly categories to cover:
{anomaly_list_text}

Return STRICT JSON with this schema (do not echo anything else):
{{
  "global": ["... phrases ..."],
  "anomalies": {{
    "<anomaly_1>": ["... phrases ..."],
    "<anomaly_2>": ["..."],
    "...": ["..."]
  }}
}}

Please provide detailed visual descriptions following the system instructions and return JSON only as specified above."""
        
        messages = [
            {
                "role": "system",
                "content": [{"text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"text": user_prompt},
                    {"image": image_path}
                ]
            }
        ]
        
        # 生成回复
        response = self.generate_response(messages, max_new_tokens=512)
        
        print(f"\n模型输出:\n{response}\n")
        
        # 解析 JSON
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")
            return {"global": [], "anomalies": {}}
    
    def stage2_anomaly_detection(self, class_name: str, original_image: str, 
                                roi_images: List[str], roi_bboxes: List[tuple],
                                anomaly_categories: List[str],
                                stage1_response: str) -> Dict[str, Any]:
        """
        第二阶段：基于视觉描述进行异常检测
        
        Args:
            class_name: 物体类别名称
            original_image: 原始图片路径
            roi_images: ROI 裁剪图片路径列表
            roi_bboxes: ROI 边界框列表 [(x_min, y_min, x_max, y_max), ...]
            anomaly_categories: 候选异常类型列表
            stage1_response: 第一阶段的回复（用于构建对话历史）
            
        Returns:
            包含 anomaly_present, top_anomaly, visual_descriptions 的字典
        """
        print("=" * 60)
        print("第二阶段：异常检测")
        print("=" * 60)
        
        # 获取原始图片尺寸
        with Image.open(original_image) as img:
            img_width, img_height = img.size
        
        # 第一阶段的对话（作为历史）
        stage1_system = """You are a vision expert specialized in industrial inspection.

Goal:
Observe the given object image carefully and produce precise, concrete visual descriptions that capture what is visibly present.

Input:
(1) The object class name
(2) ONE image of that object
(3) A list of anomaly categories for this class

Output:
Return a single JSON object with two parts:
- "global" — phrases that objectively describe what this image visibly shows about the object's structure, material, color, texture, or shape.
- "anomalies" — for each anomaly category, phrases describing how that anomaly would visually appear on this class, limited to what is plausible from this image.

Language & style requirements:
- Use short noun/adjective phrases only (no full sentences, numbering, or punctuation).
- Focus on visible cues only — do not infer causes or unseen parts.
- Each phrase should describe at least two aspects among material/texture, color/contrast/tone, edge/shape/geometry, size/extent, or location/orientation.
- Avoid vague placeholders such as "residue" or "foreign object" unless qualified by appearance (e.g., "opaque residue on inner rim").
- Prefer diverse cues and avoid near duplicates.

Formatting rule:
Return JSON only (no additional text).
Example schema:
{
  "global": ["..."],
  "anomalies": {
    "<anomaly_1>": ["..."],
    "<anomaly_2>": ["..."]
  }
}"""
        
        anomaly_list_text = "\n".join([f"- \"{cat}\"" for cat in anomaly_categories])
        
        stage1_user = f"""You are now analyzing a new sample from the "{class_name}" class.

Class: "{class_name}"

Anomaly categories to cover:
{anomaly_list_text}

Return STRICT JSON with this schema (do not echo anything else):
{{
  "global": ["... phrases ..."],
  "anomalies": {{
    "<anomaly_1>": ["... phrases ..."],
    "<anomaly_2>": ["..."],
    "...": ["..."]
  }}
}}

Please provide detailed visual descriptions following the system instructions and return JSON only as specified above."""
        
        # 第二阶段的 system prompt
        stage2_system = """You are a vision expert specialized in industrial anomaly detection.
Building on your prior visual understanding of this object class, you will now evaluate a specific instance and determine whether it is normal or abnormal.

Input:
(A) The ORIGINAL image of the object
(B) One or more ROI-cropped images, each showing a localized region
(C) The pixel bounding boxes of these ROIs in the original image (top-left x_min,y_min; bottom-right x_max,y_max)
(D) A list of candidate anomaly types for this class

Task:
1. Examine the original image for overall integrity, material consistency, and surface condition.
2. Review each ROI crop as a localized detail and compare it with the full image.
3. Decide whether the object is normal or abnormal based purely on visible evidence.
4. If abnormal, choose the most fitting anomaly type from the list (use "unknown" if none apply).

Decision policy:
- Treat ROI crops as neutral visual hints; confirm each cue using the full image.
- Mark abnormal only when there are clear and consistent indicators such as cracks, foreign materials, deformations, or structured contamination.
- Otherwise mark normal.
- Ignore mild lighting glare, slight blur, compression artifacts, or cropping seams.

Output format:
<think>
Describe your reasoning process: how the original and ROI images together support your judgment.
</think>
<answer>{"anomaly_present": true/false, "top_anomaly": "<label or 'none'>", "visual_descriptions": ["..."]}</answer>

Requirements:
- Keep descriptions visual and objective; do not mention ROI indices, coordinates, or prompts.
- If normal → anomaly_present=false, top_anomaly="none", visual_descriptions=[].
- If abnormal → list concise visual phrases describing visible cues (e.g., color difference, irregular texture, geometric distortion).
- Respond only with the two XML blocks — no extra text."""
        
        # 构建 ROI bbox 文本
        roi_bbox_text = "\n".join([
            f"- ROI {i+1} bbox (pixels in ORIGINAL {img_width}x{img_height}): {bbox}"
            for i, bbox in enumerate(roi_bboxes)
        ])
        
        # 构建候选异常类型文本
        candidate_anomalies_text = "\n".join([f"- {cat}" for cat in anomaly_categories])
        
        stage2_user = f"""You will now inspect a test image from the same "{class_name}".

- Class: {class_name}
- You will receive the ORIGINAL image followed by ROI-CROPPED images.

ROI pixel bounding boxes in the ORIGINAL image:
{roi_bbox_text}

Candidate anomaly types (choose from this list when possible):
{candidate_anomalies_text}

Return EXACTLY in the required XML format (<think>...</think><answer>{{...}}</answer>).

Based on the visual evidence, determine normal vs abnormal and respond strictly in the XML format defined above."""
        
        # 构建完整的多轮对话
        # 准备第二阶段用户消息的 content：文本 + 多张图片
        # 每个图片需要作为独立的 content 项
        all_images = [original_image] + roi_images
        stage2_user_content = [{"text": stage2_user}]
        
        # 添加所有图片作为独立的 content 项
        for img_path in all_images:
            stage2_user_content.append({"image": img_path})
        
        messages = [
            {
                "role": "system",
                "content": [{"text": stage1_system}]
            },
            {
                "role": "user",
                "content": [
                    {"text": stage1_user},
                    {"image": original_image}
                ]
            },
            {
                "role": "assistant",
                "content": [{"text": stage1_response}]
            },
            {
                "role": "system",
                "content": [{"text": stage2_system}]
            },
            {
                "role": "user",
                "content": stage2_user_content
            }
        ]
        
        # 生成回复
        response = self.generate_response(messages, max_new_tokens=512)
        
        print(f"\n模型输出:\n{response}\n")
        
        # 解析响应
        result = self._parse_stage2_response(response)
        return result
    
    def _parse_stage2_response(self, response: str) -> Dict[str, Any]:
        """解析第二阶段的响应"""
        import re
        
        # 提取 <answer> 标签中的 JSON
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            try:
                result = json.loads(answer_text)
                return result
            except json.JSONDecodeError as e:
                print(f"JSON 解析失败: {e}")
        
        return {
            "anomaly_present": False,
            "top_anomaly": "none",
            "visual_descriptions": []
        }
    
    def generate_roi_from_stage1(self, class_name: str, image_path: str, 
                                 stage1_result: Dict[str, Any],
                                 output_dir: str = "./roi_output",
                                 threshold: float = 0.7,
                                 min_region_area: int = 50,
                                 clip_image_size: int = 240,
                                 clip_gpu_id: str = None) -> Dict[str, Any]:
        """
        基于第一阶段的异常描述，使用 CLIP 工具生成 ROI 区域
        
        Args:
            class_name: 物体类别名称
            image_path: 原始图片路径
            stage1_result: 第一阶段的结果（包含 global 和 anomalies）
            output_dir: ROI 输出目录
            threshold: CLIP 检测阈值
            min_region_area: 最小区域面积
            clip_image_size: CLIP 处理的图像尺寸
            clip_gpu_id: CLIP 使用的 GPU ID（None 则使用 cuda:0，字符串如 "1" 表示 cuda:1）
            
        Returns:
            包含 roi_images（图片路径列表）和 roi_bboxes（边界框列表）的字典
        """
        print("\n" + "=" * 60)
        print("中间阶段：生成 ROI 区域")
        print("=" * 60)
        
        # 提取异常描述
        anomalies_dict = stage1_result.get("anomalies", {})
        abnormal_texts = []
        
        # 收集所有异常类别的描述
        for anomaly_type, descriptions in anomalies_dict.items():
            if isinstance(descriptions, list):
                abnormal_texts.extend(descriptions)
            elif isinstance(descriptions, str):
                abnormal_texts.append(descriptions)
        
        # 如果没有异常描述，使用类别名称
        if not abnormal_texts:
            abnormal_texts = [f"damaged {class_name}", f"{class_name} with defect"]
            print(f"警告：未找到异常描述，使用默认描述")
        
        print(f"提取到的异常描述 ({len(abnormal_texts)} 条):")
        for i, text in enumerate(abnormal_texts[:5]):  # 只显示前5条
            print(f"  - {text}")
        if len(abnormal_texts) > 5:
            print(f"  ... 还有 {len(abnormal_texts) - 5} 条")
        
        # 导入 CLIP 检测函数
        try:
            from single_image_test import detect_anomaly
        except ImportError as e:
            print(f"错误：无法导入 single_image_test: {e}")
            print(f"当前 sys.path: {sys.path[:3]}")
            raise
        
        # 设置 CLIP 使用的设备
        # 如果指定了 clip_gpu_id，使用该 GPU；否则使用 cuda 或 cpu
        if clip_gpu_id is not None:
            # 临时设置 CUDA_VISIBLE_DEVICES 来指定 CLIP 使用的 GPU
            # 保存当前环境变量
            original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            
            # 设置 CLIP 使用的 GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(clip_gpu_id)
            clip_device = "cuda"
            print(f"\n正在使用 CLIP 工具生成 ROI 区域...")
            print(f"  - CLIP GPU: cuda:{clip_gpu_id} (通过 CUDA_VISIBLE_DEVICES)")
        else:
            clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"\n正在使用 CLIP 工具生成 ROI 区域...")
            print(f"  - CLIP 设备: {clip_device}")
        
        print(f"  - 类别: {class_name}")
        print(f"  - 图像尺寸: {clip_image_size}x{clip_image_size}")
        print(f"  - 阈值: {threshold}")
        
        try:
            result = detect_anomaly(
                image_path=image_path,
                class_name=class_name,
                output_dir=output_dir,
                threshold=threshold,
                device=clip_device,
                min_region_area=min_region_area,
                model_name='ViT-B-16-plus-240',
                image_size=clip_image_size,
                abnormal_texts=abnormal_texts  # 使用提取的异常描述
            )
        finally:
            # 恢复原始的 CUDA_VISIBLE_DEVICES
            if clip_gpu_id is not None:
                if original_cuda_visible is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                else:
                    # 如果原来没有设置，则删除
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        del os.environ["CUDA_VISIBLE_DEVICES"]
        
        # 提取 ROI 信息
        roi_regions = result.get('white_bg_regions', [])
        roi_images = []
        roi_bboxes = []
        
        print(f"\n检测到 {len(roi_regions)} 个 ROI 区域:")
        for region in roi_regions:
            roi_path = region.get('path')
            bbox_resized = region.get('bbox_resized')
            
            if roi_path and bbox_resized:
                roi_images.append(roi_path)
                # 提取 bbox (x_min, y_min, x_max, y_max)
                bbox_tuple = (
                    bbox_resized['x_min'],
                    bbox_resized['y_min'],
                    bbox_resized['x_max'],
                    bbox_resized['y_max']
                )
                roi_bboxes.append(bbox_tuple)
                print(f"  - 区域 {region.get('region_index', 0)}: {roi_path}")
                print(f"    bbox (resize后): {bbox_tuple}")
        
        # 如果没有检测到 ROI，返回空列表（第二阶段会处理这种情况）
        if not roi_images:
            print("  警告：未检测到任何 ROI 区域")
        
        return {
            'roi_images': roi_images,
            'roi_bboxes': roi_bboxes,
            'anomaly_score': result.get('anomaly_score', 0.0),
            'clip_result': result
        }


def main():
    """主函数：演示三阶段推理流程（自动生成 ROI）"""
    
    # 初始化检测器（使用 VLM GPU）
    detector = AnomalyDetector(MODEL_PATH, VLM_GPU_ID)
    
    # ===== 配置测试数据 =====
    class_name = "bottle"
    original_image = "/NEW_EDS/miaojw/datasets/mvtec_dataset/bottle/train/good/000.png"
    
    # 异常类别
    anomaly_categories = ["large breakage", "small breakage", "contamination"]
    
    # ROI 生成配置
    roi_output_dir = "./roi_output"
    clip_threshold = 0.6  # CLIP 检测阈值（可调整）
    clip_image_size = 240  # CLIP 处理的图像尺寸
    
    # ===== 第一阶段：生成视觉描述 =====
    stage1_result = detector.stage1_visual_description(
        class_name=class_name,
        image_path=original_image,
        anomaly_categories=anomaly_categories
    )
    
    # 将第一阶段结果转为 JSON 字符串（用于第二阶段）
    stage1_response = json.dumps(stage1_result, ensure_ascii=False)
    
    # ===== 中间阶段：基于异常描述生成 ROI =====
    roi_result = detector.generate_roi_from_stage1(
        class_name=class_name,
        image_path=original_image,
        stage1_result=stage1_result,
        output_dir=roi_output_dir,
        threshold=clip_threshold,
        min_region_area=50,
        clip_image_size=clip_image_size,
        clip_gpu_id=CLIP_GPU_ID  # 使用单独的 GPU 运行 CLIP
    )
    
    roi_images = roi_result['roi_images']
    roi_bboxes = roi_result['roi_bboxes']
    
    # 如果没有生成 ROI，创建一个空白区域（正常样本的情况）
    if not roi_images:
        print("\n提示：未检测到 ROI 区域，将使用空白 ROI 进入第二阶段")
        # 创建一个小的虚拟 ROI
        roi_images = []
        roi_bboxes = [(0, 0, 1, 1)]  # 虚拟的小 bbox
    
    # ===== 第二阶段：异常检测 =====
    if roi_images:  # 只有在有 ROI 时才进行第二阶段
        stage2_result = detector.stage2_anomaly_detection(
            class_name=class_name,
            original_image=original_image,
            roi_images=roi_images,
            roi_bboxes=roi_bboxes,
            anomaly_categories=anomaly_categories,
            stage1_response=stage1_response
        )
    else:
        print("\n跳过第二阶段（无有效 ROI）")
        stage2_result = {
            "anomaly_present": False,
            "top_anomaly": "none",
            "visual_descriptions": []
        }
    
    # ===== 输出最终结果 =====
    print("\n" + "=" * 60)
    print("最终结果汇总")
    print("=" * 60)
    print(f"\n【第一阶段 - 视觉描述】")
    print(f"  Global 描述: {stage1_result.get('global', [])[:3]}...")
    print(f"  Anomalies 类型数: {len(stage1_result.get('anomalies', {}))}")
    
    print(f"\n【中间阶段 - ROI 生成】")
    print(f"  CLIP 异常分数: {roi_result.get('anomaly_score', 0):.4f}")
    print(f"  检测到 ROI 区域数: {len(roi_images)}")
    
    print(f"\n【第二阶段 - 异常检测】")
    print(f"  异常存在: {stage2_result['anomaly_present']}")
    print(f"  主要异常类型: {stage2_result['top_anomaly']}")
    print(f"  视觉描述: {stage2_result['visual_descriptions']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

