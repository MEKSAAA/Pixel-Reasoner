#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单样本异常检测工具
支持对单张图片进行异常检测，输出score、anomaly map和分割图像
"""

import os
import cv2
import json
import torch
import torch.nn as nn
import argparse
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms


def normalize(pred, max_value=None, min_value=None):
    """归一化异常分数图"""
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_white_background_mask_multi_regions(image, scoremap, threshold=0.5, min_area=50):
    """
    检测多个异常区域，每个区域单独裁剪并返回
    
    Args:
        image: 原始图像 (numpy array, RGB)
        scoremap: 归一化的异常分数图 (0-1之间)
        threshold: 异常检测阈值，超过此值的区域被保留
        min_area: 最小区域面积阈值，小于此值的区域会被忽略
    
    Returns:
        regions: 列表，每个元素包含 {
            'image': 裁剪后的图像,
            'bbox': (x_min, y_min, x_max, y_max) 在resize后图像中的坐标,
            'area': 区域面积
        }
    """
    np_image = np.asarray(image, dtype=np.uint8).copy()
    h, w = np_image.shape[:2]
    
    # 创建二值mask：超过阈值的区域为异常
    binary_mask = (scoremap > threshold).astype(np.uint8)
    
    # 对mask进行形态学操作，去除噪点并使边界更清晰
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # 找到所有连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    regions = []
    
    # 遍历每个连通区域（跳过背景，从1开始）
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # 过滤掉太小的区域
        if area < min_area:
            continue
        
        # 获取当前区域的mask
        region_mask = (labels == i).astype(np.uint8)
        
        # 创建白色背景图像
        white_background = np.ones_like(np_image) * 255
        
        # 只保留当前区域的原图
        mask_area = region_mask[:, :, np.newaxis].astype(bool)
        white_background[mask_area.squeeze()] = np_image[mask_area.squeeze()]
        
        # 获取区域的边界框（不加padding）
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        x_min = x
        y_min = y
        x_max = x + width
        y_max = y + height
        
        # 直接裁剪到mask边界
        cropped_image = white_background[y_min:y_max, x_min:x_max]
        
        regions.append({
            'image': cropped_image,
            'bbox': (x_min, y_min, x_max, y_max),  # resize后图像中的bbox
            'area': area
        })
    
    return regions


class prompt_order():
    """文本提示词生成器"""
    def __init__(self):
        super().__init__()
        self.state_normal_list = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
            "{} without flaw",
            "{} without defect",
            "{} without damage"
        ]

        self.state_anomaly_list = [
            "damaged {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage"
        ]

        self.template_list = [
            "a cropped photo of the {}.",
            "a close-up photo of a {}.",
            "a close-up photo of the {}.",
            "a bright photo of a {}.",
            "a bright photo of the {}.",
            "a dark photo of the {}.",
            "a dark photo of a {}.",
            "a jpeg corrupted photo of the {}.",
            "a jpeg corrupted photo of the {}.",
            "a blurry photo of the {}.",
            "a blurry photo of a {}.",
            "a photo of a {}.",
            "a photo of the {}.",
            "a photo of a small {}.",
            "a photo of the small {}.",
            "a photo of a large {}.",
            "a photo of the large {}.",
            "a photo of the {} for visual inspection.",
            "a photo of a {} for visual inspection.",
            "a photo of the {} for anomaly detection.",
            "a photo of a {} for anomaly detection."
        ]
    
    def prompt(self, class_name):
        """生成正常和异常的提示词"""
        class_state = [ele.format(class_name) for ele in self.state_normal_list]
        normal_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in self.template_list]
    
        class_state = [ele.format(class_name) for ele in self.state_anomaly_list]
        anomaly_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in self.template_list]
        return normal_ensemble_template, anomaly_ensemble_template


class patch_scale():
    """图像patch生成器"""
    def __init__(self, image_size):
        self.h, self.w = image_size
 
    def make_mask(self, patch_size=16, kernel_size=16, stride_size=16): 
        self.patch_size = patch_size
        self.patch_num_h = self.h // self.patch_size
        self.patch_num_w = self.w // self.patch_size
        ###################################################### patch_level
        self.kernel_size = kernel_size // patch_size
        self.stride_size = stride_size // patch_size
        self.idx_board = torch.arange(1, self.patch_num_h * self.patch_num_w + 1, dtype=torch.float32).reshape((1, 1, self.patch_num_h, self.patch_num_w))
        patchfy = torch.nn.functional.unfold(self.idx_board, kernel_size=self.kernel_size, stride=self.stride_size)
        return patchfy


class CLIP_AD(nn.Module):
    """CLIP异常检测模型"""
    def __init__(self, model_name='ViT-B-16-plus-240', device='cuda'):
        super(CLIP_AD, self).__init__()
        # 动态导入open_clip
        try:
            from src import open_clip
        except ImportError:
            raise ImportError("请确保 src 目录在同一目录下")
        
        self.model, _, self.preprocess = open_clip.create_customer_model_and_transforms(model_name, pretrained='laion400m_e31')
        self.mask = patch_scale((240, 240))
        self.device = device
    
    def encode_text(self, text):
        return self.model.encode_text(text)
    
    def encode_image(self, image, patch_size, mask=True):
        if mask:
            b, _, _, _ = image.shape
            large_scale = self.mask.make_mask(kernel_size=48, patch_size=patch_size).squeeze().to(self.device)
            mid_scale = self.mask.make_mask(kernel_size=32, patch_size=patch_size).squeeze().to(self.device)
            tokens_list, class_tokens, patch_tokens = self.model.encode_image(image, [large_scale, mid_scale], proj=False)
            large_scale_tokens, mid_scale_tokens = tokens_list[0], tokens_list[1]
            return large_scale_tokens, mid_scale_tokens, patch_tokens.unsqueeze(2), class_tokens, large_scale, mid_scale
        return None


def compute_score(image_features, text_features):
    """计算图像-文本相似度分数"""
    image_features /= image_features.norm(dim=1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    text_probs = (torch.bmm(image_features.unsqueeze(1), text_features) / 0.07).softmax(dim=-1)
    return text_probs


def compute_sim(image_features, text_features):
    """计算图像-文本相似度"""
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    simmarity = (torch.bmm(image_features.squeeze(2), text_features) / 0.07).softmax(dim=-1)
    return simmarity


def harmonic_aggregation(score_size, simmarity, mask):
    """调和聚合"""
    b, h, w = score_size
    simmarity = simmarity.double()
    score = torch.zeros((b, h * w)).to(simmarity).double()
    mask = mask.T
    for idx in range(h * w):
        patch_idx = [bool(torch.isin(idx + 1, mask_patch)) for mask_patch in mask]
        sum_num = sum(patch_idx)
        harmonic_sum = torch.sum(1.0 / simmarity[:, patch_idx], dim=-1)
        score[:, idx] = sum_num / harmonic_sum

    score = score.reshape(b, h, w)
    return score


def tokenize(texts):
    """简单的文本tokenizer"""
    try:
        from src.open_clip.tokenizer import tokenize as clip_tokenize
        return clip_tokenize(texts)
    except ImportError:
        raise ImportError("请确保 src 模块可用")


class SingleImageDetector:
    """单图像异常检测器"""
    
    def __init__(self, class_name, model_name='ViT-B-16-plus-240', image_size=240, device='cuda', abnormal_texts=None):
        """
        初始化检测器
        
        Args:
            class_name: 目标类别名称（如 "bottle", "carpet" 等）
            model_name: 使用的模型名称
            image_size: 图像resize的尺寸
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.class_name = class_name
        self.image_size = image_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.abnormal_texts = abnormal_texts  # 用户自定义的异常文本描述列表（可选）
        
        # 初始化模型
        print(f"正在加载模型 {model_name}...")
        self.model = CLIP_AD(model_name, device=self.device)
        self.model.to(self.device)
        self.model.eval()
        
        # 准备文本特征
        print(f"正在准备类别 '{class_name}' 的文本特征...")
        self._prepare_text_features()
        
        # 图像预处理
        self.preprocess = self.model.preprocess
        self.preprocess.transforms[0] = transforms.Resize(
            size=(image_size, image_size), 
            interpolation=transforms.InterpolationMode.BICUBIC,
            max_size=None, 
            antialias=None
        )
        self.preprocess.transforms[1] = transforms.CenterCrop(size=(image_size, image_size))
        
        print("初始化完成！")
    
    def _prepare_text_features(self):
        """准备文本特征"""
        text_generator = prompt_order()
        normal_description, abnormal_description = text_generator.prompt(self.class_name)
        # 如果用户提供了异常描述列表，则直接使用这些描述，不再通过template扩展
        if self.abnormal_texts is not None:
            if isinstance(self.abnormal_texts, (list, tuple)):
                abnormal_description = list(self.abnormal_texts)
            else:
                abnormal_description = [self.abnormal_texts]
        
        normal_tokens = tokenize(normal_description).to(self.device)
        abnormal_tokens = tokenize(abnormal_description).to(self.device)
        
        with torch.no_grad():
            normal_text_features = self.model.encode_text(normal_tokens).float()
            abnormal_text_features = self.model.encode_text(abnormal_tokens).float()
        
        self.avg_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        self.avg_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)
    
    @torch.no_grad()
    def detect(self, image_path):
        """
        对单张图像进行异常检测
        
        Args:
            image_path: 图像路径
            
        Returns:
            dict: 包含检测结果的字典
                - 'anomaly_score': 异常分数 (float)
                - 'anomaly_map': 异常分数图 (numpy array, HxW)
                - 'original_size': 原始图像尺寸 (height, width)
        """
        # 读取图像
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        original_h, original_w = original_img.shape[:2]
        
        # 预处理
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # 编码图像
        patch_size = 16
        large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_scale, mid_scale = \
            self.model.encode_image(img_tensor, patch_size)
        
        b, c, h, w = img_tensor.shape
        
        # 准备文本特征
        text_features = torch.cat((
            self.avg_normal_text_features.unsqueeze(0), 
            self.avg_abnormal_text_features.unsqueeze(0)
        ), dim=1).permute(0, 2, 1)
        
        # 计算分数
        zscore = compute_score(class_tokens, text_features)
        z0score = zscore[:, 0, 1]
        
        # 多尺度相似度
        large_scale_simmarity = compute_sim(large_scale_tokens, text_features)[:, :, 1]
        mid_scale_simmarity = compute_sim(mid_scale_tokens, text_features)[:, :, 1]
        
        # 多尺度分数
        large_scale_score = harmonic_aggregation((b, h // patch_size, w // patch_size), large_scale_simmarity, large_scale)
        mid_scale_score = harmonic_aggregation((b, h // patch_size, w // patch_size), mid_scale_simmarity, mid_scale)
        
        multiscale_score = torch.nan_to_num(
            3.0 / (1.0 / large_scale_score.to(self.device) + 1.0 / mid_scale_score.to(self.device) + 1.0 / z0score.unsqueeze(1).unsqueeze(1)),
            nan=0.0, posinf=0.0, neginf=0.0
        )
        
        multiscale_score = multiscale_score.unsqueeze(1)
        multiscale_score = F.interpolate(multiscale_score, size=(h, w), mode='bilinear')
        multiscale_score = multiscale_score.squeeze()
        
        # 返回结果
        return {
            'anomaly_score': float(z0score.cpu().numpy()),
            'anomaly_map': multiscale_score.cpu().numpy(),
            'original_size': (original_h, original_w)
        }
    
    def visualize_and_save(self, image_path, output_dir, threshold=0.5, min_region_area=50):
        """
        可视化并保存检测结果
        
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            threshold: 异常检测阈值
            min_region_area: 最小区域面积
            
        Returns:
            dict: 保存结果的详细信息
        """
        print(f"正在检测图像: {image_path}")
        result = self.detect(image_path)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取文件名
        filename = os.path.basename(image_path)
        filename_base = os.path.splitext(filename)[0]
        filename_ext = os.path.splitext(filename)[1]
        
        # 读取原图并resize
        original_img = cv2.imread(image_path)
        vis = cv2.cvtColor(cv2.resize(original_img, (self.image_size, self.image_size)), cv2.COLOR_BGR2RGB)
        
        # 归一化异常图
        anomaly_map_normalized = normalize(result['anomaly_map'])
        
        output_info = {
            'input_image': os.path.abspath(image_path),
            'anomaly_score': result['anomaly_score'],
            'original_size': {'height': result['original_size'][0], 'width': result['original_size'][1]},
            'resized_size': {'height': self.image_size, 'width': self.image_size},
            'threshold': threshold,
            'class_name': self.class_name
        }
        
        # 保存白色背景分割图（多区域）
        regions = apply_white_background_mask_multi_regions(
            vis, anomaly_map_normalized, 
            threshold=threshold, 
            min_area=min_region_area
        )
        
        original_h, original_w = result['original_size']
        scale_x = original_w / self.image_size
        scale_y = original_h / self.image_size
        
        region_info_list = []
        
        if len(regions) == 0:
            # 没有检测到异常区域，保存白色图像
            white_image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255
            white_bg_path = os.path.join(output_dir, f"{filename_base}_white_bg{filename_ext}")
            cv2.imwrite(white_bg_path, white_image)
            region_info_list.append({
                'region_index': 0,
                'path': os.path.abspath(white_bg_path),
                'num_regions': 0,
                'bbox_resized': None,
                'bbox_original': None
            })
            print(f"  - 未检测到异常区域，保存空白图: {white_bg_path}")
        else:
            # 为每个区域保存单独的图像
            for region_idx, region in enumerate(regions):
                if len(regions) > 1:
                    seg_filename = f"{filename_base}_region_{region_idx}{filename_ext}"
                else:
                    seg_filename = f"{filename_base}_white_bg{filename_ext}"
                
                seg_path = os.path.join(output_dir, seg_filename)
                region_image = cv2.cvtColor(region['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(seg_path, region_image)
                
                x_min, y_min, x_max, y_max = region['bbox']
                
                # 计算原图尺寸的bbox
                x_min_orig = int(x_min * scale_x)
                y_min_orig = int(y_min * scale_y)
                x_max_orig = int(x_max * scale_x)
                y_max_orig = int(y_max * scale_y)
                
                region_info_list.append({
                    'region_index': region_idx,
                    'path': os.path.abspath(seg_path),
                    'num_regions': len(regions),
                    'bbox_resized': {
                        'x_min': int(x_min),
                        'y_min': int(y_min),
                        'x_max': int(x_max),
                        'y_max': int(y_max),
                        'width': int(x_max - x_min),
                        'height': int(y_max - y_min)
                    },
                    'bbox_original': {
                        'x_min': x_min_orig,
                        'y_min': y_min_orig,
                        'x_max': x_max_orig,
                        'y_max': y_max_orig,
                        'width': x_max_orig - x_min_orig,
                        'height': y_max_orig - y_min_orig
                    }
                })
                print(f"  - 区域 {region_idx} 已保存: {seg_path}")
        
        output_info['white_bg_regions'] = region_info_list
        
        # 保存JSON结果
        json_path = os.path.join(output_dir, f"{filename_base}_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_info, f, indent=4, ensure_ascii=False)
        
        print(f"检测完成！异常分数: {result['anomaly_score']:.4f}")
        print(f"结果JSON已保存: {json_path}")
        
        return output_info


def detect_anomaly(image_path, class_name, output_dir='./output', 
                   threshold=0.7, device='cuda', min_region_area=50,
                   model_name='ViT-B-16-plus-240', image_size=240,
                   abnormal_texts=None):
    """
    一键异常检测函数 - 处理单张图片并生成所有输出
    
    Args:
        image_path: 输入图像路径
        class_name: 目标类别名称（如bottle、carpet等）
        output_dir: 输出目录，默认'./output'
        threshold: 异常检测阈值，默认0.7
        device: 运行设备，默认'cuda'
        min_region_area: 最小区域面积，默认50
        model_name: 模型名称，默认'ViT-B-16-plus-240'
        image_size: 图像resize尺寸，默认240
        abnormal_texts: 自定义异常文本列表（直接tokenize，不使用template）
    
    Returns:
        dict: 包含检测结果的字典，包括：
            - anomaly_score: 异常分数
            - white_bg_regions: 分割区域列表
            - result_json_path: JSON结果文件路径
    
    Examples:
        >>> # 基本使用
        >>> result = detect_anomaly('test.png', 'bottle')
        >>> print(f"异常分数: {result['anomaly_score']:.4f}")
        
        >>> # 自定义参数
        >>> result = detect_anomaly(
        ...     image_path='test.png',
        ...     class_name='capsule',
        ...     threshold=0.6,
        ...     device='cuda'
        ... )
    """
    # 创建检测器
    detector = SingleImageDetector(
        class_name=class_name,
        model_name=model_name,
        image_size=image_size,
        device=device,
        abnormal_texts=abnormal_texts
    )
    
    # 执行检测并保存结果
    result = detector.visualize_and_save(
        image_path=image_path,
        output_dir=output_dir,
        threshold=threshold,
        min_region_area=min_region_area
    )
    
    # 添加JSON路径到返回结果
    filename_base = os.path.splitext(os.path.basename(image_path))[0]
    result['result_json_path'] = os.path.abspath(os.path.join(output_dir, f"{filename_base}_result.json"))
    
    return result


def main():
    parser = argparse.ArgumentParser(description='单图像异常检测工具')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--class_name', type=str, required=True, help='目标类别名称（如bottle、carpet等）')
    parser.add_argument('--output', type=str, default='./output', help='输出目录')
    parser.add_argument('--model', type=str, default='ViT-B-16-plus-240', help='模型名称')
    parser.add_argument('--image_size', type=int, default=240, help='图像resize尺寸')
    parser.add_argument('--threshold', type=float, default=0.7, help='异常检测阈值')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备 (cuda/cpu)')
    parser.add_argument('--min_region_area', type=int, default=50, help='最小区域面积')
    parser.add_argument('--abnormal_text', type=str, action='append', default=None,
                        help='自定义异常文本描述（可多次提供，例如 --abnormal_text "crack" --abnormal_text "scratch"）')
    
    args = parser.parse_args()
    
    # 使用一键检测函数
    result = detect_anomaly(
        image_path=args.image,
        class_name=args.class_name,
        output_dir=args.output,
        threshold=args.threshold,
        device=args.device,
        min_region_area=args.min_region_area,
        model_name=args.model,
        image_size=args.image_size,
        abnormal_texts=args.abnormal_text
    )
    
    print(f"\n{'='*60}")
    print(f"检测完成！")
    print(f"{'='*60}")
    print(f"异常分数: {result['anomaly_score']:.4f}")
    print(f"检测区域数: {len(result.get('white_bg_regions', []))}")
    print(f"结果文件: {result['result_json_path']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

