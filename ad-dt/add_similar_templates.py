#!/usr/bin/env python3
import json
import os

# 读取 mmad.json
print("读取 mmad.json...")
mmad_path = "/NEW_EDS/miaojw/projects/MMAD/origin/mmad.json"
with open(mmad_path, 'r', encoding='utf-8') as f:
    mmad_data = json.load(f)

print(f"mmad.json 包含 {len(mmad_data)} 个条目")

# 读取当前样本文件
print("\n读取 test6400md.json...")
test_file = "/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt/train366grpo-md-new.json"
with open(test_file, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"train366grpo-md-new.json.json 包含 {len(test_data)} 个样本")

# 定义基础路径前缀
base_prefix = "/NEW_EDS/miaojw/projects/MMAD/"

# 统计信息
matched_count = 0
not_found_count = 0
no_similar_templates_count = 0
invalid_file_count = 0
invalid_files = []

def convert_to_absolute_path(relative_path, base_prefix):
    """将相对路径转换为全局路径"""
    if relative_path.startswith('/'):
        # 已经是绝对路径
        return relative_path
    
    if relative_path.startswith('MVTec-AD/'):
        # 转换为 DS-MVTec 路径
        relative_path = relative_path.replace('MVTec-AD/', 'DS-MVTec/')
    
    # 构建完整路径
    full_path = os.path.join(base_prefix, relative_path)
    # 标准化路径（处理多余的斜杠等）
    full_path = os.path.normpath(full_path)
    
    return full_path

def convert_to_save_path(absolute_path):
    """将绝对路径转换为保存格式：保持绝对路径，但将DS-MVTec替换为MVTec-AD"""
    # 将路径中的所有DS-MVTec替换为MVTec-AD，保持绝对路径格式
    return absolute_path.replace('DS-MVTec/', 'MVTec-AD/')

# 为每个样本添加 similar_templates 字段
for i, sample in enumerate(test_data):
    image_path = sample.get('image', '')
    
    if not image_path:
        sample['similar_templates'] = []
        not_found_count += 1
        continue
    
    # 从完整路径中提取相对路径
    if image_path.startswith(base_prefix):
        relative_path = image_path[len(base_prefix):]
    else:
        # 尝试直接使用路径
        relative_path = image_path
    
    # 在 mmad.json 中查找对应的条目
    if relative_path in mmad_data:
        mmad_entry = mmad_data[relative_path]
        
        # 检查是否有 similar_templates 字段
        if 'similar_templates' in mmad_entry and mmad_entry['similar_templates']:
            # 将相对路径转换为完整路径，保存所有路径（不检查文件是否存在）
            similar_templates_full = []
            for template_path in mmad_entry['similar_templates']:
                # 转换为全局路径
                full_path = convert_to_absolute_path(template_path, base_prefix)
                
                # 转换为保存格式：保持绝对路径，但将DS-MVTec替换为MVTec-AD
                save_path = convert_to_save_path(full_path)
                similar_templates_full.append(save_path)
                
                # 统计不存在的文件（仅用于统计，不影响保存）
                if not os.path.exists(full_path):
                    invalid_file_count += 1
                    invalid_files.append(full_path)
            
            sample['similar_templates'] = similar_templates_full
            matched_count += 1
        else:
            # 没有 similar_templates 字段，设置为空数组
            sample['similar_templates'] = []
            no_similar_templates_count += 1
    else:
        # 在 mmad.json 中未找到，设置为空数组
        sample['similar_templates'] = []
        not_found_count += 1
    
    # 每处理 1000 个样本打印一次进度
    if (i + 1) % 1000 == 0:
        print(f"已处理 {i + 1}/{len(test_data)} 个样本...")

# 保存更新后的文件
print("\n保存更新后的文件...")
output_file = "/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt/train366grpo-md-new.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("\n完成！")
print(f"成功匹配并添加 similar_templates: {matched_count}")
print(f"未在 mmad.json 中找到: {not_found_count}")
print(f"找到但无 similar_templates 字段: {no_similar_templates_count}")
print(f"不存在的文件数量: {invalid_file_count}")
print(f"总计处理: {len(test_data)} 个样本")

if invalid_files:
    print(f"\n不存在的文件示例（前10个）：")
    for f in invalid_files[:10]:
        print(f"  - {f}")
    
    # 将所有不存在的文件保存到文件
    invalid_files_output = "/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt/invalid_files.txt"
    with open(invalid_files_output, 'w', encoding='utf-8') as f:
        for file_path in invalid_files:
            f.write(f"{file_path}\n")
    print(f"\n所有不存在的文件已保存到: {invalid_files_output}")
    print(f"共 {len(invalid_files)} 个不存在的文件")

