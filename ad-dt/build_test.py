import json
import os

# 读取文件
with open('mvtec_meta.json', 'r', encoding='utf-8') as f:
    mvtec_meta = json.load(f)

with open('class_category.json', 'r', encoding='utf-8') as f:
    class_category = json.load(f)['mvtec']

# 数据集基础路径
dataset_base_path = "/NEW_EDS/miaojw/datasets/mvtec_dataset"

# 生成测试数据
result = []
qid_counter = 0

##Candidate anomaly types:
##{anomaly_bullet_points}

for class_name, items in mvtec_meta.items():
    # 获取该类别的所有异常类型
    anomaly_types = class_category.get(class_name, {})
    anomaly_list = sorted(list(anomaly_types.values()))
    
    # 构建候选异常类型的字符串
    anomaly_bullet_points = "\n".join([f"- {anomaly}" for anomaly in anomaly_list])
    
    # 构建问题模板
    question_template = f"""Evaluate the following image from the class "{class_name}".


Determine if the object is normal or abnormal.Follow the instruction and output the final answer in the required XML format."""
    
    # 处理每个测试样本
    for item in items:
        img_path = item['img_path']
        anomaly = item['anomaly']
        
        # 构建完整图片路径
        full_img_path = os.path.join(dataset_base_path, img_path)
        
        # 创建测试样本
        test_sample = {
            "qid": f"mvtec_test_{qid_counter}",
            "question": question_template,
            "image": full_img_path,
            "class_name": class_name,
            "anomaly_list": anomaly_list,
            "gt_answer": bool(anomaly)
        }
        
        result.append(test_sample)
        qid_counter += 1

# 保存结果
output_file = 'mvtec_agent_test_new.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"转换完成！共生成 {len(result)} 条测试数据")
print(f"结果已保存到: {output_file}")

# 打印一些统计信息
print("\n各类别数据统计:")
class_count = {}
for item in result:
    class_name = item['class_name']
    class_count[class_name] = class_count.get(class_name, 0) + 1

for class_name, count in sorted(class_count.items()):
    print(f"  {class_name}: {count} 条")

