#!/usr/bin/env python3
"""
整合 train1600sft-md.json 和 train1600_with_messages.json
将 train1600_with_messages.json 中的 message_list 替换到 train1600sft-md.json 中对应的记录
"""
import json
from pathlib import Path

def main():
    base_path = Path("/NEW_EDS/miaojw/projects/Pixel-Reasoner/ad-dt")
    
    # 输入文件
    file_sft = base_path / "train1600sft-md.json"
    file_messages = base_path / "train1600_with_messages.json"
    
    # 输出文件
    output_file = base_path / "train1600sft-md-merged.json"
    
    print(f"正在加载 {file_sft}...")
    with open(file_sft, "r", encoding="utf-8") as f:
        data_sft = json.load(f)
    print(f"已加载 {len(data_sft)} 条记录")
    
    print(f"正在加载 {file_messages}...")
    with open(file_messages, "r", encoding="utf-8") as f:
        data_messages = json.load(f)
    print(f"已加载 {len(data_messages)} 条记录")
    
    # 建立 {qid: message_list} 映射
    qid_to_message_list = {item["qid"]: item["message_list"] for item in data_messages}
    
    # 替换 message_list
    replaced_count = 0
    not_found_qids = []
    
    for item in data_sft:
        qid = item.get("qid")
        if qid:
            if qid in qid_to_message_list:
                item["message_list"] = qid_to_message_list[qid]
                replaced_count += 1
            else:
                not_found_qids.append(qid)
        else:
            print(f"警告: 记录缺少 qid 字段，无法替换: {item.get('id', 'unknown')}")
    
    # 保存结果
    print(f"\n替换完成！")
    print(f"- 成功替换: {replaced_count}/{len(data_sft)} 条记录")
    print(f"- 未找到匹配: {len(not_found_qids)} 条记录")
    
    if not_found_qids:
        print(f"\n未找到匹配的 qid（前10个）:")
        for qid in not_found_qids[:10]:
            print(f"  - {qid}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_sft, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到 {output_file}")

if __name__ == "__main__":
    main()

