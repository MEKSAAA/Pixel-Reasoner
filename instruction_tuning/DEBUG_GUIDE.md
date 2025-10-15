# PDB 调试指南

## 🎯 快速导航

**想查看VLM的输出？直接跳到这两个断点：**
- ⭐ **断点4.5**: VLM输入准备完成 - 查看送入模型的数据
- ⭐⭐ **断点5**: VLM输出 - 查看模型预测结果和logits（最重要！）

---

## 已添加的断点位置

### 断点1: `sft_tool.py` - main函数入口 (第215行)
**位置**: main函数开始处  
**目的**: 查看所有启动参数  
**可以检查的内容**:
```python
# 在pdb中输入这些命令查看:
print(script_args)      # 脚本参数
print(training_args)    # 训练参数
print(model_args)       # 模型参数
print(model_args.model_name_or_path)  # 模型路径
```

### 断点2: `sft_tool.py` - 数据加载完成 (第234行)
**位置**: 数据集加载后  
**目的**: 查看数据集内容和格式  
**可以检查的内容**:
```python
# 查看数据集
print(len(train_dataset))        # 数据集大小
print(train_dataset[0])          # 第一个样本
print(train_dataset.data[:3])    # 前3个样本

# 查看数据结构
import json
print(json.dumps(train_dataset[0], indent=2, ensure_ascii=False))
```

### 断点3: `sft_tool.py` - Trainer初始化完成 (第261行)
**位置**: trainer创建后，训练开始前  
**目的**: 查看trainer配置和模型状态  
**可以检查的内容**:
```python
# 查看trainer配置
print(trainer.args)              # 训练参数
print(trainer.model)             # 模型结构
print(trainer.vlm_module)        # VLM模块

# 查看模型参数
total_params = sum(p.numel() for p in trainer.model.parameters())
trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
print(f"总参数: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")

# 查看冻结的参数
for name, param in trainer.model.named_parameters():
    if not param.requires_grad:
        print(f"冻结: {name}")
```

### 断点4: `sft_tooltrainer.py` - compute_loss开始 (第372行)
**位置**: 每个训练batch的loss计算前  
**目的**: 查看实际输入数据和处理流程  
**可以检查的内容**:
```python
# 查看batch内容
print(len(inputs))               # batch size
print(inputs[0].keys())          # 数据字段
print(message_lists[0])          # 第一个消息列表

# 查看设备和分布式信息
print(device)
print(self.accelerator.process_index)
```

### 断点4.5: `sft_tooltrainer.py` - VLM输入准备完成 (第386行)
**位置**: VLM输入准备完成后，模型forward前  
**目的**: 查看传入VLM的输入张量结构  
**可以检查的内容**:
```python
# 查看VLM输入结构
print(inputs.keys())                    # 输入字段（如input_ids, pixel_values等）
print(inputs.input_ids.shape)           # 输入token的形状
print(inputs.attention_mask.shape)      # attention mask形状

# 如果有图像输入
if hasattr(inputs, 'pixel_values'):
    print(inputs.pixel_values.shape)    # 图像特征形状
if hasattr(inputs, 'image_grid_thw'):
    print(inputs.image_grid_thw.shape)  # 图像网格信息

# 查看实际的input tokens（解码查看）
batch_idx = 0
tokens = inputs.input_ids[batch_idx]
decoded = self.processing_class.tokenizer.decode(tokens)
print(f"输入文本:\n{decoded}")
```

### 断点5: `sft_tooltrainer.py` - VLM输出 (第233行) ⭐ 重点
**位置**: VLM模型forward后，获得logits  
**目的**: 查看VLM的实际输出和预测结果  
**可以检查的内容**:
```python
# 查看输出形状
print(f"Logits shape: {logits.shape}")  # (batch_size, seq_len, vocab_size)
print(f"Input IDs shape: {input_ids.shape}")

# 查看模型预测的token（贪婪解码）
predicted_ids = logits.argmax(dim=-1)   # 获取概率最高的token
print(f"Predicted IDs shape: {predicted_ids.shape}")

# 解码查看模型预测的文本
batch_idx = 0
predicted_text = self.processing_class.tokenizer.decode(predicted_ids[batch_idx])
actual_text = self.processing_class.tokenizer.decode(input_ids[batch_idx])
print(f"\n实际输入的文本:\n{actual_text}")
print(f"\n模型预测的文本:\n{predicted_text}")

# 查看特定位置的概率分布
pos = 10  # 查看第10个位置
top_k = 5
top_probs, top_indices = logits[batch_idx, pos].softmax(dim=-1).topk(top_k)
print(f"\n位置{pos}的top-{top_k}预测:")
for prob, idx in zip(top_probs, top_indices):
    token = self.processing_class.tokenizer.decode([idx])
    print(f"  {token}: {prob.item():.4f}")

# 查看loss相关的token
# 找到assistant的回复部分
assistant_mask = logits_to_keep[batch_idx]
assistant_positions = assistant_mask.nonzero().flatten()
print(f"\nAssistant回复的token位置: {assistant_positions.tolist()}")
if len(assistant_positions) > 0:
    start = assistant_positions[0].item()
    end = min(start + 20, len(input_ids[batch_idx]))
    assistant_text = self.processing_class.tokenizer.decode(input_ids[batch_idx][start:end])
    print(f"Assistant回复内容: {assistant_text}")
```

## 运行调试

### 1. 启动调试脚本
```bash
cd /NEW_EDS/miaojw/projects/Pixel-Reasoner
bash instruction_tuning/sft-test-debug.sh
```

### 2. PDB常用命令

| 命令 | 说明 | 示例 |
|------|------|------|
| `n` (next) | 执行下一行 | `n` |
| `s` (step) | 进入函数内部 | `s` |
| `c` (continue) | 继续执行到下一个断点 | `c` |
| `p` (print) | 打印变量 | `p train_dataset` |
| `pp` (pretty print) | 格式化打印 | `pp train_dataset[0]` |
| `l` (list) | 显示当前代码 | `l` |
| `w` (where) | 显示调用栈 | `w` |
| `u` (up) | 上一层调用栈 | `u` |
| `d` (down) | 下一层调用栈 | `d` |
| `q` (quit) | 退出调试 | `q` |
| `h` (help) | 帮助 | `h` |

### 3. 高级技巧

#### 查看变量类型和属性
```python
type(train_dataset)
dir(train_dataset)
vars(train_dataset)
```

#### 执行多行代码
```python
!import json
!with open('/tmp/debug.json', 'w') as f:
!    json.dump(train_dataset.data[0], f, indent=2)
```

#### 条件断点（在代码中设置）
```python
if condition:
    import pdb; pdb.set_trace()
```

#### 动态修改变量
```python
(Pdb) script_args.per_device_train_batch_size = 2
```

## 理解项目流程

1. **启动流程** (sft_tool.py):
   - 解析命令行参数 → 加载数据集 → 初始化VLM模块 → 创建Trainer → 开始训练

2. **训练流程** (sft_tooltrainer.py):
   - compute_loss被循环调用
   - 每个batch: 获取消息列表 → 准备VLM输入 → 计算logits → 计算loss

3. **关键组件**:
   - `VLMModule`: 封装模型的处理逻辑
   - `SFT_DATASET`: 数据集类
   - `Qwen2VLSFTToolTrainer`: 训练器类

## 快速调试检查清单

### 在断点1检查:
- [ ] 模型路径是否正确
- [ ] 数据路径是否正确
- [ ] 训练参数是否符合预期

### 在断点2检查:
- [ ] 数据集是否成功加载
- [ ] 数据格式是否正确
- [ ] message_list结构是否符合预期

### 在断点3检查:
- [ ] 模型是否正确加载
- [ ] vision模块是否正确冻结
- [ ] 可训练参数数量是否正确

### 在断点4检查:
- [ ] batch数据是否正确
- [ ] 图像是否正确加载
- [ ] 消息格式是否正确

### 在断点4.5检查:
- [ ] VLM输入张量形状是否正确
- [ ] pixel_values是否包含图像特征
- [ ] input_ids解码后的文本是否符合预期

### 在断点5检查（⭐ 查看VLM输出的最佳位置）:
- [ ] logits形状是否正确 (batch, seq_len, vocab_size)
- [ ] 模型预测的文本是否合理
- [ ] 预测概率分布是否正常
- [ ] assistant回复部分是否被正确识别

## 移除断点

如果想移除某个断点，注释掉或删除对应的这一行:
```python
import pdb; pdb.set_trace()  # 断点X: ...
```

## 注意事项

1. **单卡调试**: 调试脚本已配置为单卡模式（CUDA_VISIBLE_DEVICES=1）
2. **内存使用**: 调试时注意GPU内存，可能需要减小batch_size
3. **断点4会频繁触发**: 每个batch都会停在断点4，如果不需要可以注释掉
4. **分布式训练**: 如果要调试多卡，需要使用debugpy而不是pdb

## 进阶: 使用debugpy进行远程调试

如果需要在VSCode中调试多卡训练，可以使用已经导入的debugpy（见sft_tool.py第18-53行）

