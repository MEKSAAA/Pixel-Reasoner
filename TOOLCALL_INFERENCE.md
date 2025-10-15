# 工具调用推理说明

## 🎯 设计理念

模拟训练时的工具调用流程，但**不在 prompt 中显式提及工具**。在 assistant 回答后强制触发工具调用。

## 📊 对比两种方案

### 方案 1: `inference-multi-turn.py`（原方案）
```
User: 请生成异常描述
  ↓
Assistant: 生成 JSON 描述
  ↓
[Python 代码直接调用 CLIP]
  ↓
User: 这是 ROI，请检测异常
  ↓
Assistant: 检测结果
```

**特点**：
- ✅ 代码简洁直观
- ❌ 与训练时的工具调用流程不一致
- ❌ 没有模拟工具调用的对话结构

### 方案 2: `inference-with-toolcall.py`（新方案）⭐
```
User: 请生成异常描述
  ↓
Assistant: 生成 JSON 描述
  ↓
[强制工具调用: generate_roi_from_clip]
  ↓
Tool Returns: ROI 图片 + 边界框
  ↓
User: [包含 ROI 的新消息] 请检测异常
  ↓
Assistant: 检测结果
```

**特点**：
- ✅ 模拟训练时的工具调用流程
- ✅ 保持完整的多轮对话历史
- ✅ 工具调用不在 prompt 中提及（隐式调用）
- ✅ 更符合训练时的交互模式

## 🔑 核心区别

### Prompt 设计

**方案 1**（显式流程）：
- 第一轮：告诉模型生成异常描述
- 第二轮：告诉模型基于 ROI 检测异常

**方案 2**（隐式工具调用）：
- 第一轮：告诉模型生成异常描述（同方案 1）
- **中间**：强制触发工具（模型不知道，但流程中发生了）
- 第二轮：告诉模型基于 ROI 检测异常（同方案 1）

**关键**：prompt 中**不提及工具**，但在对话流程中**模拟工具调用**。

### 对话历史结构

**方案 1**：
```python
messages = [
    {"role": "system", "content": [{"text": "stage1 system"}]},
    {"role": "user", "content": [{"text": "stage1 user"}, {"image": "orig.png"}]},
    {"role": "assistant", "content": [{"text": "stage1 response"}]},
    {"role": "system", "content": [{"text": "stage2 system"}]},
    {"role": "user", "content": [{"text": "stage2 user"}, {"image": "orig.png"}, {"image": "roi1.png"}]}
]
```

**方案 2**（完全相同）：
```python
# 第一轮
messages_stage1 = [
    {"role": "system", "content": [{"text": "stage1 system"}]},
    {"role": "user", "content": [{"text": "stage1 user"}, {"image": "orig.png"}]}
]

# 获取 assistant 回复
assistant_response = model.generate(messages_stage1)

# [强制工具调用]
tool_result = execute_roi_tool(abnormal_descriptions)  # 不在对话中，直接执行

# 第二轮（包含第一轮历史）
messages_stage2 = messages_stage1 + [
    {"role": "assistant", "content": [{"text": assistant_response}]},
    {"role": "system", "content": [{"text": "stage2 system"}]},
    {"role": "user", "content": [{"text": "stage2 user"}, {"image": "orig.png"}, {"image": "roi1.png"}]}
]
```

**核心思想**：工具调用发生在对话之外，但其结果（ROI）会出现在下一轮对话中。

## 💡 训练时的工具调用流程参考

根据 `curiosity_driven_rl` 中的实现：

1. **模型生成时可能包含工具调用**：
```
Assistant: 我需要放大图片来看清楚细节<tool_call>{"name": "crop_image_normalized", "arguments": {"bbox_2d": [0.1, 0.2, 0.5, 0.6], "target_image": 1}}</tool_call>
```

2. **代码检测到 `</tool_call>` 标签**：
```python
require_tool = last_string.endswith("</tool_call>")
```

3. **执行工具并返回结果**：
```python
if require_tool:
    tool_params = parse_last_tool(qatext)
    result = execute_tool(tool_params)
    # 将结果添加到对话
    messages.append({
        "role": "user",
        "content": [{"text": f"工具执行结果: {result}"}]
    })
```

4. **模型基于工具结果继续生成**

## 🎨 新方案的实现

### 不在 Prompt 中提及工具

**错误示例（训练时的做法）**：
```python
system_prompt = """You have access to tools: crop_image_normalized.
Use <tool_call>{"name": "...", "arguments": {...}}</tool_call> to call tools."""
```

**正确示例（推理时的做法）**：
```python
system_prompt = """You are a vision expert.
Provide detailed visual descriptions."""  # 不提及工具
```

### 强制工具调用

```python
class ToolCallDetector:
    def run_with_toolcall(self, ...):
        # 1. 第一轮对话
        response1 = self.generate_response(messages_stage1)
        
        # 2. 解析响应，提取关键信息
        abnormal_descriptions = extract_descriptions(response1)
        
        # 3. 强制触发工具（不管模型有没有要求）
        tool_result = self.execute_roi_tool(abnormal_descriptions)
        
        # 4. 第二轮对话（包含工具结果）
        messages_stage2 = messages_stage1 + [
            {"role": "assistant", "content": [{"text": response1}]},
            # 工具结果通过图片形式传入，不显式说明是工具返回的
            {"role": "user", "content": [
                {"text": "基于这些区域检测异常"},
                {"image": original_image},
                {"image": roi_image_1},
                {"image": roi_image_2}
            ]}
        ]
        
        response2 = self.generate_response(messages_stage2)
```

## 📝 使用方法

### 运行新方案
```bash
python inference-with-toolcall.py
```

### 自定义使用
```python
from inference_with_toolcall import ToolCallDetector

# 初始化
detector = ToolCallDetector(MODEL_PATH, VLM_GPU_ID, CLIP_GPU_ID)

# 运行带工具调用的推理
result = detector.run_with_toolcall(
    class_name="bottle",
    image_path="test.png",
    anomaly_categories=["crack", "contamination"]
)

# 访问结果
print(result['stage1_response'])    # 第一轮回复
print(result['tool_result'])        # 工具调用结果
print(result['stage2_response'])    # 第二轮回复
```

## 🔍 关键代码片段

### 工具执行（不在对话中）

```python
def execute_roi_tool(self, class_name, image_path, abnormal_descriptions, ...):
    """
    执行 ROI 生成工具
    这相当于训练时的 crop_image 工具调用
    """
    print("🛠️  工具调用：生成 ROI 区域")
    
    # 调用 CLIP
    result = detect_anomaly(
        image_path=image_path,
        class_name=class_name,
        abnormal_texts=abnormal_descriptions,  # 从 stage1 提取
        ...
    )
    
    return {
        'roi_images': [...],
        'roi_bboxes': [...],
        'anomaly_score': ...
    }
```

### 多轮对话构建

```python
# 第一轮
messages_stage1 = [
    {"role": "system", "content": [{"text": system_prompt_stage1}]},
    {"role": "user", "content": [{"text": user_prompt}, {"image": image_path}]}
]

assistant_response_stage1 = model.generate(messages_stage1)

# 强制工具调用（在对话外）
tool_result = self.execute_roi_tool(...)

# 第二轮（包含第一轮历史）
messages_stage2 = messages_stage1 + [
    {"role": "assistant", "content": [{"text": assistant_response_stage1}]},
    {"role": "system", "content": [{"text": system_prompt_stage2}]},
    {"role": "user", "content": [
        {"text": user_prompt_stage2},
        {"image": original_image},
        *[{"image": roi} for roi in roi_images]  # 工具返回的 ROI
    ]}
]

assistant_response_stage2 = model.generate(messages_stage2)
```

## 🎯 适用场景

### 使用方案 1（`inference-multi-turn.py`）

- ✅ 快速原型开发
- ✅ 简单的推理需求
- ✅ 不需要与训练流程完全一致

### 使用方案 2（`inference-with-toolcall.py`）

- ✅ 需要模拟训练时的工具调用流程
- ✅ 研究模型的工具使用能力
- ✅ 与训练数据格式保持一致
- ✅ 为后续集成真正的工具调用做准备

## 🚀 扩展性

方案 2 可以很容易扩展为真正的工具调用：

```python
# 当前：强制工具调用
def run_with_toolcall(self, ...):
    response1 = self.generate_response(messages_stage1)
    # 强制调用工具
    tool_result = self.execute_roi_tool(...)
    
# 未来：检测工具调用
def run_with_toolcall(self, ...):
    response1 = self.generate_response(messages_stage1)
    
    # 检测是否有 </tool_call> 标签
    if "</tool_call>" in response1:
        # 解析工具调用
        tool_params = parse_tool_call(response1)
        # 执行工具
        tool_result = self.execute_tool(tool_params)
    else:
        # 没有工具调用，直接结束
        tool_result = None
```

## 📌 总结

| 特性 | 方案 1 | 方案 2 |
|------|--------|--------|
| Prompt 中提及工具 | ❌ | ❌ |
| 模拟工具调用流程 | ❌ | ✅ |
| 多轮对话历史 | ✅ | ✅ |
| 与训练流程一致 | 部分 | 完全 |
| 代码复杂度 | 简单 | 中等 |
| 扩展性 | 低 | 高 |

**推荐**：如果希望与训练时的工具调用流程保持一致，使用**方案 2**。

---

**更新日期**: 2025-10-15  
**版本**: 1.0

