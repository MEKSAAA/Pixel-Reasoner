# 自定义工具调用完整指南 🛠️

## 📋 你的需求

- 两轮对话
- 每次 assistant 回答后调用一次工具
- 有自己的数据格式

## 🏗️ 现有架构分析

### 1. 工具的定义方式

工具使用 `qwen_agent.tools.base.BaseTool` 类定义：

```python
from qwen_agent.tools.base import BaseTool, register_tool

@register_tool("your_tool_name")  # 注册工具名称
class YourTool(BaseTool):
    @property
    def description(self):
        """工具的描述，会告诉模型这个工具是做什么的"""
        return "Your tool description"
    
    # 参数定义（JSON Schema格式）
    parameters = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "参数1的描述"
            },
            "param2": {
                "type": "number",
                "description": "参数2的描述"
            }
        },
        "required": ["param1"]  # 必填参数
    }
    
    def call(self, param1, param2=None):
        """实际执行工具的函数"""
        # 你的工具逻辑
        result = do_something(param1, param2)
        return result
```

### 2. 工具的组织结构

在 `experience_maker.py` 第 1452 行：

```python
# 定义所有可用的工具
self.operations = dict(
    crop_image_normalized=CropImageNormalized(), 
    select_frames=SelectFrames()
)

# 生成工具列表（传给prompt生成器）
self.tools = [self.operations[k].function for k in ['crop_image_normalized', 'select_frames']]
```

### 3. 工具调用的检测机制

**关键点**：工具调用不是代码主动触发的，而是检测模型生成的文本

```python
# 第 2244 行：检测是否有工具调用
require_tool = last_string.endswith("</tool_call>")

# 如果有工具调用：
if require_tool:
    # 解析工具参数
    tool_params = parse_last_tool(qatext)
    tool_name = tool_params['name']
    tool_args = tool_params['arguments']
    
    # 执行工具
    result = execute_tool(...)
```

---

## 🎯 针对你的需求：两轮对话方案

### 方案1：主动工具调用（推荐）

如果你想**主动控制**在每次assistant回答后调用工具（而不是依赖模型生成`</tool_call>`），可以这样做：

#### 1.1 定义你的工具

```python
# 在 experience_maker.py 开头添加你的工具类

@register_tool("your_custom_tool")
class YourCustomTool(BaseTool):
    @property
    def description(self):
        return "你的工具描述"
    
    parameters = {
        "type": "object",
        "properties": {
            "input_data": {
                "type": "string",
                "description": "输入数据"
            }
        },
        "required": ["input_data"]
    }
    
    def call(self, input_data):
        """
        你的工具逻辑
        例如：处理图片、调用API、计算等
        """
        # 示例：返回处理后的结果
        result = f"处理结果: {input_data}"
        return result
```

#### 1.2 注册工具

```python
# 在 RemoteExperienceMaker.__init__ 中（约第 1452 行）

self.operations = dict(
    crop_image_normalized=CropImageNormalized(), 
    select_frames=SelectFrames(),
    your_custom_tool=YourCustomTool()  # 添加你的工具
)

self.tools = [
    self.operations[k].function 
    for k in ['crop_image_normalized', 'select_frames', 'your_custom_tool']
]
```

#### 1.3 修改多轮对话逻辑（两轮固定调用）

在 `make_experience_list` 中（约第 2223-2360 行），修改 `while True` 循环：

```python
# 原来的代码是：
while True:  # 无限循环，直到模型不再生成 </tool_call>
    ...
    require_tool = last_string.endswith(tool_end)
    if not require_tool:
        break

# 修改为固定两轮：
MAX_TURNS = 2  # 固定两轮
for turn_idx in range(MAX_TURNS):
    print(f"========= 第 {turn_idx + 1} 轮对话 ==========")
    
    for out_idx, (qqid, out, qatext, fflag) in enumerate(...):
        if fflag: continue
        
        # 获取assistant的响应
        rsp = solutions_round0[out_idx]
        
        # ⭐ 主动调用工具（不检测 </tool_call>）
        # 方式1: 从响应中提取信息作为工具输入
        tool_input = extract_info_from_response(rsp)
        
        # 方式2: 从你的数据格式中获取工具输入
        tool_input = get_tool_input_from_data(qqid, turn_idx)
        
        # 执行工具
        tool_result = self.operations['your_custom_tool'].call(tool_input)
        
        # 将工具结果添加到对话
        msg_this = [
            dict(role='assistant', content=[
                dict(type='text', text=rsp)
            ]),
            dict(role='user', content=[
                dict(type='text', text=f"工具执行结果: {tool_result}")
            ])
        ]
        
        all_conversations[uuid] = all_conversations[uuid] + msg_this
        
        # 准备下一轮生成
        req_vllminputs.append(vllm_inputs[uuid])
    
    # 如果是最后一轮，不再生成
    if turn_idx == MAX_TURNS - 1:
        break
    
    # 提交下一轮生成请求
    # ... (使用 vLLM 生成下一轮响应)
```

---

### 方案2：基于数据格式的工具调用

如果你的数据格式本身包含了工具调用信息，可以这样：

#### 2.1 数据格式示例

```json
{
  "qid": "question_001",
  "question": "用户问题",
  "turns": [
    {
      "turn": 1,
      "assistant_response": "第一轮助手回答",
      "tool_call": {
        "name": "your_custom_tool",
        "arguments": {"input_data": "xxx"}
      }
    },
    {
      "turn": 2,
      "assistant_response": "第二轮助手回答",
      "tool_call": {
        "name": "your_custom_tool",
        "arguments": {"input_data": "yyy"}
      }
    }
  ]
}
```

#### 2.2 解析数据并调用工具

```python
# 在生成循环中

# 加载你的数据格式
data_info = load_your_data_format(qqid)

for turn_idx in range(len(data_info['turns'])):
    turn_data = data_info['turns'][turn_idx]
    
    # 获取这一轮的工具调用信息
    if 'tool_call' in turn_data:
        tool_name = turn_data['tool_call']['name']
        tool_args = turn_data['tool_call']['arguments']
        
        # 执行工具
        tool_result = self.operations[tool_name].call(**tool_args)
        
        # 添加到对话
        msg_this = [
            dict(role='assistant', content=[
                dict(type='text', text=turn_data['assistant_response'])
            ]),
            dict(role='user', content=[
                dict(type='text', text=f"工具结果: {tool_result}")
            ])
        ]
```

---

### 方案3：混合模式（模型决定 + 你控制轮数）

结合模型自主决定是否调用工具，但你限制最大轮数：

```python
MAX_TURNS = 2
turn_count = 0

while turn_count < MAX_TURNS:
    for out_idx, (...) in enumerate(...):
        rsp = solutions_round0[out_idx]
        
        # 检测模型是否想调用工具
        require_tool = rsp.endswith("</tool_call>")
        
        if require_tool:
            # 解析并执行工具
            tool_params = parse_last_tool(qatext)
            result = execute_tool(...)
        else:
            # 即使模型不想调用，也可以主动调用
            if turn_count < MAX_TURNS - 1:
                result = force_call_tool(...)
        
        # ... 添加到对话
    
    turn_count += 1
    
    # 检查是否所有对话都完成
    if all(all_flags):
        break
```

---

## 🔧 需要修改的关键位置

### 位置1: 工具定义（文件开头）

```python
文件: openrlhf/trainer/ppo_utils/experience_maker.py
位置: 第 1-200 行

# 添加你的工具类
@register_tool("your_tool")
class YourTool(BaseTool):
    ...
```

### 位置2: 工具注册（__init__ 方法）

```python
文件: openrlhf/trainer/ppo_utils/experience_maker.py
位置: 第 1452 行

self.operations = dict(
    # 原有工具
    crop_image_normalized=CropImageNormalized(),
    select_frames=SelectFrames(),
    # 添加你的工具
    your_tool=YourTool()
)
```

### 位置3: 多轮对话循环（核心逻辑）

```python
文件: openrlhf/trainer/ppo_utils/experience_maker.py
位置: 第 2223-2360 行

# 修改 while True 循环
# 改为固定轮数或其他控制逻辑
```

### 位置4: 工具调用方式

有两种方式：

**方式A: 检测模型输出（现有方式）**
```python
# 第 2244 行
require_tool = last_string.endswith("</tool_call>")
```

**方式B: 主动调用（你的需求）**
```python
# 不检测 </tool_call>，直接调用
tool_result = self.operations['your_tool'].call(your_params)
```

---

## 📝 完整示例代码

### 示例：固定两轮对话，每轮后调用自定义工具

```python
# ========== 1. 定义工具 ==========
@register_tool("custom_analyzer")
class CustomAnalyzer(BaseTool):
    @property
    def description(self):
        return "分析助手响应的自定义工具"
    
    parameters = {
        "type": "object",
        "properties": {
            "response_text": {
                "type": "string",
                "description": "助手的响应文本"
            }
        },
        "required": ["response_text"]
    }
    
    def call(self, response_text):
        # 你的分析逻辑
        analysis = f"分析结果: {len(response_text)} 个字符"
        return analysis


# ========== 2. 在 __init__ 中注册 ==========
class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.operations = dict(
            crop_image_normalized=CropImageNormalized(),
            select_frames=SelectFrames(),
            custom_analyzer=CustomAnalyzer()  # 注册你的工具
        )
        
        self.tools = [self.operations[k].function for k in self.operations.keys()]


# ========== 3. 修改生成循环 ==========
def make_experience_list(self, all_prompts, is_eval=False, **kwargs):
    # ... 前面的代码（vLLM 生成等）
    
    # 固定两轮对话
    MAX_TURNS = 2
    
    for turn_idx in range(MAX_TURNS):
        print(f"========= 第 {turn_idx + 1} 轮 ==========")
        
        req_indexlist = []
        req_vllminputs = []
        
        for out_idx, (qqid, out, qatext, fflag) in enumerate(zip(...)):
            if fflag: 
                continue
            
            uuid = idx2uid[out_idx]
            rsp = solutions_round0[out_idx]
            
            # ⭐ 主动调用工具（不检测 </tool_call>）
            tool_result = self.operations['custom_analyzer'].call(
                response_text=rsp
            )
            
            # 构建对话消息
            msg_this = [
                # Assistant 的回答
                dict(role='assistant', content=[
                    dict(type='text', text=rsp)
                ]),
                # 工具的结果（作为 user 消息）
                dict(role='user', content=[
                    dict(type='text', text=f"工具分析: {tool_result}")
                ])
            ]
            
            # 更新对话历史
            all_conversations[uuid] = all_conversations[uuid] + msg_this
            
            # 如果不是最后一轮，准备下一轮生成
            if turn_idx < MAX_TURNS - 1:
                # 更新 prompt
                new_prompt = self.prompt_maker.build_prompt(
                    all_conversations[uuid]
                )
                vllm_inputs[uuid]['prompt'] = new_prompt
                req_vllminputs.append(vllm_inputs[uuid])
                req_indexlist.append(out_idx)
        
        # 如果是最后一轮，退出
        if turn_idx == MAX_TURNS - 1:
            break
        
        # 提交下一轮生成请求
        if req_vllminputs:
            # 使用 vLLM 生成下一轮响应
            refs = []
            for i, llm in enumerate(llms):
                batch_inputs = req_vllminputs[i*batch_size:(i+1)*batch_size]
                refs.append(llm.add_requests_vlm.remote(
                    rank, 
                    sampling_params=sampling_params, 
                    vllm_vision_input=batch_inputs
                ))
            ray.get(refs)
            
            # 获取生成结果
            all_outputs_new = get_vllm_responses(llms, rank)
            
            # 更新 solutions_round0 为下一轮的输入
            for idx, new_out in zip(req_indexlist, all_outputs_new):
                solutions_round0[idx] = self.tokenizer.decode(
                    new_out.outputs[0].token_ids
                )
    
    # ... 后续处理（构建 Experience 对象等）
```

---

## 🎨 适配你的数据格式

### 数据加载

如果你有自己的数据格式，需要在这里加载：

```python
class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 加载你的数据格式
        self.custom_data = self.load_custom_data()
    
    def load_custom_data(self):
        """加载你的自定义数据"""
        data_path = self.strategy.args.custom_data_path
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # 构建 qid -> data 的映射
        qid2data = {item['qid']: item for item in data}
        return qid2data
    
    def get_tool_params_from_data(self, qid, turn_idx):
        """根据 qid 和轮次获取工具参数"""
        data = self.custom_data.get(qid, {})
        turns = data.get('turns', [])
        
        if turn_idx < len(turns):
            return turns[turn_idx].get('tool_params', {})
        return {}
```

---

## 🚦 调试建议

### 1. 先测试工具本身

```python
# 单独测试你的工具
tool = YourCustomTool()
result = tool.call(your_params)
print(f"工具结果: {result}")
```

### 2. 添加调试日志

```python
for turn_idx in range(MAX_TURNS):
    print(f"\n{'='*80}")
    print(f"第 {turn_idx + 1} 轮对话")
    print(f"当前响应: {rsp}")
    print(f"工具参数: {tool_params}")
    print(f"工具结果: {tool_result}")
    print(f"{'='*80}\n")
```

### 3. 使用断点

```python
if turn_idx == 0 and out_idx == 0:
    import pdb; pdb.set_trace()
```

---

## 📋 总结清单

- [ ] 定义你的工具类（继承 BaseTool）
- [ ] 在 `__init__` 中注册工具
- [ ] 决定工具调用方式：
  - [ ] 方式A: 检测模型输出的 `</tool_call>`
  - [ ] 方式B: 主动调用（固定轮数）
  - [ ] 方式C: 从数据格式中读取
- [ ] 修改多轮对话循环逻辑
- [ ] 处理工具结果（添加到对话历史）
- [ ] 测试和调试

---

## ❓ 需要考虑的问题

1. **工具输入从哪里来？**
   - 从模型响应中提取？
   - 从你的数据格式中读取？
   - 固定的参数？

2. **工具输出如何使用？**
   - 添加到对话历史继续生成？
   - 直接作为最终结果？
   - 需要格式化吗？

3. **轮数控制**
   - 固定两轮？
   - 可变轮数（根据某些条件）？
   - 每轮都必须调用工具吗？

4. **图片/视频处理**
   - 你的工具需要处理图片吗？
   - 如果需要，如何获取图片？

---

如果你告诉我：
1. 你的数据格式是什么样的
2. 工具的具体功能是什么
3. 两轮对话的具体流程

我可以给你更具体的实现方案！

