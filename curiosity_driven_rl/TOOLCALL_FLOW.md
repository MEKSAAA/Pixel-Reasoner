# ToolCall 完整调用流程 🔄

## 推理时 ToolCall 的完整链路

### 📍 核心原理

**ToolCall 不是主动调用的，而是模型生成的！**

流程：
1. 先让模型自由生成文本
2. 检查生成的文本是否以 `</tool_call>` 结尾
3. 如果是，则解析工具参数并执行
4. 将工具结果添加到对话历史，继续生成
5. 循环直到模型不再调用工具

---

## 🔢 完整的调用顺序（从头到尾）

### 1️⃣ 入口：eval_ray.py (第244行)
```python
文件: openrlhf/cli/eval_ray.py
行号: 244

evaluator.evaluate(0)
```

↓

### 2️⃣ Evaluator2.evaluate() 
```python
文件: openrlhf/trainer/ray/evaluator2.py
行号: 132-163

def evaluate(self, global_steps):
    # 加载评估数据
    eval_data = blending_datasets(...)
    self.eval_data = PromptDataset(...)
    
    # 调用父类的 evaluate
    status = super().evaluate(args, self.eval_data)
```

↓

### 3️⃣ Evaluator.evaluate()
```python
文件: openrlhf/trainer/evaluator.py
行号: 313-349

def evaluate(self, args, eval_data) -> None:
    # 创建 dataloader
    eval_dataloader = self.strategy.setup_dataloader(eval_data, ...)
    
    # 调用 eval_unit 执行推理
    info = self.eval_unit(args, 0, self.eval_step, eval_dataloader)
```

↓

### 4️⃣ Evaluator.eval_unit()
```python
文件: openrlhf/trainer/evaluator.py
行号: 174-209

def eval_unit(self, args, ep, global_step, dataloader):
    for idx, rand_prompts in enumerate(dataloader):
        # 🔥 关键：调用 get_explist_from_prompts 生成响应
        exp_list = self.get_explist_from_prompts(
            args, ep, rand_prompts, 
            is_eval=True, 
            eval_step=global_step
        )
```

↓

### 5️⃣ Evaluator.get_explist_from_prompts()
```python
文件: openrlhf/trainer/evaluator.py
行号: 269-310

def get_explist_from_prompts(self, args, ep, all_prompts, ...):
    # 调用 experience_maker 生成 experience
    return self.experience_maker.make_experience_list(
        all_prompts, 
        is_eval=is_eval, 
        eval_step=eval_step, 
        **generate_kwargs
    )
```

↓

### 6️⃣ ExperienceMaker.make_experience_list() - 第一次生成
```python
文件: openrlhf/trainer/ppo_utils/experience_maker.py
行号: 2110-2205

# ⭐ 第一步：调用 vLLM 生成初始响应（第 2110-2195 行）
sampling_params = SamplingParams(temperature=..., max_tokens=...)

# 提交请求到 vLLM 引擎
for i, llm in enumerate(llms):
    refs.append(
        llm.add_requests_vlm.remote(rank, sampling_params=sampling_params, ...)
    )

# 获取生成结果
all_outputs = sum(ray.get(all_output_refs), [])

# ⭐ 第二步：解码生成的文本（第 2203-2208 行）
solutions_round0 = self.tokenizer.batch_decode(all_outputs_, skip_special_tokens=False)
all_qa_texts = [question + solution for question, solution in zip(questions, solutions_round0)]
```

↓

### 7️⃣ 检测工具调用 - **这是最先检测 ToolCall 的地方！**
```python
文件: openrlhf/trainer/ppo_utils/experience_maker.py
行号: 2223-2244

# 🔥 关键位置：这是推理时第一次检测工具调用的地方！

while True:  # 第 2223 行 - 多轮对话循环
    print(f"========= niter {niter}")
    
    for out_idx, (qqid, out, qatext, fflag) in enumerate(...):
        # 获取生成的响应
        rsp = solutions_round0[out_idx].replace("<|im_end|>","")
        
        # ⭐⭐⭐ 第 2243-2244 行：检测是否需要调用工具
        last_string = rsp[-len(tool_end)-10:] if len(rsp)>len(tool_end)+10 else rsp
        require_tool = last_string.endswith(tool_end)  # 检查是否以 </tool_call> 结尾
        
        # 如果检测到工具调用，继续执行...
```

**这就是最先检测 toolcall 的地方！第 2244 行**

↓

### 8️⃣ 解析工具参数
```python
文件: openrlhf/trainer/ppo_utils/experience_maker.py
行号: 2272-2275

if require_tool:  # 如果需要调用工具
    try:
        # 🔥 第 2272 行：解析工具调用参数
        tool_params = parse_last_tool(qatext)
        tool_name = tool_params['name']
        tool_args = tool_params['arguments']
```

↓

### 9️⃣ 执行工具
```python
文件: openrlhf/trainer/ppo_utils/experience_maker.py
行号: 2276

# 🔥 第 2276 行：执行工具
raw_result = execute_tool(
    imagelist, rawimagelist, 
    tool_args, tool_name, 
    is_video=video_flag, 
    function=self.operations[tool_name].call
)
```

↓

### 🔟 工具执行函数
```python
文件: openrlhf/trainer/ppo_utils/experience_maker.py
行号: 1170-1240

def execute_tool(images, rawimages, args, toolname, is_video, function=None):
    """实际执行工具的地方"""
    
    if toolname == 'select_frames':
        # 从视频中选择帧
        selected_frames = function(candidates, target_frames)
        return selected_frames, message
    
    else:  # crop_image_normalized
        # 裁剪图片
        image_to_crop = images[index]
        cropped = function(image_to_crop, bbox_2d)
        return cropped
```

↓

### 1️⃣1️⃣ 继续下一轮生成
```python
文件: openrlhf/trainer/ppo_utils/experience_maker.py
行号: 2290-2360

# 将工具结果添加到对话历史
msg_this.append(
    dict(role='user', content=[
        dict(type='image', image=added_image),
        dict(type='text', text="Here is the cropped image...")
    ])
)

# 更新对话
all_conversations[uuid] = all_conversations[uuid] + msg_this

# 准备下一轮请求
req_vllminputs.append(vllm_inputs[uuid])

# niter += 1, 回到步骤 7，继续循环
```

---

## 🎯 最关键的代码位置总结

### **最先检测 ToolCall 的位置：**

```python
文件: openrlhf/trainer/ppo_utils/experience_maker.py
行号: 2243-2244

last_string = rsp[-len(tool_end)-10:] if len(rsp)>len(tool_end)+10 else rsp
require_tool = last_string.endswith(tool_end)  # ⭐ 这是最先检测的地方！
```

### **工具调用的三个核心步骤：**

1. **检测** (第 2244 行)：`require_tool = last_string.endswith(tool_end)`
2. **解析** (第 2272 行)：`tool_params = parse_last_tool(qatext)`
3. **执行** (第 2276 行)：`raw_result = execute_tool(...)`

---

## 📊 时间线视图

```
时间 →

[启动] eval_ray.py
   ↓
[初始化] Evaluator.evaluate()
   ↓
[遍历数据] Evaluator.eval_unit()
   ↓
[生成请求] get_explist_from_prompts()
   ↓
[vLLM生成] 第 2110-2195 行
   ↓ (生成包含 <tool_call> 的文本)
   ↓
[检测工具] 第 2244 行 ⭐ require_tool = ...  ← 最先在这里！
   ↓
[解析工具] 第 2272 行 → parse_last_tool()
   ↓
[执行工具] 第 2276 行 → execute_tool()
   ↓
[添加结果] 第 2290-2360 行
   ↓
[继续生成] 回到 vLLM生成
   ↓
[循环直到] 模型不再生成 </tool_call>
```

---

## 🔍 推荐的调试断点位置

如果你想从最开始追踪 toolcall，按照这个顺序添加断点：

### 断点顺序：

**1. 在 vLLM 生成后** (第 2200 行)：
```python
print(f"===> [verbose] decode and evaluate the initial round of responses")

# 🔍 断点：查看第一次生成的结果
print(f"\n{'='*80}")
print(f"第一次生成完成，共 {len(solutions_round0)} 个响应")
if solutions_round0:
    print(f"第一个响应: {solutions_round0[0][:200]}...")
print(f"{'='*80}\n")
import pdb; pdb.set_trace()
```

**2. 在检测工具调用处** (第 2244 行)：
```python
require_tool = last_string.endswith(tool_end)

# 🔍 断点：检测到工具调用
if require_tool and out_idx == 0:
    print(f"\n{'='*80}")
    print(f"⭐ 检测到工具调用！")
    print(f"响应末尾: {last_string}")
    print(f"{'='*80}\n")
    import pdb; pdb.set_trace()
```

**3. 在解析工具参数处** (第 2272 行)：
```python
tool_params = parse_last_tool(qatext)

# 🔍 断点：解析工具参数
if out_idx == 0:
    print(f"⭐ 工具参数: {tool_params}")
    import pdb; pdb.set_trace()
```

**4. 在执行工具处** (第 2276 行)：
```python
raw_result = execute_tool(...)

# 🔍 断点：执行工具
if out_idx == 0:
    print(f"⭐ 工具执行完成")
    import pdb; pdb.set_trace()
```

---

## 💡 关键理解

1. **模型决定是否调用工具**：不是代码主动调用，而是模型在生成文本时决定是否输出 `</tool_call>`

2. **多轮对话**：工具调用可能发生多轮，每次执行工具后继续生成，直到模型决定不再调用

3. **检测机制**：通过简单的字符串匹配 `endswith(tool_end)` 来检测

4. **所有代码在一个文件**：整个工具调用逻辑都在 `experience_maker.py` 中

---

## 🎓 示例：一次完整的工具调用

```
用户问题: "图片右上角的数字是多少？"
   ↓
[vLLM 生成] 
输出: "我需要放大图片的右上角区域。<tool_call>{"name":"crop_image_normalized","arguments":{"bbox_2d":[0.7,0,1,0.3],"target_image":1}}</tool_call>"
   ↓
[检测] 第 2244 行
require_tool = True  ← 检测到 </tool_call>
   ↓
[解析] 第 2272 行
tool_name = "crop_image_normalized"
tool_args = {"bbox_2d": [0.7, 0, 1, 0.3], "target_image": 1}
   ↓
[执行] 第 2276 行
执行裁剪，返回放大后的图片
   ↓
[添加到对话]
"user: [裁剪后的图片] Here is the cropped image..."
   ↓
[继续生成]
输出: "我看到数字是 42。"
   ↓
[检测]
require_tool = False  ← 没有 </tool_call>，结束
```

