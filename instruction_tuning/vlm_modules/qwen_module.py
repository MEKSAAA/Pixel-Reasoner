from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor,Qwen2_5_VLProcessor,Qwen2VLImageProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
import json
from vlm_modules.vlm_module import VLMBaseModule
from PIL import Image
from typing import List
from qwen_agent.llm.schema import Message
from qwen_vl_utils import process_vision_info
class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()
    def get_question_template(self):
        return "{Question} Please think step by step. Put your final answer in \\boxed{{}}"

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self):

        return Qwen2_5_VLForConditionalGeneration
    
    def post_model_init(self, model, processing_class):
        pass
    def get_model(self,model_id:str,model_init_kwargs:dict,**kwargs):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id,**model_init_kwargs)
        return model
    def get_processor(self,model_id:str,**kwargs):
        processor = AutoProcessor.from_pretrained(model_id)
        pad_token_id = processor.tokenizer.pad_token_id
        processor.pad_token_id = pad_token_id
        processor.eos_token_id = processor.tokenizer.eos_token_id
        processor.padding_side = "left"
        processor.tokenizer.padding_side = "left"
        if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
            processor.image_processor.max_pixels = kwargs.get("max_pixels", 28*28*5120)
            processor.image_processor.min_pixels = kwargs.get("min_pixels", 3136)
        return processor
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    def tool_des_postprocess(self,func_sets:str)->str:

        return func_sets

    def if_is_tool_call(self,output_text:str)->bool:
        return '<tool_call>' in output_text
    def if_is_think(self,message_lists:List[List[Message]])->bool:
        if_is_think = []
        for message_list in message_lists:
            found_not_think = False
            for message in message_list:
                if message.role == 'assistant':
                
                    for content in message.content:
                        if content.type == 'text':
                            if '</think>' not in content.text :
                                if_is_think.append(False)
                                found_not_think = True
                                break
                if found_not_think:
                        break
            if not found_not_think:
                if_is_think.append(True)
        return if_is_think
    def get_tool_param_key(self):
        return "arguments"
    def load_tool(self,output_text:str)->str:
        """
        Load the tool from the output text.
        """
        return json.loads(output_text.split('<tool_call>')[1].split('</tool_call>')[0])
    def prepare_from_msg_2_vlm_inputs(self,processor:AutoProcessor,messages,add_wait_IV:bool=False,video_max_pixels:int=420*360,**kwargs)->dict[str,Union[torch.Tensor,Any]]:
        if add_wait_IV:
            messages[0].append({'role':'user', 'content':[{'image':'/home/ma-user/work/haozhe/muze/modelartsdata/2000-4000/2002/2.jpg'},{'video':['/home/ma-user/work/haozhe/muze/modelartsdata/2000-4000/2002/2.jpg']}]})
        
        # 预处理：将图片列表展开为多个单独的图片项
        # messages 是批次级别的：[[msg1, msg2, ...], [msg1, msg2, ...], ...]
        for sample_messages in messages:  # 遍历批次中的每个样本
            for message in sample_messages:  # 遍历每个样本的消息
                if 'content' in message:
                    new_content = []
                    for content_item in message['content']:
                        if 'image' in content_item:
                            # 如果 image 是列表，展开为多个单独的 content 项
                            if isinstance(content_item['image'], list):
                                for img_path in content_item['image']:
                                    new_content.append({'image': img_path})
                            else:
                                # 如果已经是字符串，直接添加
                                new_content.append(content_item)
                        elif 'video' in content_item:
                            # 处理 video：如果是列表，需要包装
                            video = content_item['video']
                            if isinstance(video, list):
                                video_content = {'video': video, 'max_pixels': video_max_pixels}
                            else:
                                video_content = content_item
                            new_content.append(video_content)
                        else:
                            # 非图片/视频的 content（如 text）直接添加
                            new_content.append(content_item)
                    message['content'] = new_content
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        def add_think_tag(text:str)->str:
            return text +'<think>'
        

        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
        
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        return inputs
    # def 

    

    def get_im_start(self):
        return "<|im_start|>"
    def get_im_end(self):
        return "<|im_end|>"
    def get_assistant(self):
        return "assistant"