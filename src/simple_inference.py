#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import torch
import base64
from io import BytesIO
from PIL import Image

# 条件导入，根据选择的推理引擎
try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    transformers_available = True
except ImportError:
    transformers_available = False
    print("警告: transformers相关库未安装，无法使用transformers引擎")

try:
    from vllm import LLM, SamplingParams
    vllm_available = True
except ImportError:
    vllm_available = False
    print("警告: vllm相关库未安装，无法使用vllm引擎")

# 合并 qwen_vl_utils 的代码
def process_vision_info(messages):
    """处理多模态消息中的图像和视频信息

    Args:
        messages: 包含图像或视频的消息列表

    Returns:
        images_data: 处理后的图像数据
        videos_data: 处理后的视频数据
    """
    images_list, videos_list = [], []
    for message in messages:
        content = message.get("content", None)
        if isinstance(content, str):
            # 纯文本消息，不处理
            continue
        elif isinstance(content, list):
            # 混合消息，可能包含图像或视频
            for item in content:
                if not isinstance(item, dict):
                    continue
                
                # 处理图像
                if item.get("type") == "image" and "image" in item:
                    image = item["image"]
                    if isinstance(image, str):
                        # 图像URL或路径，尝试加载
                        try:
                            image = Image.open(image)
                        except Exception as e:
                            print(f"图像加载失败: {e}")
                            continue
                    
                    # 转换PIL图像为base64编码
                    if isinstance(image, Image.Image):
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        images_list.append(image_str)
                
                # 处理视频（如有需要）
                elif item.get("type") == "video" and "video" in item:
                    # 暂不支持视频
                    pass
                    
    return images_list or None, videos_list or None

def predict_location(
    image_path, 
    model_name="Qwen/Qwen2.5-VL-7B-Instruct", 
    inference_engine="transformers"
):
    """
    对单个图片进行位置识别预测
    
    参数:
        image_path: 图片文件路径
        model_name: 模型名称或路径
        inference_engine: 推理引擎，"vllm" 或 "transformers"
        
    返回:
        预测结果文本
    """
    # 检查图片是否存在
    if not os.path.exists(image_path):
        return f"错误: 图片文件不存在: {image_path}"
    
    # 加载图片
    try:
        image = Image.open(image_path)
        print(f"成功加载图片: {image_path}")
    except Exception as e:
        return f"错误: 无法加载图片: {str(e)}"
    
    # 加载处理器
    print(f"加载处理器: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, padding_side='left')
    
    # 构建提示消息 - 简化版本，没有SFT和COT
    question_text = "In which country and within which first-level administrative region of that country was this picture taken?Please answer in the format of <answer>$country,administrative_area_level_1$</answer>?"
    system_message = "You are a helpful assistant good at solving problems with step-by-step reasoning. You should first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags."
    
    # 构建简化后的提示消息
    prompt_messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_message}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question_text}
            ]
        }
    ]
    
    # 根据选定的引擎进行推理
    if inference_engine == "vllm":
        if not vllm_available:
            return "错误: vLLM库不可用，请安装vllm或选择transformers引擎"
        
        # 使用vLLM进行推理
        print(f"使用vLLM加载模型: {model_name}")
        llm = LLM(
            model=model_name,
            limit_mm_per_prompt={"image": 10, "video": 10},
            dtype="auto",
            gpu_memory_utilization=0.95,
        )
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            max_tokens=2048,
            stop_token_ids=[],
        )
        
        # 处理消息为vLLM格式
        prompt = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # 处理图像数据
        image_inputs, video_inputs = process_vision_info(prompt_messages)
        
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        
        # 构建vLLM输入
        llm_input = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        
        # 生成回答
        outputs = llm.generate([llm_input], sampling_params=sampling_params)
        response = outputs[0].outputs[0].text
        
    else:  # transformers
        if not transformers_available:
            return "错误: Transformers相关库不可用，请安装必要的包"
        
        # 使用transformers加载模型
        print(f"使用transformers加载模型: {model_name}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="cuda:0"
        )
        
        # 准备输入
        text = processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        
        # 处理输入
        inputs = processor(
            text=text,
            images=prompt_messages[1]['content'][0]['image'],
            return_tensors="pt",
        )
        
        inputs = inputs.to(model.device)
        
        # 生成回答
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=2048)
        
        # 处理输出
        generated_ids_trimmed = generated_ids[0][len(inputs['input_ids'][0]):]
        response = processor.decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n=== 推理结果 ===")
    print(response)
    print("=================\n")
    
    return response

if __name__ == "__main__":
    # 命令行参数设置
    parser = argparse.ArgumentParser(description='对单个图片进行位置识别预测')
    parser.add_argument('--image_path', type=str, required=True,
                        help='图片文件路径')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help='模型名称或路径')
    parser.add_argument('--inference_engine', type=str, default="transformers", choices=["vllm", "transformers"],
                        help='推理引擎: vllm 或 transformers')
    
    args = parser.parse_args()
    
    # 单个图片推理
    result = predict_location(
        image_path=args.image_path,
        model_name=args.model_name,
        inference_engine=args.inference_engine
    )
    
    print(f"最终预测结果: {result}")

# 使用示例:
# python simple_inference.py --image_path /data/phd/tiankaibin/dataset/data/streetview_images_first_tier_cities/testaccio_rome_italy_h45_r100_20250317_183133.jpg --model_name TheEighthDay/SeekWorld_RL_PLUS --inference_engine vllm 