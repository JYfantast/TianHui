from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch
from peft import PeftModel

def model_tokenizer_baichuan2(model_namepath):
    """
    加载模型和分词器。

    参数:
    model_namepath (str): 模型和分词器的路径。

    返回:
    tuple: 包含模型和分词器的元组。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_namepath, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_namepath, device_map="auto", trust_remote_code=True,torch_dtype=torch.bfloat16)
    model.generation_config = GenerationConfig.from_pretrained(model_namepath)
    return model, tokenizer

def model_tokenizer_lora_baichuan2(model_namepath,lora_path):
    """
    加载模型和分词器。

    参数:
    model_namepath (str): 模型和分词器的路径。

    返回:
    tuple: 包含模型和分词器的元组。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_namepath, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_namepath, device_map="auto", trust_remote_code=True,torch_dtype=torch.bfloat16)
    model.generation_config = GenerationConfig.from_pretrained(model_namepath)
    model = PeftModel.from_pretrained(model, lora_path)
    return model, tokenizer

def response_baichuan2(model, tokenizer, system_prompt, user_input):
    """
    生成回复。

    参数:
    model (AutoModelForCausalLM): 已加载的模型。
    tokenizer (AutoTokenizer): 已加载的分词器。
    system_prompt (str): 系统提示信息。
    user_input (str): 用户输入信息。

    返回:
    str: 生成的回复。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    response = model.chat(tokenizer, messages)
    return response
