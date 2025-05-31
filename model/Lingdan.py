
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

def model_tokenizer_Lingdan(model_namepath):
    """
    加载模型和分词器。

    参数:
    model_namepath (str): 模型和分词器的路径。

    返回:
    tuple: 包含模型和分词器的元组。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_namepath, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_namepath, device_map="auto", trust_remote_code=True)
    return model, tokenizer

def response_Lingdan(model, tokenizer, system_prompt, user_input):
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
    input_text = system_prompt + "\n" + user_input
    inputs = tokenizer(input_text, return_tensors='pt')
    pred = model.generate(**inputs)
    return tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)

