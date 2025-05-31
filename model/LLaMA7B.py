from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def model_tokenizer_lora_LLaMA7B(model_namepath,lora_path):
    """
    加载模型和分词器。

    参数:
    model_namepath (str): 模型和分词器的路径。

    返回:
    tuple: 包含模型和分词器的元组。
    """
    model = AutoModelForCausalLM.from_pretrained(model_namepath)
    tokenizer = AutoTokenizer.from_pretrained(model_namepath)
    model = PeftModel.from_pretrained(model, lora_path)
    return model, tokenizer

def response_LLaMA7B(model, tokenizer, system_prompt, user_input):
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
    # 截断 input_text，确保其不超过1024个字符
    if len(input_text) > 1024:
        input_text = input_text[-1024:]
    print(f"问题 : {input_text}")
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"])
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
