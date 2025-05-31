from modelscope import AutoTokenizer, AutoModelForCausalLM

def model_tokenizer_Llama3(model_namepath):
    """
    加载模型和分词器。

    参数:
    model_namepath (str): 模型和分词器的路径。

    返回:
    tuple: 包含模型和分词器的元组。
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_namepath, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_namepath)
    return model, tokenizer


def response_Llama3(model, tokenizer, system_prompt, user_input):
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
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=8192,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]

    return tokenizer.decode(response, skip_special_tokens=True)



