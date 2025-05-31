from modelscope import AutoTokenizer, AutoModel

def model_tokenizer_chatglm3(model_path):
    """
    加载模型和tokenizer

    :param model_path: 模型路径
    :return: tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
    model = model.eval()
    return model, tokenizer

def response_chatglm3(model, tokenizer, system_prompt, user_input):
    """
    生成回答

    :param tokenizer: 已加载的tokenizer
    :param model: 已加载的模型
    :param prompt: 输入的提示
    :return: 模型生成的回答
    """
    full_input = system_prompt + "\n" + user_input
    response = model.chat(tokenizer, full_input)
    return response[0]

