from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline

def model_tokenizer_bianque2(model_namepath):
    """
    加载模型和分词器。

    参数:
    model_namepath (str): 模型和分词器的路径。

    返回:
    tuple: 包含模型和分词器的元组。
    """
    pipe = pipeline(task=Tasks.chat, model=model_namepath, model_revision='v1.0.0')
    return pipe

def response_bianque2(pipe, system_prompt, user_input):
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
    result = pipe(input_text)
    return result['response']

