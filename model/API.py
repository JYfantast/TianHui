from openai import OpenAI
import time

def response_API(base_url, api_key, model_namepath, system_prompt, input_text):
    """
    使用OpenAI API生成回复。

    参数:
    base_url (str): OpenAI API的基础URL。
    api_key (str): OpenAI API的密钥。
    model_namepath (str): 模型的名称或路径。
    system_prompt (str): 系统提示信息。
    input_text (str): 用户输入信息。

    返回:
    str: 生成的回复。
    """
    client = OpenAI(base_url=base_url, api_key=api_key)
    max_retries = 5
    retries = 0

    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_namepath,  # this field is currently unused
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {"role": "user", "content": input_text}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Attempt {retries + 1} failed: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Waiting for 10 seconds before retrying...")
                time.sleep(10)
            else:
                print("All retries failed.")
                raise e
