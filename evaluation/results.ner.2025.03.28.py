# -*- encoding: utf-8 -*-
'''
Filename         :tcm_entity.py
Description      :
Time             :2024/03/19 09:16:40
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

import json
import argparse
from loguru import logger
import re
import pandas as pd

def calculate_f1_score(true_entities, predicted_entities):
    true_entities = set(true_entities)
    predicted_entities = set(predicted_entities)
    true_positive = len(true_entities & predicted_entities)
    precision = true_positive / len(predicted_entities) if len(predicted_entities) > 0 else 0
    recall = true_positive / len(true_entities) if len(true_entities) > 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # print(precision,recall,f1_score)
    return precision,recall,f1_score

def calculate_accuracy(question_data):
    correct_count_p = 0
    correct_count_r = 0
    correct_count_f = 0
    total_count = len(question_data)
    for i, question in enumerate(question_data):
        #true_output = [i.split(":")[-1] for i in question['output'].split("；")]

        if isinstance(question['output'], str):
            true_output = [i.split(":")[-1] for i in question['output'].split("；")]
        else:
            true_output = []


        true_outputs = []
        for tt in true_output:
            true_outputs.extend([i.strip() for i in tt.split("，") if i.strip() != ""])
        #my_output = [i.split(":")[-1] for i in question['candidate'].split("；")]

        if isinstance(question['candidate'], str):
            my_output = [i.split(":")[-1] for i in question['candidate'].split("；")]
        else:
            my_output = []



        my_outputs = []
        for tt in my_output:
            my_outputs.extend([i.strip() for i in tt.split("，") if i.strip() != ""])

        precision, recall, f1_score = calculate_f1_score(true_outputs, my_outputs)
        #print(f1_score)
        correct_count_p += precision
        correct_count_r += recall
        correct_count_f += f1_score

    return correct_count_p / total_count, correct_count_r / total_count, correct_count_f / total_count


def run(args):
    data_dir = args.input_file_path
    save_dir = args.save_dir
    # with open(data_dir, "r") as json_file:
    #     json_data = json.load(json_file)
    #     p, r, f = calculate_accuracy(json_data)

    # 读取CSV并转换为JSON字符串，然后解析为字典
    #json_data_str = pd.read_csv(data_dir).to_json(orient='records', lines=True)
    #json_data_str = pd.read_csv(data_dir).to_json(orient='records', lines=True, force_ascii=False)
    json_data_str = pd.read_csv(data_dir)

    # 检查candidate列中空行的数量
    empty_candidate_count = json_data_str['candidate'].isna().sum()
    print(f"candidate列中空行的数量: {empty_candidate_count}")


    # 处理 candidate 列中的 "</think>"
    if json_data_str['candidate'].str.contains('</think>').any():
        json_data_str['candidate'] = json_data_str['candidate'].apply(lambda x: x.split('</think>')[1] if '</think>' in x else x)
        print("已处理 candidate 列中的 '</think>' 标记。")
    else:
        print("candidate 列中未找到 '</think>' 标记，未进行处理。")


    # 将DataFrame转换为JSON格式
    json_data_str = json_data_str.to_json(orient='records', lines=True, force_ascii=False)



    json_data = [json.loads(line) for line in json_data_str.splitlines()]
    # json_data = json.load(open(data_dir, "r", encoding="utf-8"))

    p, r, f = calculate_accuracy(json_data)
    logger.info(f"precision: {p:.3f}，recall: {r:.3f}，f1_score: {f:.3f}")
    if save_dir:
        with open(save_dir, "w") as json_file:
            json.dump({"precision": p, "recall": r, "f1_score": f}, json_file)

    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'precision': [p],
        'recall': [r],
        'f1_score': [f],
    })

    # 提取模型名称
    model_name_match = re.search(r'\\([^\\]+)\\[^\\]+\.result.csv$', data_dir.replace('/', '\\'))
    if model_name_match:
        model_name = model_name_match.group(1)
        print(f"模型名称: {model_name}")
    else:
        raise ValueError("无法从输入文件路径中提取模型名称")

    # 提取 project 名称
    project_match = re.search(r'results2\\([^\\]+)\\', data_dir.replace('/', '\\'))
    if project_match:
        project = project_match.group(1)
    else:
        raise ValueError("无法从输入文件路径中提取项目名称")

    # 添加model列
    result_df.insert(0, 'model', model_name)

    # 构建输出文件路径
    output_file_path = f'E:\\6LLM\\6evaluation\\results2\\{project}\\results\\{model_name}.csv'

    # 导出为新的 CSV 文件
    result_df.to_csv(output_file_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_path', type=str, default="E:/6LLM/6evaluation/results2/NER/DeepSeek-R1-Distill-Qwen-7B/NER.20250325_134407.result.csv")
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    run(args)
