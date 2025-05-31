# -*- encoding: utf-8 -*-
'''
Filename         :提取生成中药-方剂信息.py
Description      :
Time             :2024/03/25 13:35:08
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
import json
from copy import deepcopy
import argparse
import pandas as pd


herb_dir = "E:/6LLM/6evaluation/results2/acupoint/our_model-7B/unique_rank_list2.txt"
formula_dir = "E:/6LLM/6evaluation/results2/acupoint/our_model-7B/unique_rank_list2.txt"

# data_path = "E:/6LLM/6evaluation/results2/acupoint/our_model-7B/acupoint.20250325_070056.result.1.json"
# save_path = "E:/6LLM/6evaluation/results2/acupoint/our_model-7B/acupoint.20250325_070056.result.4.json"

def build_index(list, text):
    data = []
    for l in list:
        idx = text.index(l)
        data.append((idx, l))
    data = sorted(data, key=lambda x: x[0])
    return [x[1] for x in data]

def remove_dump(list):
    ## 去除重复
    list = sorted(list, key=lambda x: len(x), reverse=True)
    arr = []
    for i in list:
        if len(arr)==0:
            arr.append(i)
            continue
        else:
            arr_copy = deepcopy(arr)
            flag = True
            for j in arr_copy:
                if i in j or j in i:
                    flag = False
                    break
            if flag:
                arr.append(i)
    return arr

def build_dictionary(data_dir):
    with open(data_dir, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [i.strip() for i in lines if i.strip() != ""]
        line_list = []
        for line in lines:
            words = line.split("|||")[-1]
            line_list.extend(words.split("、"))
        return list(set(line_list))


parser = argparse.ArgumentParser(description="Process data paths.")
parser.add_argument("--data_path", type=str,
                    default="E:/6LLM/6evaluation/results2/acupoint/gpt-3.5-turbo/acupoint.final.result.1.json")
parser.add_argument("--save_path", type=str,
                    default="E:/6LLM/6evaluation/results2/acupoint/gpt-3.5-turbo/acupoint.final.result.2.json")
args = parser.parse_args()
data_path = args.data_path
save_path = args.save_path


herb_list = build_dictionary(herb_dir)
formula_list = build_dictionary(formula_dir)
herb_list = [i for i in herb_list if not pd.isna(i)]
formula_list = [i for i in formula_list if not pd.isna(i)]

contents_json = json.load(open(data_path, "r", encoding="utf-8"))
all_contents = []
for idx, data in enumerate(contents_json):
    if "rank" not in data:
        print(data)
        raise ValueError 
    candidate = data["candidate"]
    #tishi = data["tishi"]
    rank = data["rank"]
    rank_list = []
    if "herb" in rank or "formula" in rank:
        rank_list.extend(rank["herb"])
        rank_list.extend(rank["formula"])
    else:
        rank_list = rank
        
    candidate_rank = []
    
    for h in herb_list:
        if h in candidate:
            candidate_rank.append(h)
    candidate_rank = remove_dump(candidate_rank)
    
    for h in formula_list:
        if h in candidate:
            candidate_rank.append(h)
    candidate_rank = remove_dump(candidate_rank)
    
    candidate_rank = build_index(candidate_rank, candidate)
    data["candidate_list"] = candidate_rank
    #data["rank_list"] = rank_list
    
    all_contents.append(data)

json.dump(all_contents, 
          open(save_path, "w", encoding="utf-8"),
          ensure_ascii=False,
          indent=2)
