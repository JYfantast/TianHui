from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_chinese import Rouge
from bert_score import score
import pandas as pd
import argparse
import jieba
import re

parser = argparse.ArgumentParser()
parser.add_argument('--input_file_path', type=str,
                    default="E:/6LLM/6evaluation/results2/pharmacological/DeepSeek-R1-Distill-Qwen-7B/pharmacological.20250327_163752.result.csv",
                    help='输入文件路径')
args = parser.parse_args()
input_file_path = args.input_file_path

# 计算累计BLEU
def cumulative_bleu(candidate, reference):

    bleu_1_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu_2_gram = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    bleu_3_gram = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu_4_gram = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram

# 计算BLEU分数
def calculate_bleu_scores(hyps,refs):
    # 初始化累加器
    total_bleu_1_gram = 0
    total_bleu_2_gram = 0
    total_bleu_3_gram = 0
    total_bleu_4_gram = 0

    # 遍历每一行并计算BLEU分数
    for hyp, ref in zip(hyps, refs):
        bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram = cumulative_bleu(hyp, [ref])
        total_bleu_1_gram += bleu_1_gram
        total_bleu_2_gram += bleu_2_gram
        total_bleu_3_gram += bleu_3_gram
        total_bleu_4_gram += bleu_4_gram

    # 计算平均值
    num_samples = len(hyps)
    avg_bleu_1_gram = total_bleu_1_gram / num_samples
    avg_bleu_2_gram = total_bleu_2_gram / num_samples
    avg_bleu_3_gram = total_bleu_3_gram / num_samples
    avg_bleu_4_gram = total_bleu_4_gram / num_samples

    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'avg_bleu_1_gram': [avg_bleu_1_gram],
        'avg_bleu_2_gram': [avg_bleu_2_gram],
        'avg_bleu_3_gram': [avg_bleu_3_gram],
        'avg_bleu_4_gram': [avg_bleu_4_gram]
    })

    return result_df

# 计算bert_scores分数
def calculate_bert_scores(hyps, refs):

    # 计算BERT Score
    P, R, F1 = score(hyps, refs, lang="zh", verbose=True)

    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'bert_score_P': [P.mean().item()],
        'bert_score_R': [R.mean().item()],
        'bert_score_F1': [F1.mean().item()]
    })

    return result_df

# 计算ROUGE分数
def calculate_rouge_scores(hyps, refs):

    # 初始化Rouge对象
    rouge = Rouge()

    # 计算ROUGE分数
    scores = rouge.get_scores(hyps, refs, avg=True)

    # 提取分数
    rouge_1_r = scores['rouge-1']['r']
    rouge_1_p = scores['rouge-1']['p']
    rouge_1_f = scores['rouge-1']['f']
    rouge_2_r = scores['rouge-2']['r']
    rouge_2_p = scores['rouge-2']['p']
    rouge_2_f = scores['rouge-2']['f']
    rouge_l_r = scores['rouge-l']['r']
    rouge_l_p = scores['rouge-l']['p']
    rouge_l_f = scores['rouge-l']['f']

    # 创建结果DataFrame
    rouge_scores_result_df = pd.DataFrame({
        'rouge_1_r': [rouge_1_r],
        'rouge_1_p': [rouge_1_p],
        'rouge_1_f': [rouge_1_f],
        'rouge_2_r': [rouge_2_r],
        'rouge_2_p': [rouge_2_p],
        'rouge_2_f': [rouge_2_f],
        'rouge_l_r': [rouge_l_r],
        'rouge_l_p': [rouge_l_p],
        'rouge_l_f': [rouge_l_f],
    })

    return rouge_scores_result_df

# 计算METEOR分数
def calculate_meteor_scores(hyps, refs):
    # 初始化累加器
    total_meteor_score = 0

    # 定义预处理函数
    def preprocess_text(text):
        return ' '.join(jieba.cut(text)).split()

    # 遍历每一行并计算METEOR分数
    for ref, hyp in zip(refs, hyps):
        processed_reference = preprocess_text(ref)
        processed_candidate = preprocess_text(hyp)
        score = meteor_score([processed_reference], processed_candidate)
        total_meteor_score += score

    # 计算平均METEOR分数
    num_samples = len(hyps)
    avg_meteor_score = total_meteor_score / num_samples

    # 创建结果DataFrame
    meteor_scores_result_df = pd.DataFrame({
        'avg_meteor_score': [avg_meteor_score]
    })
    return meteor_scores_result_df

def main():

    # 提取模型名称
    model_name_match = re.search(r'\\([^\\]+)\\[^\\]+\.result\.csv$', input_file_path.replace('/', '\\'))
    if model_name_match:
        model_name = model_name_match.group(1)
    else:
        raise ValueError("无法从输入文件路径中提取模型名称")

    # 提取 project 名称
    project_match = re.search(r'results2\\([^\\]+)\\', input_file_path.replace('/', '\\'))

    if project_match:
        project = project_match.group(1)
    else:
        raise ValueError("无法从输入文件路径中提取项目名称")

    # 读取CSV文件
    data = pd.read_csv(input_file_path)


    # 处理 candidate 列中的 "</think>"
    if data['candidate'].str.contains('</think>').any():
        data['candidate'] = data['candidate'].apply(lambda x: x.split('</think>')[1] if '</think>' in x else x)
        print("已处理 candidate 列中的 '</think>' 标记。")
    else:
        print("candidate 列中未找到 '</think>' 标记，未进行处理。")

    # 检查candidate列中空行的数量
    empty_candidate_count = data['candidate'].isna().sum()
    print(f"candidate列中空行的数量: {empty_candidate_count}")

    # # 删除 candidate 和 output 列中每行的空格，并将空值替换为 "无无无无无"
    # data['candidate'] = data['candidate'].str.replace(' ', '').fillna('无无无无无')
    # data['output'] = data['output'].str.replace(' ', '').fillna('无无无无无')

    # 将 candidate 列中的空行或只有空格的行替换为 "无无无"
    data['candidate'] = data['candidate'].apply(lambda x: "无" if pd.isna(x) or x.strip() == "" else x)

    # 删除 candidate 列中的换行符
    data['candidate'] = data['candidate'].str.replace('\n', '', regex=False)

    # 提取hyps和refs列的数据
    hyps = data['candidate'].tolist()
    refs = data['output'].tolist()

    # 计算BLEU分数
    BLEU_result_df = calculate_bleu_scores(hyps, refs)

    # 计算BERT分数
    Bert_scores_result_df = calculate_bert_scores(hyps, refs)

    # 计算ROUGE分数
    hyps_tokenized = [' '.join(jieba.cut(hyp)) for hyp in hyps]
    refs_tokenized = [' '.join(jieba.cut(ref)) for ref in refs]
    Rouge_scores_result_df = calculate_rouge_scores(hyps_tokenized, refs_tokenized)

    # 计算METEOR分数
    Meteor_scores_result_df = calculate_meteor_scores(hyps, refs)

    # 合并两个DataFrame
    merged_result_df = pd.concat([BLEU_result_df, Bert_scores_result_df, Rouge_scores_result_df,
                                  Meteor_scores_result_df], axis=1)

    # 添加model列
    merged_result_df.insert(0, 'model', model_name)

    # 构建输出文件路径
    output_file_path = f'E:\\6LLM\\6evaluation\\results2\\{project}\\results\\{model_name}.csv'

    # 导出为新的 CSV 文件
    merged_result_df.to_csv(output_file_path, index=False)

# 使用示例
if __name__ == "__main__":
    main()
