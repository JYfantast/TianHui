from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import pandas as pd
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--input_file_path', type=str,
                    default="E:/6LLM/1datasets/results2/choice/our_model-7B/choice.final.result.csv",
                    help='输入文件路径')
args = parser.parse_args()
input_file_path = args.input_file_path

# 计算 precision, recall, f1
def calculate_scores(hyps, refs):

    # 计算BERT Score
    precision, recall, f1, _ = precision_recall_fscore_support(refs, hyps, average='macro')

    # 计算 accuracy
    acc = accuracy_score(refs, hyps)

    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'precision': [precision],
        'recall': [recall],
        'f1': [f1],
        'accuracy': [acc]
    })

    return result_df

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

    # 检查candidate列中空行的数量
    empty_candidate_count = data['candidate'].isna().sum()
    print(f"candidate列中空行的数量: {empty_candidate_count}")

    # 处理 candidate 列中的 "</think>"
    if data['candidate'].str.contains('</think>').any():
        data['candidate'] = data['candidate'].apply(lambda x: x.split('</think>')[1] if '</think>' in x else x)
        print("已处理 candidate 列中的 '</think>' 标记。")
    else:
        print("candidate 列中未找到 '</think>' 标记，未进行处理。")

    # 将data['candidate']每行不是A或B或C或D或E的行改为F
    data['candidate'] = data['candidate'].apply(lambda x: x if x in ['A', 'B', 'C', 'D', 'E'] else 'F')

    # 提取hyps和refs列的数据
    hyps = data['candidate'].tolist()
    refs = data['answer'].tolist()
    merged_result_df = calculate_scores(hyps, refs)

    # 添加model列
    merged_result_df.insert(0, 'model', model_name)

    # 构建输出文件路径
    output_file_path = f'E:\\6LLM\\6evaluation\\results2\\{project}\\results\\{model_name}.csv'

    # 导出为新的 CSV 文件
    merged_result_df.to_csv(output_file_path, index=False)

# 使用示例
if __name__ == "__main__":
    main()
