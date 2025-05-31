import pandas as pd
import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str,
                    default="E:/6LLM/1datasets/results2/abstract/results",
                    help='结果文件路径')
parser.add_argument('--out_file', type=str,
                    default="all_results.csv",
                    help='保存文件')
args = parser.parse_args()
folder_path = args.folder_path

# 提取folder_path中的abstract部分
abstract_part = re.search(r'/([^/]+)/results$', folder_path).group(1)

# 构造新的文件名
out_file = f"{abstract_part}.{args.out_file}"

# 获取文件夹下所有CSV文件的路径
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

# 检查是否有CSV文件
if not csv_files:
    raise FileNotFoundError("文件夹中没有找到CSV文件")

# 读取所有CSV文件并合并
merged_df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

# 定义输出文件路径
output_file_path = os.path.join(folder_path, out_file)

# 导出合并后的DataFrame为CSV文件
merged_df.to_csv(output_file_path, index=False)

print(f"合并后的CSV文件已保存到 {output_file_path}")
