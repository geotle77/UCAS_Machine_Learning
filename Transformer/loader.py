import pickle
import pandas as pd
DATA_FILE_PATH = "./DataSet/"

with open(DATA_FILE_PATH + 'text_features.pkl', 'rb') as f:
    text_features = pickle.load(f)

# 提取以"dialog+数字ID"为键的二维数组
dialog_ID_features = {k: v for k, v in text_features.items() if k.startswith('dialog_') and k[7:].isdigit()}

# 打印提取的数据
for key, value in dialog_ID_features.items():
    print(f"{key}: {value}")

import csv

# 打开CSV文件
with open('./DataSet/train_ids.csv', 'r') as file:
    # 创建CSV读取器
    csv_reader = csv.reader(file)

    # 从每一行中获取第一列数据，并将其存储在一个列表中
    first_column_data = [row[0] for row in csv_reader]

# 打印或使用得到的列表
first_column_data.pop(0)
print(first_column_data)
