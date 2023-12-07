import pandas as pd
import csv
# Load data
test_ids = pd.read_csv('F:/CODES/Python/UCAS-Machine-Learning/DataSet/test_ids.csv')
train_labels = pd.read_pickle('F:/CODES/Python/UCAS-Machine-Learning/DataSet/train_label.pkl')

# Find corresponding values and concatenate them into a long list
result_list = []
for id in test_ids['ids']:
    if id in train_labels:
        result_list.extend(train_labels[id])

data = [[value] for value in result_list]

# 指定CSV文件路径
csv_file_path = "output.csv"

# 写入CSV文件
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)