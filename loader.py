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