import pickle
import pandas as pd
DATA_FILE_PATH = "./DataSet/"

# 打开并加载pkl文件
with open(DATA_FILE_PATH+'text_features.pkl', 'rb') as f:
    text_features = pickle.load(f)

# 打印加载的数据
print(text_features)