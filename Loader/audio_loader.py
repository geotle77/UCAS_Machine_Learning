import numpy as np
import pickle
import pandas as pd
DATA_FILE_PATH = 'F:/CODES/Python/UCAS-Machine-Learning/UCAS-Machine-Learning/data/'
# Load the pickle file
with open(DATA_FILE_PATH + 'text_features_v1.pkl', 'rb') as f:
    data = pickle.load(f)
len_list =[]
sum = 0
for key in data.keys():
    print(key, len(data[key]))
    len_list.append(len(data[key]))
    # print(data[key])
    sum +=len(data[key])
    print('-----------------')
print(len_list)