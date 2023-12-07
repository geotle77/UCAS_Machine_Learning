import numpy as np
import pickle
import pandas as pd
DATA_FILE_PATH = 'F:/CODES/Python/UCAS-Machine-Learning/DataSet/'
# Load the pickle file
with open(DATA_FILE_PATH + 'audio_features.pkl', 'rb') as f:
    data = pickle.load(f)

for key in data.keys():
    print(key, len(data[key]))
    print(data[key])
    print('-----------------')