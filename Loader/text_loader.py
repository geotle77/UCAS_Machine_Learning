import numpy as np
import pickle
import pandas as pd
import torch
class DataLoader:
    def __init__(self, data_file_path,dataset_name):
        self.data_file_path = data_file_path
        self.data = None
        self.dataset_name = dataset_name

    def load_data(self):
        with open(self.data_file_path + self.dataset_name + '.pkl', 'rb') as f:
            self.data = pickle.load(f)

    def show_data(self):
        for key in self.data.keys():
            print(key, len(self.data[key]))
            print(self.data[key])
            print('-----------------')
    
if __name__ == '__main__':
    data_loader = DataLoader('F:/CODES/Python/UCAS-Machine-Learning/DataSet/', 'IDs')
    data_loader.load_data()
    data_loader.show_data()