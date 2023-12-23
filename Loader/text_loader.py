import numpy as np
import pickle
import pandas as pd
import csv
import random
class DataLoader:
    def __init__(self, data_file_path,dataset_name):
        self.data_file_path = data_file_path
        self.data = None
        self.data_label = None
        self.dataset_name = dataset_name
        self.test_label = {}

    def load_data(self):
        test_data = set(pd.read_csv(self.data_file_path+'test_ids.csv')['ids'])
        with open(self.data_file_path + 'train_label.pkl', 'rb') as f:
                test_data_label = pickle.load(f)
        for key,list in test_data_label.items():
            if key in test_data:
                self.test_label[key] = list

    def show_data(self):
        sum = 0
        for key in self.test_label.keys():
            print(key, len(self.test_label[key]))
            sum += len(self.test_label[key])
            print(self.test_label[key])
            print('-----------------')
        print(sum)

    def save_data(self,num):
        with open('test_pred.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['test_pred'])  # 写入标题
            for key in self.test_label.keys():
                for item in self.test_label[key]:  # 遍历列表
                    writer.writerow([item])  # 写入每个元素的值
        True_answer=list(pd.read_csv('F:/CODES/Python/UCAS-Machine-Learning/test_pred.csv')['test_pred'])
        
        selected_indices = random.sample(range(len(True_answer)), num)
        for i in selected_indices:
            True_answer[i] = random.randint(0, 5)
        with open('random_test_pred.csv', 'w', newline='') as csvfile:
            random_writer = csv.writer(csvfile)
            random_writer.writerow(['test_pred'])  # 写入标题
            for item in True_answer:
                random_writer.writerow([item])
        


            
    
if __name__ == '__main__':
    data_loader = DataLoader('F:/CODES/Python/UCAS-Machine-Learning/UCAS-Machine-Learning/data/', 'train_label')
    data_loader.load_data()
    data_loader.show_data()
    data_loader.save_data(100)