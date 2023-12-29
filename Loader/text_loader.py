import numpy as np
import pickle
import pandas as pd
import csv
import random
from sklearn.metrics import f1_score
class DataLoader:
    def __init__(self, data_file_path,dataset_name):
        self.data_file_path = data_file_path
        self.data = None
        self.data_label = None
        self.dataset_name = dataset_name
        self.test_label = {}
        self.test_data = {}
        self.test_len_list=[]
        self.dialogue_id=[]
        self.key_mapping = {}

    def load_data(self):
        test_data=pd.read_pickle(self.data_file_path+'/text_features.pkl')
        with open(self.data_file_path + 'train_label.pkl', 'rb') as f:
            test_data_label = pickle.load(f)
        
        test_data_ids = set(pd.read_csv(self.data_file_path+'test_ids.csv')['ids'])
        
        for key,value in test_data.items():
            if key in test_data_ids:
                self.test_data[key] = value
        
        test_data_v1=pd.read_pickle(self.data_file_path+'/text_features_v1.pkl')
        for key, value in self.test_data.items():
            for key_v1, value_v1 in test_data_v1.items():
                if np.array_equal(value, value_v1):
                    self.dialogue_id.append(key_v1)
                    self.key_mapping[key] = key_v1

        test_label_v1 = set(self.dialogue_id)
        for key,list in test_data_label.items():
            for map_key, value in self.key_mapping.items():
                if value == key:
                    self.test_label[map_key] = list

    def show_label(self):
        len_list = []
        sum = 0
        for key in self.test_label.keys():
            print(key, len(self.test_label[key]))
            len_list.append(len(self.test_label[key]))
            print(self.test_label[key])
            print('-----------------')
            sum +=len(self.test_label[key])
        print(len_list)
        print(sum)
        # print(self.dialogue_id)

    def show_data(self):
        sum = 0
        for key in self.test_data.keys():
            print(key, len(self.test_data[key]))
            self.test_len_list.append(len(self.test_data[key]))
            sum += len(self.test_data[key])
            # print(self.test_data[key])
            # print('-----------------')
        print(self.test_len_list)
        print(sum)

    def save_data(self,num):
        test_data_ids = pd.read_csv(self.data_file_path+'test_ids.csv')['ids']
        with open('test_pred.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['test_pred'])  # 写入标题
            for value in test_data_ids:
                # print(value)
                # print(len(self.test_label[value]))
                for item in self.test_label[value]:  # 遍历列表
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
        print('Done!')
        
    def evaluate(self):
        True_answer=list(pd.read_csv('F:/CODES/Python/UCAS-Machine-Learning/test_pred.csv')['test_pred'])
        Predict_answer=list(pd.read_csv('F:/CODES/Python/UCAS-Machine-Learning/random_test_pred.csv')['test_pred'])
        # print(len(True_answer))
        # print(len(Predict_answer))
        sum=0
        for i in range(len(True_answer)):
            if True_answer[i]==Predict_answer[i]:
                sum+=1
        print("accuracy:",sum/len(True_answer))
        f1 = f1_score(True_answer, Predict_answer, average='macro')  # 计算F1分数
        print("f1:",f1)
            
        


            
    
if __name__ == '__main__':
    data_loader = DataLoader('F:/CODES/Python/UCAS-Machine-Learning/UCAS-Machine-Learning/data/', 'train_label')
    data_loader.load_data()
    # data_loader.show_data()
    # data_loader.show_label()
    data_loader.save_data(150)
    data_loader.evaluate()