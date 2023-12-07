import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class DataLoader(object):
    def __init__(self, DataPath):
        self.train_ids = set(pd.read_csv(DataPath+'/train_ids.csv')['ids'])
        self.test_ids = set(pd.read_csv(DataPath+'/test_ids.csv')['ids'])
        self.text_features = pd.read_pickle(DataPath+'/text_features.pkl')
        self.all_label = pd.read_pickle(DataPath+'/train_label.pkl')
        self.dimension = 1024
        self.train_label = {}
        self.test_data = {}
        self.train_data = {}
        self.X_train = []
        self.Y_train = []
        self.X_test = []

    def load_train_data(self):
        for key,value in self.text_features.items():
            if key in self.train_ids:
                self.train_data[key] = value
        # for key,list in self.train_data.items():
        #     for array in list:
        #         print(len(array))
        #         self.dimension = max(self.dimension,len(array))
        # return self.train_data

    def load_train_label(self):
        for key,value in self.all_label.items():
            if key in self.train_ids:
                self.train_label[key] = value
        # return self.train_label

    def load_test_data(self):
        for key,value in self.text_features.items():
            if key in self.test_ids:
                self.test_data[key] = value
        # return self.test_data
    
    def DataProcess(self):
        for key,list in self.train_data.items():
            for array in list:
                self.X_train.append(array)
        for key,list in self.train_label.items():
            for array in list:
                self.Y_train.append(array)
        for key,list in self.test_data.items():
            for array in list:
                self.X_test.append(array)
        self.X_train = np.array(self.X_train)
        self.Y_train = np.array(self.Y_train)
        self.X_test = np.array(self.X_test)
    
    def show_data_info(self):
        print('X_train:',self.X_train.shape)
        print('Y_train:',self.Y_train.shape)
        print('X_test:',self.X_test.shape)



class DecisionTree(object):
    def __init__(self, train_data, train_label, test_data):
        self.model = DecisionTreeClassifier()
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.Y_predict = np.array([])
    
    def train(self):
        self.model.fit(self.train_data, self.train_label)
    
    def predict(self):
        self.Y_predict = self.model.predict(self.test_data)
        # print(Y_predict)

    def save(self):
        Y_predict = self.Y_predict.astype(int)
        Y_predict = Y_predict.reshape(-1,1)
        np.savetxt('dicision_output.csv', Y_predict, delimiter = ',', fmt = '%d')

    def cal_correct_rate(self):
        Y_true = np.loadtxt('F:/CODES/Python/UCAS-Machine-Learning/output.csv', dtype = int)
        correct = np.equal(Y_true, self.Y_predict)
        correct_rate = np.mean(correct)
        print(correct_rate)


if __name__ == '__main__':
    data_loader = DataLoader('F:/CODES/Python/UCAS-Machine-Learning/DataSet/')
    data_loader.load_train_data()
    data_loader.load_train_label()
    data_loader.load_test_data()
    data_loader.DataProcess() 
    data_loader.show_data_info()
    X_train = data_loader.X_train
    Y_train = data_loader.Y_train
    X_test = data_loader.X_test
    model = DecisionTree(X_train, Y_train, X_test)
    model.train()
    model.predict()
    model.save()
    model.cal_correct_rate()