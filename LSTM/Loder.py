import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch

class DataLoader(object):
    def __init__(self, DataPath):
        self.train_ids = set(pd.read_csv(DataPath+'/train_ids.csv')['ids'])
        self.test_ids = set(pd.read_csv(DataPath+'/test_ids.csv')['ids'])
        self.audio_features = pd.read_pickle(DataPath+'/audio_features.pkl')
        self.text_features = pd.read_pickle(DataPath+'/text_features.pkl')
        self.visual_features = pd.read_pickle(DataPath+'/visual_features.pkl')
        self.all_label = pd.read_pickle(DataPath+'/train_label.pkl')
        self.train_label = {}
        self.test_data = {}
        self.train_data = {}
        self.text_train_data = {}
        self.text_test_data = {}
        self.visual_train_data = {}
        self.visual_test_data = {}
        self.audio_train_data = {}
        self.audio_test_data = {}

    def load_train_data(self):
        for key,value in self.audio_features.items():
            if key in self.train_ids:
                self.audio_train_data[key] = value
        for key,value in self.text_features.items():
            if key in self.train_ids:
                self.text_train_data[key] = value
        for key,value in self.visual_features.items():
            if key in self.train_ids:
                self.visual_train_data[key] = value
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
        for key,value in self.audio_features.items():
            if key in self.test_ids:
                self.audio_test_data[key] = value
        for key,value in self.text_features.items():
            if key in self.test_ids:
                self.text_test_data[key] = value
        for key,value in self.visual_features.items():
            if key in self.test_ids:
                self.visual_test_data[key] = value
        # return self.test_data
                
    def show_data_info(self):
        print('self.text_train_data',self.text_train_data)
        print('self.text_test_data',self.text_test_data)
        print('self.visual_train_data',self.visual_train_data)
        print('self.visual_test_data',self.visual_test_data)
        print('self.audio_train_data',self.audio_train_data)
        print('self.audio_test_data',self.audio_test_data)



