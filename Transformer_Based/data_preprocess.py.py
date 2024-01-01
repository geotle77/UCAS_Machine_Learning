import pickle
import pandas as pd
import csv
import os
print(os.getcwd())

DATA_FILE_PATH = "SDT/data/"

with open(DATA_FILE_PATH + 'text_features.pkl', 'rb') as f:
    features = pickle.load(f)
# 提取以"dialog+数字ID"为键的二维数组
#videoText = [v for k, v in features.items() if k.startswith('dialog_') and k[7:].isdigit()]
videoText = features

with open(DATA_FILE_PATH + 'train_label.pkl', 'rb') as f:
    features = pickle.load(f)
    print(features)
# 提取以"dialog+数字ID"为键的二维数组
#videoLabels = [v for k, v in features.items() if k.startswith('dialog_') and k[7:].isdigit()]
videoLabels = features

with open(DATA_FILE_PATH + 'Speakers.pkl', 'rb') as f:
    features = pickle.load(f)
# 提取以"dialog+数字ID"为键的二维数组
#videoSpeakers = [v for k, v in features.items() if k.startswith('dialog_') and k[7:].isdigit()]
videoSpeakers = features

'''
with open(DATA_FILE_PATH + 'IDs.pkl', 'rb') as f:
    features = pickle.load(f)
# 提取以"dialog+数字ID"为键的二维数组
videoIDs = [v for k, v in features.items() if k.startswith('dialog_') and k[7:].isdigit()]
'''

with open(DATA_FILE_PATH + 'audio_features.pkl', 'rb') as f:
    features = pickle.load(f)
# 提取以"dialog+数字ID"为键的二维数组
#videoAudio = [v for k, v in features.items() if k.startswith('dialog_') and k[7:].isdigit()]
videoAudio = features

with open(DATA_FILE_PATH + 'visual_features.pkl', 'rb') as f:
    features = pickle.load(f)
# 提取以"dialog+数字ID"为键的二维数组
#videoVisual = [v for k, v in features.items() if k.startswith('dialog_') and k[7:].isdigit()]
videoVisual = features

with open('SDT/data/train_ids.csv', 'r') as file:
    # 创建CSV读取器
    csv_reader = csv.reader(file)

    # 从每一行中获取第一列数据，并将其存储在一个列表中
    #trainVid = [row[0][7:] for row in csv_reader]
    trainVid = [row[0] for row in csv_reader]
    trainVid.pop(0)
    #trainVid = [int(i)-1 for i in trainVid]

with open('SDT/data/test_ids.csv', 'r') as file:
    # 创建CSV读取器
    csv_reader = csv.reader(file)

    # 从每一行中获取第一列数据，并将其存储在一个列表
    #testVid = [row[0][7:] for row in csv_reader]
    testVid = [row[0] for row in csv_reader]
    print(testVid)
    testVid.pop(0)
    temp = 0
    for i in testVid:
        temp += len(videoText[i])
        print(temp)
    #testVid = [int(i)-1 for i in testVid]

save_data = (videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, trainVid, testVid)

with open('./data/iemocap_multimodal_features.pkl', 'wb') as file:
    pickle.dump(save_data, file)

with open('./data/iemocap_multimodal_features.pkl', 'rb') as file:
    features = pickle.load(file)
    print("1")