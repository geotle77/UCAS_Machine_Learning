from Loder import DataLoader
from lstm import LSTMModel
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

class MultiModalModel(object):
    def __init__(self, path):
        super().__init__()
        self.model = LSTMModel(1024, 512, 3, 6)
        data=DataLoader(path)
        data.load_train_data()
        data.load_train_label()
        data.load_test_data()
        self.text_train_data=data.text_train_data
        self.text_test_data=data.text_test_data
        self.train_label=data.train_label
        self.pad_value = -1

    def dataprocess(self):
        dialog_text_train = [torch.tensor(self.text_train_data[key]) for key in self.text_train_data.keys()]
        padded_dialogues = torch.nn.utils.rnn.pad_sequence(dialog_text_train, batch_first=True, padding_value=self.pad_value)
        print("padded_dialogue's shape,for training:",   padded_dialogues.shape)
        dialog_label_train = [torch.tensor(self.train_label[key]) for key in self.train_label.keys()]
        padded_labels = torch.nn.utils.rnn.pad_sequence(dialog_label_train, batch_first=True, padding_value=self.pad_value)
        print("padded_label's shape,for training:",  padded_labels.shape)
        return padded_dialogues,padded_labels

    def train(self):
        padded_dialogues,padded_labels=self.dataprocess()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        padded_dialogues = padded_dialogues.to(device)
        padded_labels = padded_labels.to(device)

        optimizer = optim.Adam(self.model.parameters())
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.01)  # 学习率调度器
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_value) 
        best_loss = float('inf')
        batch_size = 20  # 设置批次大小
        num_batches = len(padded_dialogues) // batch_size  # 计算批次数量

        for epoch in range(40):
            correct = 0  # 正确预测的标签数
            total = 0  # 总标签数
            for i in range(num_batches):
                batch_dialogues = padded_dialogues[i*batch_size : (i+1)*batch_size]
                batch_labels = padded_labels[i*batch_size : (i+1)*batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_dialogues)
                _, predicted = torch.max(outputs.data, -1)  # 获取每个时间步的最大概率对应的标签
                total += batch_labels.numel()  # 更新总标签数
                correct += (predicted == batch_labels).sum().item()  # 更新正确预测的标签数
                loss = criterion(outputs.view(-1, 6), batch_labels.view(-1))
                loss.backward()
                optimizer.step()
                # scheduler.step()

            accuracy = correct / total  # 计算整个epoch的正确率
            print('epoch: {}, loss: {}, accuracy: {}'.format(epoch, loss.item(), accuracy))
                
    
    def evalaute(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_dialog_text = [torch.tensor(self.text_test_data[key]) for key in self.text_test_data.keys()]
        padded_test_dialogues = torch.nn.utils.rnn.pad_sequence(test_dialog_text, batch_first=True, padding_value=self.pad_value)
        padded_test_dialogues = padded_test_dialogues.to(device)
        print("padded_test_dialogue's shape,for testing:",   padded_test_dialogues.shape)
        self.model.eval()  # switch the model to evaluation mode
        with torch.no_grad():
            outputs = self.model(padded_test_dialogues)
            print(outputs.shape)
        mask = torch.full_like(padded_test_dialogues, True, dtype=torch.bool)
        for i in range(padded_test_dialogues.shape[0]):
            for j in range(padded_test_dialogues.shape[1]):
                target = torch.full_like(padded_test_dialogues[i][j], -1)
                if torch.equal(padded_test_dialogues[i][j], target):
                    mask[i,j]=False
        mask_reduced = mask.all(dim=-1)
        predictions = torch.argmax(outputs, dim=-1)
        masked_predictions = predictions[mask_reduced]
        Y_predict = masked_predictions.cpu().numpy()
        Y_true = np.loadtxt('./reference_answer.csv', dtype = int)
        weighted_f1 = f1_score(Y_true, Y_predict, average='weighted')
        accuary = np.mean(Y_true == Y_predict)
        print('accuary:',accuary)
        print('weighted_f1:',weighted_f1)

        
            

if __name__ == '__main__':
    model=MultiModalModel('./data')
    model.train()
    model.evalaute()

