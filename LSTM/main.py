from Loder import DataLoader
from lstm import LSTMModel
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        if isinstance(m, nn.ReLU):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif isinstance(m, (nn.Sigmoid, nn.Tanh)):
            torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif type(m) == nn.LSTM or type(m) == nn.GRU:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

class MultiModalModel(object):
    def __init__(self, path):
        super().__init__()
        self.model = LSTMModel(1024, 342, 1582, 1024, 2, 6)
        self.model.apply(init_weights)
        data=DataLoader(path)
        data.load_train_data()
        data.load_train_label()
        data.load_test_data()
        self.text_train_data=data.text_train_data
        self.text_test_data=data.text_test_data
        self.visual_train_data = data.visual_train_data
        self.visual_test_data = data.visual_test_data
        self.audio_train_data = data.audio_train_data
        self.audio_test_data = data.audio_test_data
        self.train_label = data.train_label
        self.pad_value = -1

    def dataprocess(self):
        dialog_text_train = [torch.tensor(self.text_train_data[key]) for key in self.text_train_data.keys()]
        dialog_visual_train = [torch.tensor(self.visual_train_data[key]) for key in self.visual_train_data.keys()]
        dialog_audio_train = [torch.tensor(self.audio_train_data[key]) for key in self.audio_train_data.keys()]
        text_padded_dialogues = torch.nn.utils.rnn.pad_sequence(dialog_text_train, batch_first=True, padding_value=self.pad_value)
        visual_padded_dialogues = torch.nn.utils.rnn.pad_sequence(dialog_visual_train, batch_first=True, padding_value=self.pad_value)
        audio_padded_dialogues = torch.nn.utils.rnn.pad_sequence(dialog_audio_train, batch_first=True, padding_value=self.pad_value)
        print("padded_dialogue's shape,for training:",   text_padded_dialogues.shape, visual_padded_dialogues.shape, audio_padded_dialogues.shape)
        dialog_label_train = [torch.tensor(self.train_label[key]) for key in self.train_label.keys()]
        padded_labels = torch.nn.utils.rnn.pad_sequence(dialog_label_train, batch_first=True, padding_value=self.pad_value)
        print("padded_label's shape,for training:",  padded_labels.shape)
        return text_padded_dialogues, visual_padded_dialogues, audio_padded_dialogues, padded_labels

    def train(self):
        text_padded_dialogues, visual_padded_dialogues, audio_padded_dialogues, padded_labels = self.dataprocess()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        text_padded_dialogues = text_padded_dialogues.to(device)
        visual_padded_dialogues = visual_padded_dialogues.to(device)
        audio_padded_dialogues = audio_padded_dialogues.to(device)

        padded_labels = padded_labels.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.01)  # 学习率调度器
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_value) 
        best_loss = float('inf')
        batch_size = 30  # 设置批次大小
        num_batches = len(text_padded_dialogues) // batch_size  # 计算批次数量

        for epoch in range(75):
            correct = 0  # 正确预测的标签数
            total = 0  # 总标签数
            for i in range(num_batches):
                text_batch_dialogues = text_padded_dialogues[i*batch_size : (i+1)*batch_size]
                visual_batch_dialogues = visual_padded_dialogues[i * batch_size: (i + 1) * batch_size]
                audio_batch_dialogues = audio_padded_dialogues[i * batch_size: (i + 1) * batch_size]
                batch_labels = padded_labels[i*batch_size : (i+1)*batch_size]

                optimizer.zero_grad()
                outputs = self.model(text_batch_dialogues, visual_batch_dialogues, audio_batch_dialogues)
                _, predicted = torch.max(outputs.data, -1)  # 获取每个时间步的最大概率对应的标签
                total += batch_labels.numel()  # 更新总标签数
                correct += (predicted == batch_labels).sum().item()  # 更新正确预测的标签数
                loss = criterion(outputs.view(-1, 6), batch_labels.view(-1))
                loss.backward()
                optimizer.step()
                scheduler.step()

                #for param in self.model.parameters():
                    #print(param.grad)

            accuracy = correct / total  # 计算整个epoch的正确率
            print('epoch: {}, loss: {}, accuracy: {}'.format(epoch, loss.item(), accuracy))
                
    
    def evalaute(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_dialog_text = [torch.tensor(self.text_test_data[key]) for key in self.text_test_data.keys()]
        visual_dialog_text = [torch.tensor(self.visual_test_data[key]) for key in self.visual_test_data.keys()]
        audio_dialog_text = [torch.tensor(self.audio_test_data[key]) for key in self.audio_test_data.keys()]
        text_padded_test_dialogues = torch.nn.utils.rnn.pad_sequence(test_dialog_text, batch_first=True, padding_value=self.pad_value)
        visual_padded_test_dialogues = torch.nn.utils.rnn.pad_sequence(visual_dialog_text, batch_first=True, padding_value=self.pad_value)
        audio_padded_test_dialogues = torch.nn.utils.rnn.pad_sequence(audio_dialog_text, batch_first=True, padding_value=self.pad_value)

        text_padded_test_dialogues = text_padded_test_dialogues.to(device)
        visual_padded_test_dialogues = visual_padded_test_dialogues.to(device)
        audio_padded_test_dialogues = audio_padded_test_dialogues.to(device)
        print("padded_test_dialogue's shape,for testing:",   text_padded_test_dialogues.shape, visual_padded_test_dialogues.shape, audio_padded_test_dialogues.shape)
        self.model.eval()  # switch the model to evaluation mode
        with torch.no_grad():
            outputs = self.model(text_padded_test_dialogues, visual_padded_test_dialogues, audio_padded_test_dialogues)
            print(outputs.shape)
        mask = torch.full_like(text_padded_test_dialogues, True, dtype=torch.bool)
        for i in range(text_padded_test_dialogues.shape[0]):
            for j in range(text_padded_test_dialogues.shape[1]):
                target = torch.full_like(text_padded_test_dialogues[i][j], -1)
                if torch.equal(text_padded_test_dialogues[i][j], target):
                    mask[i,j] = False
        mask_reduced = mask.all(dim=-1)
        predictions = torch.argmax(outputs, dim=-1)
        masked_predictions = predictions[mask_reduced]
        Y_predict = masked_predictions.cpu().numpy()
        Y_true = np.loadtxt('F:/CODES/Python/UCAS-Machine-Learning/test_pred.csv', dtype = int)
        weighted_f1 = f1_score(Y_true, Y_predict, average='weighted')
        accuary = np.mean(Y_true == Y_predict)
        print('accuary:',accuary)
        print('weighted_f1:',weighted_f1)

        
            

if __name__ == '__main__':
    model=MultiModalModel('F:/CODES/Python/UCAS-Machine-Learning/UCAS-Machine-Learning/data')
    model.train()
    model.evalaute()

