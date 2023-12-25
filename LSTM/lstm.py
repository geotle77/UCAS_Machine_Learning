import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 新增的dropout层
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 新增的全连接层
        self.fc3 = nn.Linear(hidden_size, num_classes)  # 原来的输出层现在变成了第三个全连接层

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)  # 在激活函数后使用dropout
        out = self.fc2(out)
        out = self.relu(out)  # 在第二个全连接层后也使用激活函数
        out = self.fc3(out)
        return out