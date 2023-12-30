import torch
from torch import nn

class Multimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Multimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)
        utters = torch.cat([a_new, b_new, c_new], dim=-2)
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2), self.fc(c).unsqueeze(-2)], dim=-2)
        utters_softmax = self.softmax(utters_fc)
        utters_three_model = utters_softmax * utters
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
        return final_rep

class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, a):
        z = torch.sigmoid(self.fc(a))
        final_rep = z * a
        return final_rep

class LSTMModel(nn.Module):
    def __init__(self, input_size1, input_size2, input_size3, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.text_cov = nn.Conv1d(input_size1, hidden_size, kernel_size=1, padding=0, bias=False)
        self.visual_cov = nn.Conv1d(input_size2, hidden_size, kernel_size=1, padding=0, bias=False)
        self.audio_cov = nn.Conv1d(input_size3, hidden_size, kernel_size=1, padding=0, bias=False)
        self.text_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.visual_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.audio_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.text_fc1 = nn.Linear(hidden_size*2, hidden_size)  # 注意这里的输入维度变为了2倍的hidden_size
        self.visual_fc1 = nn.Linear(hidden_size*2, hidden_size)  # 注意这里的输入维度变为了2倍的hidden_size
        self.audio_fc1 = nn.Linear(hidden_size*2, hidden_size)  # 注意这里的输入维度变为了2倍的hidden_size
        self.text_dropout = nn.Dropout(0.1)
        self.visual_dropout = nn.Dropout(0.1)
        self.audio_dropout = nn.Dropout(0.1)
        self.fc3 = nn.Linear(hidden_size, num_classes)

        self.last_gate = Multimodal_GatedFusion(hidden_size)
        self.text_gate = Unimodal_GatedFusion(hidden_size)
        self.visual_gate = Unimodal_GatedFusion(hidden_size)
        self.audio_gate = Unimodal_GatedFusion(hidden_size)

    def forward(self, text, visual, audio):
        visual = visual.float()
        text_out = self.text_cov(text.permute(1, 2, 0)).permute(2, 0, 1)
        visual_out = self.visual_cov(visual.permute(1, 2, 0)).permute(2, 0, 1)
        audio_out = self.audio_cov(audio.permute(1, 2, 0)).permute(2, 0, 1)

        text_out, _ = self.text_lstm(text_out)
        text_out = self.text_fc1(text_out)
        text_out = self.text_gate(text_out)
        # text_out = self.text_dropout(text_out)

        visual_out, _ = self.visual_lstm(visual_out)
        visual_out = self.visual_fc1(visual_out)
        visual_out = self.visual_gate(visual_out)
        # visual_out = self.visual_dropout(visual_out)

        audio_out, _ = self.audio_lstm(audio_out)
        audio_out = self.audio_fc1(audio_out)
        audio_out = self.audio_gate(audio_out)
        # audio_out = self.audio_dropout(audio_out)

        muti_output = self.last_gate(text_out, visual_out, audio_out)

        out = self.fc3(muti_output)

        return out