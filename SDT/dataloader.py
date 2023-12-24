import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd


class IEMOCAPDataset(Dataset):
    def __init__(self, train=True):
        self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.trainVid,\
        self.testVid = pickle.load(open('data/iemocap_multimodal_features.pkl', 'rb'))
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

        if(train):
            self.train_ = 1
        else:
            self.train_ = 0

    def __getitem__(self, index):
        vid = self.keys[index]
        if self.train_:
            return torch.FloatTensor(self.videoText[vid]),\
                   torch.FloatTensor(self.videoVisual[vid]),\
                   torch.FloatTensor(self.videoAudio[vid]),\
                   torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                      self.videoSpeakers[vid]]),\
                   torch.FloatTensor([1]*len(self.videoText[vid])),\
                   torch.LongTensor(self.videoLabels[vid]),\
                   vid
        else:
            return torch.FloatTensor(self.videoText[vid]), \
                   torch.FloatTensor(self.videoVisual[vid]), \
                   torch.FloatTensor(self.videoAudio[vid]), \
                   torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in \
                                      self.videoSpeakers[vid]]), \
                   torch.FloatTensor([1] * len(self.videoText[vid])), \
                   torch.zeros(3,4), \
                   vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]
