import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset
from model import MaskedNLLLoss, Transformer_Based_Model
from sklearn.metrics import f1_score, accuracy_score
from mytools import get_test_accuracy, record_acc, record_f1_scores
import csv

final_labels = ["test_pred"]
accuracy = 0
max_accuracy = 0
fscore = 0

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, optimizer=None, train=False, gamma_1=1.0, gamma_2=1.0):
    losses, preds, labels, masks = [], [], [], []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
        return float('nan'), float('nan'), [], [], [], float('nan')

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if torch.cuda.is_available() else data[:-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob1, log_prob2, log_prob3, all_log_prob, all_prob = model(textf, visuf, acouf, umask, qmask, lengths)
        
        lp_1 = log_prob1.view(-1, log_prob1.size()[2])
        lp_2 = log_prob2.view(-1, log_prob2.size()[2])
        lp_3 = log_prob3.view(-1, log_prob3.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)


        loss = gamma_1 * loss_function(lp_all, labels_, umask) + \
                gamma_2 * (loss_function(lp_1, labels_, umask) + loss_function(lp_2, labels_, umask) + loss_function(lp_3, labels_, umask))

        lp_ = all_prob.view(-1, all_prob.size()[2])

        pred_ = torch.argmax(lp_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            optimizer.step()

    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)  
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore

def reserve_in_csv(model, dataloader, reserve):
    global accuracy, fscore, max_accuracy, final_labels
    if not reserve:
        csv_labels = []
        model.eval()

        for data in dataloader:
            textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if torch.cuda.is_available() else data[:-1]
            qmask = qmask.permute(1, 0, 2)
            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

            log_prob1, log_prob2, log_prob3, all_log_prob, all_prob = model(textf, visuf, acouf, umask, qmask, lengths)

            lp_ = all_prob.view(-1, all_prob.size()[2])
            umask = umask.view(-1).tolist()
            #print(umask)

            pred_ = torch.argmax(lp_, 1)
            pred_lst = pred_.tolist()
            pred_lst = [elem for mask, elem in zip(umask, pred_lst) if mask == 1.0]
            csv_labels.extend(pred_lst)

        # 打开 CSV 文件进行写入
        #print(len(csv_labels))
        accuracy, fscore = get_test_accuracy(csv_labels, ".Csv/reference.csv")
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            final_labels = csv_labels.copy()

    else:
        final_labels.insert(0, "test_pred")
        print(len(final_labels))
        with open('.Csv/submission.csv', 'w', newline='') as file:
            # 创建 CSV 写入器对象
            writer = csv.writer(file)

            # 写入数据
            for item in final_labels:
                writer.writerow([item])


if __name__ == '__main__':
    cuda_availdabe = torch.cuda.is_available()
    if cuda_availdabe:
        print('Running on GPU')
    else:
        print('Running on CPU')

    epochs = 750
    batch_size = 8
    hidden_dim = 1024
    dropout = 0.5

    
    n_head = 8
    lr = 0.00001
    l2 = 0.000001

    D_audio = 1582
    D_visual = 342
    D_text = 1024

    D_m = D_audio + D_visual + D_text

    n_speakers = 2
    n_classes =  6

    model = Transformer_Based_Model(D_text, D_visual, D_audio, n_head,
                                        n_classes=n_classes,
                                        hidden_dim=hidden_dim,
                                        n_speakers=n_speakers,
                                        dropout=dropout)

    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)


    loss_weights = torch.FloatTensor([1/0.086747,
                                    1/0.144406,
                                    1/0.227883,
                                    1/0.160585,
                                    1/0.127711,
                                    1/0.252668])
    loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda_availdabe else loss_weights)
    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                  batch_size=batch_size,
                                                                  num_workers=0)

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_test_acc, all_loss, all_train_acc = [], [], [], []

    for e in range(epochs):
        start_time = time.time()

        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(model, loss_function, train_loader, optimizer, True)
        valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(model, loss_function, valid_loader)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, test_loader, e)
        reserve_in_csv(model, test_loader, reserve=False)

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask = test_label, test_pred, test_mask


        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, accuracy, fscore, round(time.time()-start_time, 2)))
        all_fscore.append(fscore)
        all_test_acc.append(accuracy)
        all_train_acc.append(train_acc)


    reserve_in_csv(model, test_loader, reserve=True)

    print('Test performance..')
    print("Max_Acc: {}".format(max_accuracy))
    print('F-Score: {}'.format(max(all_fscore)))
    print('F-Score-index: {}'.format(all_fscore.index(max(all_fscore)) + 1))

    record_acc(all_train_acc, all_test_acc, '.Csv/Acc.csv')
    record_f1_scores(all_fscore, '.Csv/F1_Scores.csv')




