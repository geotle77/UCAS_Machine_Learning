from transformer import *
from dataset import *
import pickle
import csv
import numpy as np

DATA_FILE_PATH = "./DataSet/"
input_data = []
output_data = []

test_input = []
test_labels = []

# 打开CSV文件
with open(DATA_FILE_PATH + 'train_ids.csv', 'r') as file:
    # 创建CSV读取器
    csv_reader = csv.reader(file)

    # 从每一行中获取第一列数据，并将其存储在一个列表中
    ids = [row[0] for row in csv_reader]

# 打印或使用得到的列表
ids.pop(0)
train_ids = ids[:100]
test_ids  = ids[100:]

with open(DATA_FILE_PATH + 'text_features.pkl', 'rb') as f:
    text_features = pickle.load(f)
# 提取以"dialog+数字ID"为键的二维数组
dialog_ID_features = {k : v for k, v in text_features.items() if k.startswith('dialog_') and k[7:].isdigit()}

with open(DATA_FILE_PATH + 'train_label.pkl', 'rb') as f:
    train_label_O = pickle.load(f)
# 提取以"dialog+数字ID"为键的二维数组
labels = {k : v for k, v in train_label_O.items() if k.startswith('dialog_') and k[7:].isdigit()}

for id in train_ids:
    for i in dialog_ID_features[id] :
        input_data.append(i)
    for j in labels[id]:
        output_data.append(j)

for id in test_ids:
    for i in dialog_ID_features[id]:
        test_input.append(i)
    for j in labels[id]:
        test_labels.append(j)

'''
# 找到最长的子序列的长度
max_length = 110
print(max_length,len(input_data), len(test_input))
# 填充每个子序列，使它们的长度都等于最长的子序列长度
for i in range(len(input_data)):
    for j in range(max_length - len(input_data[i])):
        input_data[i].append(zeros_array)
        output_data[i].append(0)

# 填充每个子序列，使它们的长度都等于最长的子序列长度
for i in range(len(test_input)):
    for j in range(max_length - len(test_input[i])):
        test_input[i].append(zeros_array)
        test_labels[i].append(0)

print('1')
'''

# 定义超参数
d_model = 8
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.01
learning_rate = 0.00001
batch_size = 64
epochs = 10
shuffle = True  # 是否打乱数据
num_workers = 4  # 设置用于加载数据的线程数

dataset = TransformerDataset(input_data, output_data)
test_dataset = TransformerDataset(test_input, test_labels)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# 创建数据加载器
#DataLoader中的collate_fn应该处理样本的填充
dataloader = DataLoader(dataset, batch_size=batch_size)

# 创建Transformer模型
model = Transformer(
    d_model=16,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    batch_first=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device) # 将模型加载到GPU中（如果已经正确安装）

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 训练循环
#model.eval()
for epoch in range(epochs):
    total_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
        # 将输入数据传递给模型
        batch_src, batch_tgt = [x.to(device) for x in batch]
        tgt = batch_tgt.view(batch_tgt.shape[0],1)
        output = model(batch_src, tgt=tgt)

        output = output.view(output.shape[0], output.shape[2])

        '''
        standard = torch.zeros((output.shape[0], 1, 6))
        for i in range(len(batch_tgt)):
                class_num = tgt[i]
                if class_num > 0:
                    standard[i][0][class_num - 1] = 1
        '''

        # 计算损失
        loss = criterion(output, batch_tgt)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 打印每个epoch的平均损失
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')


#测试过程
acc = 0
error = 0
#model.eval()
for batch in tqdm(test_dataloader, desc=f"Testing"):
    inputs, targets = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs, tgt = targets.view(targets.shape[0], 1))
        standard = torch.zeros((output.shape[0], output.shape[1], 6))

        max_idx = [0] * output.shape[0]
        print(output.shape)
        for i in range(len(output)):
            for j in range(len(output[i])):
                max = -10000
                for k in range(len(output[i][j])):
                    if (output[i][j][k] > max):
                        max_idx[i] = k + 1
                        max = output[i][j][k]

        print(max_idx)

        for i in range(len(targets)):
            if (max_idx[i] == targets[i]):
                acc += 1
            else:
                error += 1


# 输出在测试集上的准确率
print(acc)
print(f"Acc: {acc / (acc + error):.2f}")


