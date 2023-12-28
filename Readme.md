# UCAS 机器学习大作业
## 数据集
| 类别   | 文件名               | 描述                                       |
|--------|----------------------|--------------------------------------------|
| 训练集 | train_ids.csv        | 训练样本对话的ID                           |
| 测试集 | test_ids.csv         | 测试样本对话的ID                           |
| 数据文件 | text_features.pkl   | 所有对话的文本特征                         |
| 数据文件 | audio_features.pkl  | 所有对话的音频特征                         |
| 数据文件 | visual_features.pkl | 所有对话的图像特征                         |
| 数据文件 | IDs.pkl             | 每个对话的ID                               |
| 数据文件 | train_label.pkl     | 训练样本对话中对话句子的真实标签           |
| 数据文件 | Speakers.pkl        | 所有对话中男女人物对话的顺序，其中M代表男性，F代表女性 |

注意：由于比赛给定的数据集在后面发生了变化，所以Transformer_Based模型在训练时，使用的是新的数据集，而LSTM和决策树使用的是旧的数据集。

## 代码
### 1. Transformer_Based
#### 目录架构
基于Transformer编码器的模型，使用文本、音频和图像特征进行训练。
`data`文件夹存储的是数据集
`Csv`文件加存储的是结果输出的csv文件
其余为代码文件
`data_processed.py`：用于对数据集进行预处理，作用为把多个文件合并成一个pkl文件，方便读取
`model.py`：模型文件
`train.py`：训练模型
`mytools.py`：工具文件，包含一些写CSV文件和获取正确率和F1_score的函数
`show_result.py`: 将训练过程中的正确率F1_score可视化

#### 运行方法
首先运行`data_processed.py`，将数据集合并成iemocap_multimodal_features.pkl文件，然后运行train.py，训练模型。
训练结束后，如果需要显示正确率F1_score的可视化图，运行show_model.py。

#### 注意事项
由于本人使用Pycharm进行代码编写，使用的相对路径。但Pycharm的相对路径基点为项目的路径，所以如果代码出现报错，可能是文件路径的问题，可以尝试修改文件路径。
如果不想得到输出的csv文件，可以注释掉相关函数即可