import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pickle
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

text_data=pd.read_pickle('F:/CODES/Python/UCAS-Machine-Learning/UCAS-Machine-Learning/data/text_features_v1.pkl')
label=pd.read_pickle('F:/CODES/Python/UCAS-Machine-Learning/UCAS-Machine-Learning/data/train_label.pkl')

map_function = {key: list(zip(text_data[key], label[key])) for key in text_data if key in label}
data = [item[0] for sublist in map_function.values() for item in sublist]
labels = [item[1] for sublist in map_function.values() for item in sublist]

# 创建一个颜色映射，每个种类对应一个颜色
color_map = {label: idx for idx, label in enumerate(set(labels))}

# 使用PCA降维到3D
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for idx, point in enumerate(data_pca):
    ax.scatter(point[0], point[1], point[2], color=plt.cm.rainbow(color_map[labels[idx]] / len(color_map)))

plt.show()