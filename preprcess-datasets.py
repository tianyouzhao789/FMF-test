# @Time    : 2025/1/1 16:50
# @Author  : Tianyou Zhao
# @Email   : zhaotianyou@home.hpu.edu.cn
# @File    : preprcess-datasets.py
# @Software: PyCharm
import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
# 文件夹路径
folder_path = "./FMFBenchmarkV1"

# 获取文件夹中的所有 .mat 文件
mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

# 初始化空列表来存储数据
videos = []
labels = []
train_test_indices = []
currents = []

# 循环读取每个 .mat 文件的数据并合并
for mat_file in mat_files:
    file_path = os.path.join(folder_path, mat_file)
    with h5py.File(file_path, 'r') as reader:
        videos.append(np.array(reader['video'], dtype=np.uint8))
        labels.append(np.array(reader['label'], dtype=np.uint8))
        train_test_indices.append(np.array(reader['train_test_index'], dtype=np.uint8))
        currents.append(np.array(reader['current'], dtype=np.uint8))

# 合并数据
videos = np.concatenate(videos, axis=0)
labels = np.concatenate(labels, axis=0)
train_test_indices = np.concatenate(train_test_indices, axis=0)
currents = np.concatenate(currents, axis=0)
# 获取class标签
sum_result = np.sum(labels, axis=(1, 2))
# 判断是否大于 0.5
logical_result = sum_result > 0.5
# 转换为单精度浮点数
class_label = logical_result.astype(np.float32)

#将数据转换成torch tensor
videos = torch.tensor(videos)
labels = torch.tensor(labels)
train_test_indices = torch.tensor(train_test_indices).transpose(0, 1)
currents = torch.tensor(currents)
class_label = torch.tensor(class_label)
# print(train_test_indices.shape)
# train_test_indices,他是一个1*269921的二维向量，其中存放的是训练集和测试集的索引，我们可以根据这个索引来划分数据集，0 表示训练集，1 表示测试集
# 划分数据集
train_indices = train_test_indices[0] == 0
test_indices = train_test_indices[0] == 1
train_videos = videos[train_indices]
train_labels = labels[train_indices]
train_currents = currents[train_indices]
train_class_label = class_label[train_indices]   # 训练集的类别标签
test_videos = videos[test_indices]
test_labels = labels[test_indices]
test_currents = currents[test_indices]
test_class_label = class_label[test_indices]   # 测试集的类别标签

# print(f"Train video shape: {train_videos.shape}")
# print(f"Train label shape: {train_labels.shape}")
# print(f"Train current shape: {train_currents.shape}")
# print(f"Train class label shape: {train_class_label.shape}")
# print(f"Test video shape: {test_videos.shape}")
# print(f"Test label shape: {test_labels.shape}")
# print(f"Test current shape: {test_currents.shape}")
# print(f"Test class label shape: {test_class_label.shape}")

train_dataset = TensorDataset(train_videos, train_labels, train_currents, train_class_label)


