import os
import torch
from torch.utils import data
import numpy as np
from scipy.io import loadmat
from PIL import Image

class WHU_OHS_Dataset(data.Dataset):
    def __init__(self, image_file_list, label_file_list,edge_file_list=None, use_3D_input=False, channel_last=False):
        self.image_file_list = image_file_list
        self.label_file_list = label_file_list
        self.edge_file_list = edge_file_list
        self.use_3D_input = use_3D_input
        self.channel_last = channel_last

    def sample_stat(self):
        sample_per_class = torch.zeros([24])
        for label_file in self.label_file_list:
            label = Image.open(label_file)
            label = np.array(label)
            count = np.bincount(label.ravel(), minlength=25)
            count = count[1:25]
            count = torch.tensor(count)
            sample_per_class = sample_per_class + count

        return sample_per_class

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        image_file = self.image_file_list[index]
        label_file = self.label_file_list[index]
        name = os.path.basename(image_file)

        # 使用os.path.abspath获取MAT文件的绝对路径
        image_file = os.path.abspath(image_file)
        # print(name)

        # 从MAT文件中加载图像数据
        image_data = loadmat(image_file)
        image = image_data['image_data']

        label_data = loadmat(label_file)
        label = label_data['label_data']
        # edge = None
        if self.edge_file_list is not None:
            edge_file = self.edge_file_list[index]
            edge_data = loadmat(edge_file)
            edge = edge_data['label_data']  # 假设.mat文件中存储的键为'edge_data'
            edge = torch.tensor(edge, dtype=torch.long)


        assert image is not None, '图像为空'

        if self.channel_last:
            image = image.transpose(1, 2, 0)

        image = torch.tensor(image, dtype=torch.float) / 10000.0

        if self.use_3D_input:
            image = image.unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long) - 1

        return image, label, name