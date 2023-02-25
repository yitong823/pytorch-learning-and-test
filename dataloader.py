# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 22:01:37 2023

@author: 焱翊曈
"""

import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset",train=False,
                                         transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,
                         num_workers=0,drop_last=False)
#测试数据集中第一张图片和target
img,target = test_data[0]

writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs , targets = data
    writer.add_images("test_data", imgs,step)
    step = step+1

writer.close()









