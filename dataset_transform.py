# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:15:47 2023

@author: 焱翊曈
"""
import torchvision
from torch.utils.tensorboard import SummaryWriter


trans_totensor = torchvision.transforms.ToTensor()

train_set = torchvision.datasets.CIFAR10(root="./dataset",train = True,transform=trans_totensor, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train = False,transform=trans_totensor,download=True)

print(test_set[0])
print(test_set.classes)

print(test_set.classes[2])

writer = SummaryWriter("cifar10")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set", img,i)

writer.close()




