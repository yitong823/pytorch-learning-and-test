# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:04:19 2023

@author: 焱翊曈
"""

import torch
import torch.nn.functional as F

data = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

data = torch.reshape(data, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

output1 = F.conv2d(data,kernel,stride=2) #stride:卷积核每次移动的步长
print(output1)
output2 = F.conv2d(data,kernel,stride=2,padding=1)#padding:矩阵四周填充步长
print(output2)
