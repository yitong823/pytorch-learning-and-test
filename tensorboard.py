# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 20:15:58 2023

@author: 焱翊曈
"""
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

writer = SummaryWriter("logs")
#y = x
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)
writer.close()

imag_path = "E:/pytorch/hymenoptera_data/train/ants/0013035.jpg"
from PIL import Image
img_PIL = Image.open(imag_path)
img_array = np.array(img_PIL)

writer.add_image("train", img_array,1,dataformats='HWC')
writer.close()