# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 22:12:20 2023

@author: 焱翊曈
"""
from PIL import Image
from torchvision import transforms
img_path = "E:/pytorch/hymenoptera_data/train/ants/6240329_72c01e663e.jpg"
img=Image.open(img_path)

#ToTenor使用：将PIL数据类型转换为tensor数据类型
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)

import cv2
cv_img = cv2.imread(img_path)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs")
writer.add_image("tensor_img",tensor_img)
writer.close()

#Normalization
'''output[channel] = (input[channel] - mean[channel]) / std[channel]'''
trans_norm = transforms.Normalize([3,2,4], [4,5,1])
img_norm = trans_norm(tensor_img)
writer.add_image("tensor_img", img_norm,2)
writer.close()


#Resize
#改变图像尺寸
print(img.size)
trans_resize = transforms.Resize((512,512))
#img PIL -> resize-> imgresize PIL
img_resize = trans_resize(img)
#img_resize PIL -> totensor -> img_resize tensor
img_resize = tensor_trans(img_resize)
writer.add_image("Resize", img_resize,0)
print(img_resize)
writer.close()


#Compose
trans_rrsize2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_rrsize2,tensor_trans])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2,1)
writer.close()

#randomCrop 随机裁剪
trans_random = transforms.RandomCrop(300)
trans_compose_2 = transforms.Compose([trans_random,tensor_trans])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("randomcrop", img_crop,i)

writer.close()



























