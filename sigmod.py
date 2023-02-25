import torch
import torchvision
from  torch import nn
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(data,batch_size=64)
class Yyt(nn.Module):
    def __init__(self):
        super(Yyt,self).__init__()
        self.sigmoid = Sigmoid()
    def forward(self,input):
        output = self.sigmoid(input)
        return  output
yyt = Yyt()
writer = SummaryWriter("logs")
step = 0
for d in dataloader:
    imgs,targets = d
    writer.add_images("sigmoid_input",imgs,global_step=step)
    out = yyt(imgs)
    writer.add_images("sigmoid_out",out,step)
    step+=1

writer.close()