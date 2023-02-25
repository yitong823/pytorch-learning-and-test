import torch
from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.Tensor()
data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(data,batch_size=64)

class Yyt(nn.Module):
    def __init__(self):
        super(Yyt,self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output = self.maxpool(input)
        return  output
yyt  = Yyt()

writer = SummaryWriter("./logs_maxpool")
step = 0
for d in dataloader:
    imgs,targets = d
    writer.add_images("input",imgs,step)
    output = yyt(imgs)
    writer.add_images("output",output,step)
    step+=1
writer.close()





