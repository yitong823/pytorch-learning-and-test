import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data =torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                   download=True)
dataloader = DataLoader(data,batch_size=64)

class yyt(nn.Module):
    def __init__(self):
        super(yyt,self).__init__()
        self.conv1 = Conv2d(3,6,3,stride=1,padding=0)
    def forward(self,x):
        x = self.conv1(x)
        return  x
yt1 =  yyt()
step = 0
writer= SummaryWriter("./logs")
for d in dataloader:
    imgs,targets = d
    output = yt1(imgs)
    print(output.shape)
    writer.add_images("input",imgs,step)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step = step+1
