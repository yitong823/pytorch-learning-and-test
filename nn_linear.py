import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

data = torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor(),
                                    download=True)
dataloader = DataLoader(data,batch_size=64)


class Yyt(nn.Module):
    def __init__(self):
        super(Yyt, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


yyt = Yyt()

for d in dataloader:
    imgs, targets = d
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = yyt(output)
    print(output.shape)
