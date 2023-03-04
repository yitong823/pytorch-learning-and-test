import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torch import nn
#from torch.utils.tensorboard import SummaryWriter
import time

#准备数据集
train_data = torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                          download=True)
print("训练集的长度：",len(train_data))
print("测试集的长度",len(test_data))

#Dataloader加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

#搭建神经网络
class Yyt(nn.Module):
    def __init__(self):
        super(Yyt,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)

        )

    def forward(self,x):
        x = self.model(x)
        return x

#创建网络模型
yyt = Yyt()

#创建损失函数
loss_fn = nn.CrossEntropyLoss()
#优化器
optim = torch.optim.SGD(yyt.parameters(),lr=0.01)

#设置训练网络参数
total_train_step = 0 #记录训练次数
total_test_step = 0 #记录测试次数
epoch = 10 #训练的轮数

#添加tensorboard
#writer = SummaryWriter("./logs")
start_time = time.time()
for i in range(epoch):
    print("-----第{}轮训练开始--------".format(i+1))
    for data in train_dataloader:
        imgs,targets = data
        out = yyt(imgs)
        loss = loss_fn(out,targets)
        # 优化器优化模型
        optim.zero_grad()
        loss.backward()
        optim.step()
        #记录训练次数
        total_train_step+=1
        if total_train_step % 100 ==0:
            end_time = time.time()
            print("用时：",end_time-start_time)

            print("训练次数：{}，Loss{}".format(total_train_step,loss.item()))
 #           writer.add_scalar("train_loss",loss,total_train_step)



    #测试步骤
    totally_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targrts = data
            out = yyt(imgs)
            loss = loss_fn(out,targets)
            totally_test_loss+=loss
            #计算正确率
            accuracy = (out.argmax(1)==targrts).sum()
            total_accuracy+=accuracy
    print("整体测试集的loss：{}".format(totally_test_loss))
    print("整体测试集的正确率{}".format(total_accuracy/len(test_data)))
    #writer.add_scalar("test_accuracy",total_accuracy/len(test_data),total_test_step)
    #writer.add_scalar("test_loss",totally_test_loss,total_test_step)
    total_test_step+=1














