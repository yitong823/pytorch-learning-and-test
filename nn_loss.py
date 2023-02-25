import torch
from torch import nn

inputs = torch.Tensor([1,2,3])
targets = torch.Tensor([1,2,5])

inputs = torch.reshape(inputs,[1,1,1,3])
targets = torch.reshape(targets,[1,1,1,3])

#l1loss测试
loss = nn.L1Loss(reduction='mean')
result = loss(inputs,targets)

#MSEloss测试
loss_mse = nn.MSELoss()
result1  = loss_mse(inputs,targets)

#交叉熵loss测试
x = torch.tensor([0.1,0.9,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x,y)



print(result_cross)

