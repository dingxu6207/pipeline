# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:23:45 2020

@author: dingxu
"""

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data

 
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2)+0.2*torch.rand(x.size())                   # noisy y data (tensor), shape=(100, 1)
 
# 画图
plt.figure(0)
plt.scatter(x.numpy(), y.numpy())

 
LR = 0.01
BATCH_SIZE = 500
EPOCH = 120


# put dateset into torch dataset
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset = torch_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)

# default network
class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出
 
    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x
    
net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)  # net 的结构

opt_Adam  = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
loss_func = torch.nn.MSELoss()  



'''
torch.save(net, 'net.pkl')  # 保存整个网络
torch.save(net.state_dict(), 'net_params.pkl')

net2 = torch.load('net.pkl')
prediction1 = net2(x)

plt.figure(2)
plt.scatter(x.data.numpy(), prediction1.data.numpy())
'''


    


losstemp = []
epoachtemp = []
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(loader):          # for each training step
        output = net(b_x)              # get output for every net
        loss = loss_func(output, b_y)  # compute loss for every net
        
        opt_Adam.zero_grad()   # 清空上一步的残余更新参数值
        loss.backward()         # 误差反向传播, 计算参数更新值
        opt_Adam.step()
        
        lossdata = loss.data.numpy()

    print('epoach:', epoch, '   loss:',lossdata)
       
    
    losstemp.append(lossdata)
    epoachtemp.append(epoch)
    
    plt.figure(4)
    plt.cla()
    plt.subplot(121)
    plt.plot(epoachtemp, losstemp)
    plt.subplot(122)
    plt.scatter(b_x.data.numpy(), b_y.data.numpy())
    plt.scatter(b_x.data.numpy(), output.data.numpy())
    plt.pause(0.1)
    
    
       