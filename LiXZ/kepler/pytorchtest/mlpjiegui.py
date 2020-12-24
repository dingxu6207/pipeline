# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 21:31:55 2020

@author: dingxu
"""

import numpy as np
from random import shuffle
import torch
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 1000
EPOCH = 10000


data = np.loadtxt('savedatasample2.txt')

data = data[data[:,100]>70]
data = data[data[:,101]<0.4]
#data = data[data[:,103]<1.2]
print(len(data))


data[:,100] = data[:,100]
data[:,101] = data[:,101]*100
data[:,102] = data[:,102]*100
data[:,103] = data[:,103]*100

for i in range(len(data)):
    data[i,0:100] = -2.5*np.log10(data[i,0:100]) 
    data[i,0:100] = (data[i,0:100])-np.mean(data[i,0:100])
    
    
shuffle(data)

P = 0.8
duan = int(len(data)*0.8)

dataX = data[:duan,0:100]
dataY = data[:duan,100:104]
#dataY[:,0] = dataY[:,0]/90

testX = data[duan:,0:100]
testY = data[duan:,100:104]
#testY[:,0] = testY[:,0]/90aaa


train_features = torch.from_numpy(dataX)
train_labels = torch.from_numpy(dataY)

test_features = torch.from_numpy(testX)
test_labels = torch.from_numpy(testY)

train_set = TensorDataset(train_features,train_labels)
test_set = TensorDataset(test_features,test_labels)

#定义迭代器
train_data = DataLoader(dataset=train_set,batch_size=BATCH_SIZE,shuffle=True)
test_data  = DataLoader(dataset=test_set,batch_size=BATCH_SIZE,shuffle=False)

# default network
class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.hidden1 = torch.nn.Linear(n_feature, 50)   # 隐藏层线性输出
        self.hidden2 = torch.nn.Linear(50, 20)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(20, n_output)   # 输出层线性输出
 
    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = F.relu(self.hidden1(x))      # 激励函数(隐藏层的线性值)
        x = F.relu(self.hidden2(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x
    
net = Net(n_feature=100, n_hidden=100, n_output=4)
print(net)  # net 的结构

opt_Adam  = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
loss_func = torch.nn.MSELoss() 


losstemp = []
epoachtemp = []
testlosstemp = []
for epoch in range(EPOCH):
    train_loss = 0
    for step, (b_x, b_y) in enumerate(train_data):          # for each training step
        output = net(b_x.float())              # get output for every net
        loss = loss_func(output, b_y.float())  # compute loss for every net
        
        opt_Adam.zero_grad()   # 清空上一步的残余更新参数值
        loss.backward()         # 误差反向传播, 计算参数更新值
        opt_Adam.step()
        
        
        lossdata = loss.data.numpy()
        datay = output.data.numpy()
        traindatay = b_y.float().data.numpy()
        
     
        
        
    for edata, elabel in test_data:
        # 前向传播
        y_ = net(edata.float())
        # 记录单批次一次batch的loss，测试集就不需要反向传播更新网络了
        testloss = loss_func(y_, elabel.float())
        # 累计单批次误差
        testlossd = testloss.data.numpy()
        datay_ =  y_.data.numpy()
        testdatay = elabel.float().data.numpy()
       
        

    print('epoach:', epoch, 'trainloss:',lossdata, 'testloss:', testlossd)
       
    losstemp.append(lossdata)
    testlosstemp.append(testlossd)
    epoachtemp.append(epoch)
    
    
    plt.figure(4)
    plt.cla()
    plt.subplot(121)
    plt.plot(epoachtemp, losstemp, 'r')
    plt.plot(epoachtemp, testlosstemp, 'b')
    plt.subplot(122)
    plt.scatter(testdatay[:,0], datay_[:,0])
    plt.pause(0.1)
    


