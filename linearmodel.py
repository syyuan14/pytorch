import numpy as np 
import torch
import torch.nn as nn

# 产生数据
x = np.random.rand(256)#随机产生256个数字
noise = np.random.randn(256)#产生服从高斯分布的256个数字

y = 8*x + 7 + noise

#定义模型
model = nn.Linear(1,1)#输入特征维度为1，输出特征维度为1

#定义计算loss的方法
criterion = nn.MSELoss()
#定义优化器
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
#准备训练数据
x_train = x.reshape(-1,1).astype('float32')
y_train = y.reshape(-1,1).astype('float32')
#定义训练次数
epochs = 50000
#开始训练
for i in range(epochs):
  inputs = torch.from_numpy(x_train)
  labels = torch.from_numpy(y_train)

  outputs = model(inputs)

  optimizer.zero_grad()

  loss = criterion(outputs,labels)

  loss.backward()

  optimizer.step()
  if i%1000==0:
    print('epoch:{},loss:{:1.5f}'.format(i,loss))

[w,b]=model.parameters()
print('w:',w.item(),'b:',b.item())