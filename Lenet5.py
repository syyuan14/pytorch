import torch.nn as nn
import torch.nn.functional as F
class LeNet5(nn.Module):
  def __init__(self):
    super(LeNet5,self).__init__()
    self.conv1 = nn.Conv2d(1,6,5)
    self.conv2 = nn.Conv2d(6,16,5)

    self.fc1 = nn.Linear(16*5*5,120)
    self.fc2 = nn.Linear(120,84)
    self.fc3 = nn.Linear(84,10)

  def forward(self,x):
    #输入x加到第一卷积层
    x = self.conv1(x)
    print("经过第一卷积层后：",x.size())
    #经过激活操作
    x = F.relu(x)
    #卷积层输出加到池化层
    x = F.max_pool2d(x,(2,2))
    print("经过第一池化层后：",x.size())
    #经过第二卷积层
    x = self.conv2(x)
    print("经过第二卷积层后：",x.size())
    #激活操作
    x = F.relu(x)
    #池化
    x = F.max_pool2d(x,2)#如果尺寸为正方形的话，可以只指定一个数
    print("经过第二池化层后：",x.size())

    #裁剪给定尺寸
    x = x.view(-1,self.num_flat_features(x))
    print("裁剪后：",x.size())

    x = F.relu(self.fc1(x))
    print("fc1:",x.size())
    x = F.relu(self.fc2(x))
    print("fc2:",x.size())
    x = self.fc3(x)
    print("fc3:",x.size())
    return x

  def num_flat_features(self,x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
      num_features *= s
    return num_features
