import torch
import torch.nn as nn

class AlexNet(nn.Module):
  """ alexnet """
  def __init__(self):
    super(AlexNet,self).__init__()
    self.conv1 = nn.Sequential(
      nn.Conv2d(3,96,11,4,0),
      nn.MaxPool2d(5)
      nn.ReLU(),
    )