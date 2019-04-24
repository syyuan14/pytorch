import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        #input: 1*32*32
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # 6*28*28
            nn.ReLU(),  # 6*28*28
            nn.MaxPool2d(2, 2)  # 6*14*14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),  # 16*10*10
            nn.ReLU(),  # 16*10*10
            nn.MaxPool2d(2, 2)  # 16*5*5
        )
        self.fc = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, input):
        # input 1*32*32
        out = self.conv1(input)
        out = self.conv2(out)
        out = out.view(-1, self.num_flat_features(out))
        out = self.fc(out)
        return out

    def num_flat_features(sellf, x):
        # x.size()返回值为(256,16,5,5),size的值是(16,16,5),256是batch_size
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
