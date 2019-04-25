import torch
from torch.utils import data

import os
from PIL import Image
import numpy as np
from torchvision import transforms

transform = transforms.Compose(
  [
    transforms.Resize(224),#缩放图片(Image),保持长宽比不变，最短边为224像素
    transforms.CenterCrop(224),#从图片中间切除224*224的图片
    transforms.ToTensor(),#将图片(Image)转成Tensor,归一化至[0,1]
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])#标准化至[-1,1],规定均值和方差
  ]
)


class DogCat(data.Dataset):
  def __init__(self,root):
    imgs = os.listdir(root)
    #所有图片的绝对路径
    #这里不实际加载图片，只是指定路径，当调用__getitem__时才会真正读取图片
    self.imgs = [os.path.join(root,img) for img in imgs]
    self.transform = transform

  def __getitem__(self,index):
    img_path = self.imgs[index]
    label = 1 if 'dog' in img_path.split('/')[-1] else 0
    pil_img = Image.open(img_path)
    if self.transform:
      return self.transform(pil_img),label
    return data,label
  def __len__(self):
    return len(self.imgs)