import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import numpy as np


class DogCat(Dataset):
    """
    自定义dataset类
    """

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        主要目标：将图片的路径存放起来,并根据训练，测试,验证划分数据进行不同的预处理
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        # test1:data/test1/8973.jpg
        # train:data/train/cat.10004.jpg
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(
                x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)
        # shuffle imgs
        np.random.seed(100)
        imgs = np.random.permutation(imgs)  # 输入一个list然后将其打乱返回

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(imgs_num*0.7)]
        else:
            self.imgs = imgs[int(imgs_num*0.7):]

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225])

            if self.test or not train:
                self.transforms = T.Compose([
                    T.Scale(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transform = T.Compose([
                    T.Scale(256),
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """ 
        获取一张图片的数据并返回label,data
        """
        img_path = self.imgs[index]
        # test1:data/test1/8973.jpg
        # train:data/train/cat.10004.jpg
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in self.imgs[index].split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data,label

    def __len__(self):
        """
        获取总的数据集个数
        """
        return len(self.imgs)
