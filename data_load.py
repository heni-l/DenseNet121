import torch
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
torch.manual_seed(1)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
 
transform_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class MyDataset(Dataset):
    def __init__(self,num_classes, train = True, test = False):
        self.num_classes = num_classes
        if test == True:
            data = np.load('test.npy')
        else:
            data = np.load('train.npy')
        # 将每个行向量恢复成3*32*32的图像
        pic = []
        for temp in data:
            temp = np.array(temp)
            r = temp[:1024].reshape(32, 32, 1)
            g = temp[1024:2048].reshape(32, 32, 1)
            b = temp[2048:].reshape(32, 32, 1)
            temp = np.concatenate((r, g, b), -1)
            pic.append(temp)

        if test == True:
            self.pic = pic
            self.labels = [0 for img in pic]
            self.transforms = transform_valid
        else:
            if num_classes == 20:
                df = pd.read_csv('train1.csv')
                labels = df['coarse_label'].values
            elif num_classes == 100:
                df = pd.read_csv('train2.csv')
                labels = df['fine_label'].values
            #将数据分为训练集和测试集，比例为9:1
            pic_train, pic_valid, label_train, label_valid\
                = train_test_split(pic, labels, test_size=0.1, random_state=61)
            if train:
                self.pic = pic_train
                self.labels = label_train
                self.transforms = transform_train
            else:
                self.pic = pic_valid
                self.labels = label_valid
                self.transforms = transform_valid
    
    def __getitem__(self, index):
        img = self.pic[index]
        img = Image.fromarray(np.uint8(img))
        img = self.transforms(img)
        label = self.labels[index]
        return img, label
    
    def __len__(self):
        return len(self.labels)

#获取训练集和验证集数据
def GetTrainData(num_classes):
    return MyDataset(num_classes), MyDataset(num_classes, train = False)

#获取测试集数据
def GetTestData(num_classes):
    return MyDataset(num_classes, train = False, test = True)