import torch
import numpy as np
import pandas as pd
from data_load import GetTestData
from densenet import DenseNet121

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 100 #分类数，20 or 100
net = DenseNet121(num_classes)

# 读取网络参数和打开csv文件
if num_classes == 20:
    net.load_state_dict(torch.load('net_best1.pth'))
    df = pd.read_csv('results/1.csv')
else:
    net.load_state_dict(torch.load('net_best2.pth'))
    df = pd.read_csv('results/2.csv')

net = net.to(device)
torch.no_grad()
# 读取测试集数据
testset = GetTestData(num_classes)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
all_predicted = []
# 预测
for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.cpu().numpy().tolist()
    all_predicted+=predicted
print(all_predicted)

# 保存预测结果
if num_classes == 20:
    df['coarse_label'] = all_predicted
    df.to_csv('results/1.csv', index=False)
else:
    df['fine_label'] = all_predicted
    df.to_csv('results/2.csv', index=False)