import torch
from densenet import DenseNet121
from data_load import GetTrainData

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# 参数设置
num_classes = 100 #分类数，20 or 100
EPOCH = 200   #遍历数据集次数
BATCH_SIZE = 128      #批处理尺寸
LR = 0.1        #初始学习率

# 读取并处理训练集和测试集数据
trainset, validset = GetTrainData(num_classes)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
validloader = torch.utils.data.DataLoader(validset, batch_size = 100, shuffle = False, num_workers = 2)
 
# 模型定义-DenseNet
net = DenseNet121(num_classes).to(device)

# 定义损失函数和优化方式
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = LR, momentum = 0.9, weight_decay = 5e-4)

# 学习率阶梯下降
StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [25*i+50 for i in range(0,6)], gamma = 0.25)

# 训练
if __name__ == "__main__":
    best_acc = 0
    print("Start Training!")
    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
        # 每训练完一个epoch打印一次loss和准确率
        print('[epoch:%d, iter:%d] Loss: %.3f | Acc: %.3f%% '
                % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
        # 验证
        with torch.no_grad():
            correct = 0
            total = 0
            for data in validloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            acc = 100. * correct / total
            print('验证分类准确率为：%.3f%%' % (acc))
            # 保存模型参数
            if acc >= best_acc:
                if num_classes == 20:
                    torch.save(net.state_dict(), 'net_best1.pth')
                else:
                    torch.save(net.state_dict(), 'net_best2.pth')
                best_acc = acc
        StepLR.step()
    print("Training Finished, TotalEPOCH=%d" % EPOCH)