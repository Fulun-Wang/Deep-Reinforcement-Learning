import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose(
    [transforms.ToTensor(),  # 将numpy数组或PIL.Image读的图片转换成(C,H, W)的Tensor格式且把灰度范围从0-255变换到0-1之间
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 归一化为[-1, 1]
# 由于torchvision的datasets的输出是[0,1]的PILImage，所以我们先先归一化为[-1,1]的Tensor
# 首先定义了一个变换transform，利用的是上面提到的transforms模块中的Compose( )
# 把多个变换组合在一起，可以看到这里面组合了ToTensor和Normalize这两个变换
trainset = torchvision.datasets.CIFAR10(root='D:\编程\新建文件夹', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='D:\编程\新建文件夹', train=True,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=True, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 展示图片函数
# def imshow(img):
#     img = img / 2 + 0.5  # 反归一化
#     npimg = img.numpy()  # tensor转化为numpy
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 第一维与第三维转换
#     plt.show()

# 随机获取训练集图片训练
# dataiter = iter(trainloader)  # 创建一个python迭代器，读入的是我们第一步里面就已经加载好的testloader
# images, labels = dataiter.next()

# 展示图片
# imshow(torchvision.utils.make_grid(images))  # torchvision.utils.make_grid(images)将四张图片拼成一张图片

# 打印图片类别标签
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# 构建一个卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# 定义损失函数和优化器和GPU加速
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 训练网络
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # GPU加速
        optimizer.zero_grad()
        outputs = net(inputs)
        # print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 打印统计信息
        running_loss += loss.item()
        if i % 2000 == 1999:
            # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1,running_loss / 2000))
            running_loss = 0.0

print('Finish Training')

# 网络输出
# outputs = net(images)

# 预测结果
# _, predicted = torch.max(outputs, 1)
# print('Predicted:', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


# 测试集的准确率
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         # print(labels.size(0))
#         outputs = net(images)
#         # print(outputs.shape)
#         _, predicted = torch.max(outputs.data, 1)
#         # 返回输入Tensor中每行的最大值，并转换成指定的dim（维度）
#         total += labels.size(0)  # 更新测试图片的数量
#         correct += (predicted == labels).sum().item()  # 更新正确分类的图片的数量
# print('Accuracy of the network on the 10000 test images: %d %%' %(100 * correct / total))


# 每个类别的分类准确率
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()  # 每一个batch的(predicted==labels) 这段代码其实会根据预测和真实标签是否相等，输出 1 或者 0
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1
#
# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

