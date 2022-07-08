# class notes

- The first of using pytorch

### 项目说明：

- 这个md文件里的是一个利用pytorch写的一个小小的全连接网络（Fc）来实现手写数字识别，并用matplotlib来实现原图片及预测图片，loss的可视化展示。你只要复制过去，放到你的编辑器里跑就行了，如果没有下载第三方库的话，你可以按照提示，到cmd里面用conda或者pip下载，友情提示，pip下载时，请关掉你的代理，否则极有可能会出现报错。



### 项目文件

#### mnist_train.py

- 这是主函数，数据传入，激活函数，网络处理，优化器，loss 等相关内容。
- 在这个小项目中，up主的准确率为0.8923，我改了batch_size 准确率为0.9778 意外提升0.1左右，但是batch_size大的话，所得的loss比较稳定 batch_size小则loss浮动大，loss本身的大小不重要，他所呈现的趋势才能体现网络的效果。

```python
import torch
from torch import nn  # 网络
from torch.nn import functional as F  # 常用函数 # 由于激活函数（ReLu、sigmoid、Tanh）、池化（MaxPool）等层没有可学习的参数，可以使用对应的functional函数
from torch import optim  # 优化
import torchvision  # 加载数据集及可视化
from utils import plot_image, plot_curve, one_hot

batch_size = 4  # 数据原因，512数据不能整除

# batch_size = 512并行处理图片，一次处理512张
# step1.load dataset
# 'mnist_data'为mnist的下载路径，共70k，参数train=True让60k为train，10k为test，
# download=True如果当前文件没有mnist的文件，就从网上下载（numpy格式），
# torchvision.transforms.ToTensor()将numpy格式的数据集转化为Tensor格式
# 原图片数据是在0的右侧的，用Normalize减去0.1307再除0.081标准差，使图片数据在0-1均匀分配，(优化网络性能，可提10%左右)
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

x, y = next(iter(train_loader))  # next返回迭代的下一个元素，iter方法得到迭代器的元素，迭代器为list，tuple等对象
print(x.shape, y.shape, x.min(), x.max())

plot_image(x, y, 'image sample')  # 调用函数画数字图


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # xw+b第一层
        self.fc1 = nn.Linear(28 * 28, 256)  # 生成[784,256]矩阵
        self.fc2 = nn.Linear(256, 64)  # [256,64]
        self.fc3 = nn.Linear(64, 10)  # [64,10]

    def forward(self, x):  # 前向传播

        # x:[b,1,28,28]图片
        # h1 = relu(xw1+b1)  relu非线性激活函数
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b1)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3 这里按需求加不加激活函数，我们这里不加，采用均方差
        x = self.fc3(x)

        return x


net = Net()
# [w1,b1,w2,b2,w3,b3]
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_loss = []

# 大循环对整个数据集迭代3次，小循环对数据集整个迭代
for epoch in range(3):
    for batch_idx, (x, y) in enumerate(train_loader):
        #  x: [b,1,28,28], y: [12]
        #     print(x.shape, y.shape)
        #     break
        # break
        # torch.Size([12, 1, 28, 28])  torch.Size([12])
        # torch.Size([12, 1, 28, 28])  torch.Size([12])
        # torch.Size([12, 1, 28, 28])  torch.Size([12])
        # 接受的是一维的，需要打平 [b,1,28,28] => [b,feature]
        x = x.view(x.size(0), 28 * 28)
        # =>[b,10] 每一类的概率
        out = net(x)
        # [b,10]
        y_onehot = one_hot(y)  # 得到4个0-1的list
        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)  # 返回一个标量

        optimizer.zero_grad()  # 清零modle参数
        loss.backward()

        # w' = w -lr*grad
        optimizer.step()  # 更新

        train_loss.append(loss.item())

        if batch_idx % 1000 == 0:
            print(epoch, batch_idx, loss.item())  # item()变成numpy类型的数值

# print('train_loss: ',train_loss)
# print('train_loss.shape: ',train_loss.shape)  list无shape


# plot_curve(train_loss)  # batchsize要调大再调用函数画loss


total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28 * 28)
    out = net(x)
    # out: [b,10] => pred: [b]
    pred = out.argmax(dim=1)
    print(y)
    print('y: ', y.shape)
    print(pred)
    print('pred: ', pred.shape)
    print(pred.eq(y))
    correct = pred.eq(y).sum().float().item()
    total_correct += correct
    print(total_correct)

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc: ', acc)

# 随机测试一个 batch
x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28 * 28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')



```



#### utils.py

- 这部分则是可视化展示的代码，这里推荐博主的 “莫烦” python教程。 

```python
import torch
from matplotlib import pyplot as plt


def plot_curve(data):  # 画loss的曲线
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


def plot_image(img, label, name):  # 画mnist图片
    fig = plt.figure()
    for i in range(4):  # 一次显示6张图
        plt.subplot(2, 3, i + 1)
        # 使用plt.subplot来创建小图. plt.subplot(2,3,i+1)表示将整个图像窗口分为2行3列, i+1每行的第几个图像
        plt.tight_layout()  # 修饰好坐标轴标签、刻度标签以及标题的部分
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        # imshow通过色差、亮度来展示数据的差异
        plt.title("{}: {}".format(name, label[i].item()))  # item()将data转化为数值
        plt.xticks([])
        plt.yticks([])
    plt.show()


def one_hot(label, depth=10):  # 编码工具
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    # scatter填充self[ index[i][j] ] [j] = src[i][j]，此处self为out，[j]=value
    return out
```





## mnist CPU and GPU

- 通常来说，做深度学习，gpu对我们跑算法的速度提升是很有用的，以下就是将模型搬到gpu的代码展示。

```python
# 模型一:

import torch
from torch import nn  # 网络
from torch.nn import functional as F  # 常用函数 # 由于激活函数（ReLu、sigmoid、Tanh）、池化（MaxPool）等层没有可学习的参数，可以使用对应的functional函数
from torch import optim  # 优化
import torchvision  # 加载数据集及可视化
from matplotlib import pyplot as plt

batch_size = 200
learning_rate = 0.01
epochs = 10

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

w1, b1 = torch.rand(200, 784, requires_grad=True), \
         torch.zeros(200, requires_grad=True)
w2, b2 = torch.rand(200, 200, requires_grad=True), \
         torch.zeros(200, requires_grad=True)
w3, b3 = torch.rand(10, 200, requires_grad=True), \
         torch.zeros(10, requires_grad=True)

# 合理的初始化
torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)


def forward(x):
    x = x @ w1.t() + b1
    x = F.relu(x)
    x = x @ w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    x = F.relu(x)
    return x

#device = torch.device('cuda:0')
optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
criteon = nn.CrossEntropyLoss()  # 实例化一个criteon对象

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)

        logits = forward(data)
        loss = criteon(logits, target)  # 使用交叉熵损失函数

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(),w2.grad.morm())
        optimizer.step()

        if batch_idx % 100 == 0:
            print('train Epoch: {} [{}/{} ({:.0f})]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()
            ))
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        logits = forward(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    print(100. * correct / len(test_loader.dataset))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {: .4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))


```

```python
#模型二: #搬到GPU上运行


import torch
from torch import nn    #网络
from torch import optim    #优化
import torchvision     #加载数据集及可视化
#也可以这样调用torchvision
# from torchvisn import dataset, transforms 这样下载就不用写torchvision.了



batch_size = 200
learning_rate = 0.01
epochs = 10

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


device = torch.device('cuda:0')  #使用cuda
net = MLP().to(device)   #将网络模型搬到cuda

optimizer = optim.SGD(net.parameters(), lr=learning_rate)
#net.parameters()中param就是fc1.weight、fc1.bias等对应的值
# 可以查看神经网络的参数信息，用于更新参数，或者用于模型的保存  这里用于优化

criteon = nn.CrossEntropyLoss().to(device)  # loss搬进cuda


for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)  #将最后一维打平
        data, target =data.to(device), target.cuda()

        logits = net(data)
        loss = criteon(logits, target)  # 使用交叉熵损失函数

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(),w2.grad.morm())

        optimizer.step()

        if batch_idx % 100 == 0:
            print('train Epoch: {} [{}/{} ({:.0f})]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()
            ))
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        #torch.max()会返回数值和索引
        #torch.max()[0]， 只返回最大值的每个数
        # troch.max()[1]， 只返回最大值的每个索引


        correct += pred.eq(target.data).sum()

    print(100. * correct / len(test_loader.dataset))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {: .4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
```



