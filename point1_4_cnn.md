# CNN

- class notes

main.py

```python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from LeNet_5 import Lenet5

def main():
    batchsz = 32  # 每一次并行处理32张图片

    # datasets下载 存到当前路径，名为cifar的文件 并由numpy转化为tensor类型
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),  # 每一张照片为[32, 32]
        transforms.ToTensor()
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label: ', label.shape)
    '''
    Extracting cifar\cifar - 10 - python.tar.gz to cifar 
    Files already downloaded and verified
    x: torch.Size([32, 3, 32, 32]) label: torch.Size([32])
    '''

    device = torch.device('cuda')
    modle = Lenet5().to(device)  # lenet5 modle
    
    # model = ResNet18().to(device)  # resnet18 modle
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(modle.parameters(), lr=1e-3)
    print(modle)


    for epoch in range(1000):  # 训练1000次
        modle.train()  # 执行train模块
        for batchidx, (x, label) in enumerate(cifar_train):  # 完成一个for即过一遍train和test 共60000张图片=epoch
            #[b, 3, 32, 32]
            #[b]
            x, label = x.to(device), label.to(device)

            logits = modle(x)
            #logits: [b, 10]
            #label: [b]
            loss = criteon(logits, label)  # 使用交叉熵损失函数

            #backprop
            optimizer.zero_grad()  # 每一次调用都是add，要清除上一次的数据再进行优化
            loss.backward()
            optimizer.step()

        print(epoch, loss.item())

        #test
        modle.eval()  # 验证模块
        with torch.no_grad():  #不需要backward，让他不会打乱原来的计算图

            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                #[b, 3, 32, 32]
                #[b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = modle(x)  # logits.argumax = loss.argumax
                # [b]
                pred = logits.argmax(dim=1)

                # [b] vs [b] => scolar tensor
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)

        acc = total_correct / total_num
        print(epoch, acc)

	   # the result:
        # 999 0.4639



if __name__ == '__main__':
    main()
```



## LeNet 5

LeNet.py  在main.py中调用

```python
import torch
from torch import nn
from torch.nn import functional as F


class Lenet5(nn.Module):
    '''
    for cirfar10 dataset.
    '''
    def __init__(self):
        super(Lenet5, self).__init__()

        #卷积层  输入测试[2, 3, 32, 32]
        self.conv_unit = nn.Sequential(
            # x: [b, 3, 32, 32] => [b, 6,...]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            # picture channel=3，6个5*5的核， 得[2, 6, 28, 28]

            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # [2, 6, 14, 14]
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # [2, 16, 10, 10]
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # [2, 16, 5, 5]
        )

        # flatten 输入全连接
        # fc unit
        self.fc_unit = nn.Sequential(
            # 未知输入时随意一个nn.Linear(2, 120), 后面再将打平的数据输入，即2->16*5*5
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)

        )

        # [b, 3, 32, 32]测试
        # tmp = torch.randn(2, 3, 32, 32)
        # out = self.conv_unit(tmp)
        # print('conv out: ', out.shape)

        # 结果torch.Size([2, 16, 5, 5])

        # use Cross Entropy Loss （一般分类用交叉熵较好，均方差用于回归问题）
        #self.criteon = nn.CrossEntropyLoss()  #criteon 评价标准，loss的计算方法



    def forward(self, x):
        """

        :param x: [b, 3, 32, 32]
        :return:
        """
        batchsz = x.size(0)  # 即b
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv_unit(x)

        # [b, 16, 5, 5] => [b, 16*5*5]
        x = x.view(batchsz, 16*5*5)  # 卷积到全连接要手动打平图片

        # [b, 16*55*5] => [b, 10]
        logits = self.fc_unit(x)

        # # [b, 10]  放到类里面
        # # pred = F.softmax(logits, dim=1) main函数调用的CrossEntropyLoss已有softmax
        # loss = self.criteon(logits, y)
        return logits





 # 小型测试 计算各网络参数
def main():
    net = Lenet5()
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)

    print('lenet out: ', out.shape)


if __name__ == '__main__':
    main()
```



## ResNet

resnet.py  在main.py中调用

```python
import torch
from torch import nn
from torch.nn import functional as F



class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):  # 由输入控制通道数量
        """      lenet18      64     128        2

        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)  # 合理放缩，在零附近均匀分布

        self.extra = nn.Sequential()

        # [b, ch, h, w[ => [b, ch_out, h, w]
        # 将输入的channel转为输出的channel相匹配,
        # 长宽也要匹配 即stride要与conv1的输出一致
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        # print('1: ', out.shape)
        out = self.bn2(self.conv2(out))
        # print('2: ', out.shape)

        # //////  short cut.
        # extra module: [b, ch_in, h, w] with [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out  # 连接处 单张图片的单个通道自动广播成维度相同的tensor再相加
        out = F.relu(out)
        # print('out: ', out.shape)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        # 预处理层
        # 一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64))

        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h, w]
        self.blk1 = ResBlk(64, 128, stride=2)

        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(128, 256, stride=2)

        # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlk(256, 512, stride=2)

        # [b, 512, h, w] => [b, 1024/512, h, w] 一般提到升到512
        self.blk4 = ResBlk(512, 512, stride=2)

        self.outlayer = nn.Linear(512*1*1, 10)  #实例，最后一层为full connect



    def forward(self, x):

        x = F.relu(self.conv1(x))
        # print('x: ', x.shape)
        # [2, 64, 10, 10]

        x = self.blk1(x)
        # print('x1: ', x.shape)
        # [2, 128, 5, 5]

        x = self.blk2(x)
        # print('x2: ', x.shape)
        # [2, 256, 3, 3]

        x = self.blk3(x)
        # print('x3: ', x.shape)
        # [2, 512, 2, 2]

        x = self.blk4(x)
        # print('x4: ', x.shape)
        # [2, 512, 2, 2]


        # print('after conv: ', x.shape)  # [b, 512, 2, 2]
        # [b, 512, h, w] => [b, 512, 1, 1]
        x = F.adaptive_max_pool2d(x, [1, 1])  # pool2使图片只有一个像素值[1, 1]
        # print('after pool: ', x.shape)
        x = x.view(x.size(0), -1)  # 打平成[b, 512*1*1]
        x = self.outlayer(x)  # full connect
        return x

def main():

    # 随channel的增加stride减少
    blk = ResBlk(64, 128, stride=4)  # 初始化
    tmp = torch.randn(2, 64, 32, 32)
    out = blk(tmp)
    print('block: ', out.shape)

    x = torch.randn(2, 3, 32, 32)
    model = ResNet18()
    out = model(x)
    print('resnet: ', out.shape)

if __name__ == '__main__':
    main()
```