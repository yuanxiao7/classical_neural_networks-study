## LeNet 5

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
    batchsz = 32  # 每一次并行处理13张图片

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
    modle = Lenet5().to(device)  # 返回一样的modle
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





if __name__ == '__main__':
    main()
```



LeNet.py

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