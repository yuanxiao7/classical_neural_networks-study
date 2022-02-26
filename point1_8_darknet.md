## darknet53

- ### class notes

- 网络模块的主体构架，每一部分由卷积和basic block循环堆叠而成

darknet.py

```python
import math
from collections import OrderedDict  # 存储网络结构

import torch.nn as nn


# 在残差块中，输入的

# 残差结构
# 利用一个1*的卷积下降通道数，然后利用一个3*3的卷积提取特征并且上升通道数
# 最后接上一个残差边

class BesicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BesicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        # 添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前
        # 不会因为数据过大而导致网络性能的不稳定
        self.relu1 = nn.LeakyReLU(0.1)


        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out

class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32

        # [416, 416, 3] -> [416, 416, 32]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)


        # [416, 416, 32]->[208, 208, 64]
        self.layer1 = self._make_layer([32, 64], layers[0])
        # [208, 208, 64]->[104, 104, 128]
        self.layer2 = self._make_layer([64, 128], layers[1])
        # [104, 104, 128]->[52, 52, 256]
        self.layer3 = self._make_layer([128, 256], layers[2])
        # [52, 52, 256]->[26, 26, 512]
        self.layer4 = self._make_layer([256, 512], layers[3])
        # [26, 26, 512]->[13, 13, 1024]
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layer_out_filters = [64, 128, 256, 1024]


        # 进行权值初始化,即卷积核的初始化
        # self.modules(): 继承的方法self.modules()来初始化模型权重
        # nn.Module类中的一个方法:self.modules(), 他会返回该网络中的所有modules
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                # print('m.kernel_size[0]: ', m.kernel_size[0])
                # print('m.out_channels: ', m.out_channels)
                # print('n: ', n)
                # print('m. : ', m.weight.data.normal_(0, math.sqrt(2./n)).shape)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                print()


    # 在每一个layers里面，首先利用一个步长为2的3*3的卷积进行下采样
    # 然后进行残差结构的堆叠


    def _make_layer(self, planes, blocks):
        layers = []  # 字典
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1],
                                           kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        # 加入残差结构
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i),
                           BesicBlock(self.inplanes, planes)))
            return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5


def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model



if __name__ == '__main__':
    model = darknet53()
    # print(model)
```