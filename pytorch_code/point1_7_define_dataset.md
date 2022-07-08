# 自定义数据集

- ### class   notes

## 说明

### 一、初始化

- 加载图片路径，采用字典存储对象（名称）及对应的标签，切割路径得到对象为key，下标为label，人为编码
- 生成key（对象）对应label的csv文件（也可以是其他的文件类型）
  - 注意csv文件存的是对象的path，csv是以list形式储存每一个path
- 根据csv文件按比例裁剪数据  train / validation / test

### 二、返回长度

- len(self.images) 返回的是被裁剪之后的关系

### 三、数据转化

- 根据idx得到path（每一张图片路径）

- transform

  ​	path string --> image data (numpy) --> resize -->tensor

  ​	label int --> tensor

- return img , label



## 项目简介

- 当我们开始创建我们的项目时，不过是大项目还是小型的像现在这个小编写的图片识别，他们的基本过程都是一样的，所以我们大概记住一个写法，以后都这样去处理项目，会比较清晰且方便。

- 基本过程：

  1. 下载数据集
  2. 获取数据集的标签（一般大型常用的数据集会给标签的，这里是没有标签的处理），这里采用one_hot编码
  3. 对数据集和标签划分 训练，测试和验证三部分
  4. 导入数据集
  5. 对数据集做一个resize的操作及其他的预处理
  6. 网络构建，这里用的是resnet18
  7. 将数据集传入网路
  8. 加激活函数，损失函数，正则化，优化器等
  9. 得到输出结果以及评判指标

  

  

  ## 项目文件

### ponkemon.py

- 获取数据集的标签及数据集的划分

```python
import torch
import os, glob
import random, csv
from torch.utils.data import Dataset,DataLoader  # 继承母类

from torchvision import transforms
from PIL import Image


class Pokemon(Dataset):
    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()

        self.root = root  # root:当前data路径
        self.resize = resize  # resize:网络适宜的图片大小
        # 给每一种图片映射到字典中 name文件夹名称，下标即label，编码后不能改变
        self.name2label = {}  # "sq...": 0   即{"name":下标,...} 跟文件夹存储的循序的目录一样

        for name in sorted(os.listdir(os.path.join(root))):
            # os.listdir取根目录的文件和文件夹名称（含后缀），返回名称list，

            if not os.path.isdir(os.path.join(root, name)):  # 过滤文件
                # 经过.os.path.join()操作之后，会将(两个)路径进行拼接
                # os.path.isdir()：判断输入路径是否为目录/或者为文件夹
                # is->jump,not->continue

                continue
            # name文件名:key label下标:value
            self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label)

        # [[images] , [labels]] 加载文件csv
        self.images, self.labels = self.load_csv('images.csv')

        # 裁剪数据:[[images] , [labels]]
        if mode == 'train':  # self.images:60%[images]
            self.images = self.images[:int(0.6 * len(self.images))]  # 从数据集取0-60%为训练集
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':  # 20% <= 60%->80% # 从数据集取60%-80%为验证集
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:  # 20%  从数据集取80%-100%为测试集
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    # 加载图片 filename 即'images.csv'
    def load_csv(self, filename):

        # 如果csv(图片)不存在就创建
        if not os.path.exists(os.path.join(self.root, filename)):
            # 'pokemon\\mewetwo\\00001.png
            # 将文件以上述格式保存在csv文件内(原图片文件的格式是乱的，要规范化)
            images = []  # 存入images list
            for name in self.name2label.keys():
                # name：'pokemon\\mewetwo\\00001.png
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*,jpeg'))


            print(len(images), images)  # 1167, 'pokemon\\bulbasaur\\00000000.png'

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)  # 打开images.csv（没有就创建），写入
                for img in images:
                    # img:'pokemon\\bulbasaur\\00000000.png'
                    # os.sep可以将string分隔开
                    name = img.split(os.sep)[-2]  # 取第二个string:bulbasaur
                    label = self.name2label[name]  # 取字典键bulbasaur的下标为label
                    # 在csv存储格式： 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, label])  # 写入csv
                print('writen into file: ', filename)

        # read from csv file
        images, labels = [], []  # 从csv读出图片path和标签分别存入images和labels列表
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon\\bulbasaur\\00000000.png', 0
                img, label = row  # 每一行
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)  # 让数据和label数组长度一样
        return images, labels

    # 返回数据的数量
    def __len__(self):
        return len(self.images)  # 返回的是被裁剪之后的关系

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]  # RGB channel 上的均值
        std = [0.229, 0.224, 0.225]  # RGB channel 上的方差
        # x_hat = (x-mean)/std
        # x = x_hat*std + mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1] 使其满足broadcast
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_hat * std +mean
        return x



    # 返回idx的data和当前图片的label
    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'pokemen\\bulbasaur\\00000000.png'
        # label: 0
        img, label = self.images[idx], self.labels[idx]

        '''tf功能：把path string类型变成指定的数据类型'''
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path -> image data
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),  # 固定形状numpy
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),  # 转为tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # Image.open(x)加载图片，

        '''把[path,label]变成tensor的int类型'''
        img = tf(img)  # tensor类型的image data
        label = torch.tensor(label)

        return img, label  # 调用这个类最终会返回已处理好的tensor图片数据集label


def main():
    import visdom
    import time
    import torchvision
    viz = visdom.Visdom()  # visdom对象


    db = Pokemon(r'pokeman', 224, 'train')
    x, y = next(iter(db))
    print('sample:', x.shape, y.shape, y)


    # 图片比较工整（如当前目录下的pokeman文件夹）
    # 即: 数据集文件夹->对象文件夹->图片  才适合用以下方法
    # tf = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    #
    # db = torchvision.datasets.ImageFolder(root='pokeman', transform=tf)
    # loader = DataLoader(db, batch_size=32, shuffle=True)
    #
    # print(db.class_to_idx)
    #
    # for x, y in loader:
    #     viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch_y'))
    #
    #     time.sleep(10)




    # sample: torch.Size([3, 224, 224]) torch.Size([]) tensor(2)

    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    # viz.image显示图片

    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=2)
    # num_workers=8 一次取8张图片（8线程） 太大了，改为2


    for x, y in loader:
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch_y'))
        # viz.text显示文本  nrow=8一行8张

        time.sleep(10)

if __name__ == '__main__':
    main()


# 主函数测试
# db = Pokemon(r'pokeman', 224, 'train')
# i = 0
# for x, y in db:
#     i += 1
#     #
#     print('sample:', x.shape, y.shape, y)
#     if i == 3:
#         break
#
# sample: torch.Size([3, 224, 224])
# torch.Size([])
# tensor(2)
#
# sample: torch.Size([3, 224, 224])
# torch.Size([])
# tensor(2)
#
# sample: torch.Size([3, 224, 224])
# torch.Size([])
# tensor(3)
```





### resnet3.py

- 构建resnet

```python
import torch
from torch import nn
from torch.nn import functional as F


class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """

        :param ch_in:
        :param ch_out:
        :param stride: 1
        """
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)


        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """

        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut
        # extra module: [b, ch_in, h, w] => [b, ch_ut, h, w]
        # slement-wise add:
        out = self.extra(x) + out
        out = F.relu(out)  # [2, 128, 224, 224]

        return out


class Resnet18(nn.Module):
    def __init__(self, num_class):
        super(Resnet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(16)
        )


        # follow 4 block
        # [b, 16, h, w] => [b, 32, h, w]
        self.blk1 = ResBlk(16, 32, stride=3)
        # [b, 32, h, w] => [b, 128, h, w]
        self.blk2 = ResBlk(32, 64, stride=3)
        # [b, 64, h, w] => [b, 128, h, w]
        self.blk3 = ResBlk(64, 128, stride=2)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk4 = ResBlk(128, 256, stride=2)

        # [b, 256, 3, 3]
        self.outlayer = nn.Linear(256*3*3, num_class)  # num_class 共5类


    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))

        # [b, 3, h, w] => [b, 256, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print(x.shape)
        # [b, 256, 3, 3]
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)


        return x



def main():
    blk = ResBlk(64, 128)
    tmp = torch.randn(2, 64, 224, 224)
    out = blk(tmp)
    print('block:', out.shape)


    model = Resnet18(5)
    tmp = torch.randn(2, 3, 224, 224)
    out = model(tmp)
    print('resnet:', out.shape)

    p = sum(map(lambda p:p.numel(), model.parameters()))
    # lambda叫做匿名函数，是一种不需要提前对函数进行定义再使用的情况下就可以使用的函数
    # 定义规则：冒号的左边是原函数的参数，右边是原函数的返回值。
    # 函数map 其功能：将右边的参数传入左边的函数得到该函数的函数的返回值
    # model.parameters()网络参数 p.numel()参数所占内存大小 sum起来 = p 总参数量内存
    print('perameter size:', p)



if __name__ == '__main__':
    main()
```





utils1.py

```python
import torch
import torch.nn as nn
from matplotlib import pyplot as plt



class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item
        return x.view(-1, shape)


def plot_image(img, label, name):

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])

    plt.show()
```



### train_scratch.py

- 训练数据集

```python
import torch
from torch import optim, nn
import visdom
import torchvision
from torch.utils.data import DataLoader
from utils1 import Flatten

from pokemon import Pokemon
# from resnet3 import Resnet18
from torchvision.models import resnet18
# Downloading: "https://download.pytorch.org/models/resnet18-5c106cde.pth"
# to C:\Users\USER/.cache\torch\hub\checkpoints\resnet18-5c106cde.pth
#  11%|█         | 4.76M/44.7M [01:05<09:08, 76.2kB/s]  resnet18要下载


batchsz = 32  # 可尝试设置大一点
lr = 1e-3
epochs = 10

device = torch.device('cuda')
torch.manual_seed(1234)  # 随机种子，更好复现

train_db = Pokemon(r'pokeman', 224, mode='train')
val_db = Pokemon('pokeman', 224, mode='val')
test_db = Pokemon('pokeman', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True
                          )
val_loader = DataLoader(val_db, batch_size=batchsz)
test_loader = DataLoader(test_db, batch_size=batchsz)




viz = visdom.Visdom()

def evalute(model, loader):
    model.eval()

    correct = 0  # 正确的数量
    total = len(loader.dataset)  # 总测试数量
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, y,).sum().float().item()

    return correct / total



def main():

    # model = Resnet18(5).to(device)
    train_model = resnet18(pretrained=True)
    # 迁移学习，利用公共的训练好的模型，训练自己的data
    model = nn.Sequential(*list(train_model.children())[:-1],
                          Flatten(),
                          nn.Linear(512, 5)
                          ).to(device)
    x = torch.randn(2, 3, 224, 224)
    print(model(x).shape)


    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Adam优化器，SGD不能优化的Adam可以很好优化，一般采用Adam优化
    criteon = nn.CrossEntropyLoss()


    best_acc, batch_eponch = 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):
            # x:[b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)

            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            # pytroch nn.CrossEntropyLoss()内部已做one hot 传标签即可

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if epoch % 2 == 0:

            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), 'best.mdl')
                viz.line([val_acc], [global_step], win='val_acc', update='append')



    print('best acc:', best_acc, 'best epoch:', best_epoch)
    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)



if __name__ == '__main__':
    main()
```









