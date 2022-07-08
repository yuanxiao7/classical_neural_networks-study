# RNN

- **class notes**

## 项目简介

- 这是个md文件时是用循环神经网络写的简单的句子处理和正弦函数的预测，由于小编我目前主要是走cv方向的，用卷积会更多一些，这里的rnn我也只是，了解原理，敲了一下代码，跑了一下原理而已，rnn主要是由全连接组成的，很明显的的一个特点，即有一个 “记录装置” 记录前面的内容，和每一次输入的信息整和得到一个输出，且把继续这个输出作为下一个的输入，直到没有信息输入。最后将每一次信息输入得到的输出整合到一起，得到最终预测结果。

## 项目文件

### rnn.py   

- 这个代码，我只是把他跑起来而已，因为数据集有点大，下载比较久，所以我就没有跑训练 ,嘻 ~ q_p

```python
import torch
import numpy as np
from torch import nn

#1.//////  词向量表
word_to_ix = {"hello": 0, "world": 1}  # 取其索引0
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
print(lookup_tensor)  # 取词典中的词对应的idx


embeds = nn.Embedding(2, 5)  # 固定(向量)词表（实例化），其中2为词典的len，5为自定义的词向量维
hello_embed = embeds(lookup_tensor)  # 利用索引在词向量表中找出hello对应的词向量
print(hello_embed)
#直接查表无法优化，用其他表达方式可以优化

# the result: # 查表时得到对应的向量，很randn
# tensor([[-0.7494,  0.9260, -0.4104,  0.4982,  0.9014]],
#        grad_fn=<EmbeddingBackward>)



#2. //////  网络层（并行处理）
rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=1)  # 只有一层
print(rnn)
x = torch.randn(10, 3, 100)  # 每一次送10句话，记录h: 1层，3个句子，shape 20
out, h = rnn(x, torch.zeros((1, 3, 20)))  # out为整合的的h
print(out.shape, h.shape)

# result:
# RNN(100, 20)
# torch.Size([10, 3, 20]) torch.Size([1, 3, 20])


rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=2)  # 两层
print(rnn._parameters.keys())

# result:
# odict_keys(['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0',
# 'weight_ih_l1', 'weight_hh_l1', 'bias_ih_l1', 'bias_hh_l1'])



#3. //////  input dim,hidden dim
rnn = nn.RNN(100, 10, 2)  # 相当于两层全连接层

# contain:
# odict_keys(['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0',
# 'weight_ih_l1', 'weight_hh_l1', 'bias_ih_l1', 'bias_hh_l1'])
# torch.Size([10, 100]) torch.Size([10, 10])
# torch.Size([10]) torch.Size([10])
# torch.Size([10, 10]) torch.Size([10, 10])
# torch.Size([10]) torch.Size([10])


rnn = nn.RNN(100, 10, num_layers=2)
print(rnn._parameters.keys())
print(rnn.weight_hh_l0.shape, rnn.weight_ih_l0.shape)
print(rnn.weight_hh_l1.shape, rnn.weight_ih_l1.shape)

# reslut:
# odict_keys(['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0', 'weight_ih_l1', 'weight_hh_l1', 'bias_ih_l1', 'bias_hh_l1'])
# torch.Size([10, 10]) torch.Size([10, 100])
# torch.Size([10, 10]) torch.Size([10, 10])


rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=4)  #4层
print(rnn)
x = torch.randn(10, 3, 100)
out, h = rnn(x)
print(out.shape, h.shape)

# result:
# RNN(100, 20, num_layers=4)
# torch.Size([10, 3, 20]) torch.Size([4, 3, 20])



# 3. //////  每一个cell为一层
x = torch.randn(10, 3, 100)
print(x.shape)

cell1 = nn.RNNCell(100, 20)
h1 = torch.zeros(3, 20)
for xt in x:
    h1 = cell1(xt, h1)

print(h1.shape)

# result:
# torch.Size([10, 3, 100])
# torch.Size([3, 20])


x = torch.randn(10, 3, 100)  # 每一句话10个字，每一个字为一个时刻，hi为每一时刻最后的输出
cell1 = nn.RNNCell(100, 30)  # 两个cell，时刻为两层全连接
cell2 = nn.RNNCell(30, 20)
h1 = torch.zeros(3, 30)
h2 = torch.zeros(3, 20)
for xt in x:
    h1 = cell1(xt, h1)
    h2 = cell2(h1, h2)

print(h2.shape)
# the result:
# torch.Size([3, 20])
```





### RNN实现正弦函数的简单预测

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt


num_time_steps = 50
input_size = 1
hidden_size = 16
output_size = 1
lr = 0.01


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,  # 数值1
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True  # 即使用b在前，[b, seq_len,  feature]
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size)  #此处（16， 1）

    def forward(self, x, hidden_prev):  # x， h0
        out, hidden_prev = self.rnn(x, hidden_prev)
        # out=[b, seq, h] => [seq, h]
        # seq为输入时刻点(一个b有10个, b=1(每次送一条曲线)),hid为记录
        # hid=[b, 1, h]  1为网络层数，h为记录长度shape
        out = out.view(-1, hidden_size)  # 打平[seq*h]=> [seq,h]
        # 打平后，变成2维，第一维的elements-num=打平/hidden_size
        # 第二维有hidden_size个elements
        out = self.linear(out)  # [seq, h]=>[seq, 1]
        out = out.unsqueeze(dim=0)  # 插入维度与y匹配 y=[b, seq, 1]  calculate loss
        return out, hidden_prev

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)  # Adam优化

hidden_prev = torch.zeros(1, 1, hidden_size)  # [b, 1, 16]

for iter in range(6000):
    start = np.random.randint(3, size=1)[0]  # 随机返回一个一维tensor
    time_steps = np.linspace(start, start + 10, num_time_steps)  # time_steps代表所画的点
    # start为任意一点，即随机从某一点start，在区间[start,start+10]生成50个点，即50个x1（y1=sinx1）
    data = np.sin(time_steps)  # 通过np.sin生成对应曲线，得到x1对应的y1
    data = data.reshape(num_time_steps, 1)
    # 将部分y1喂给网络，让他预测一个点
    x = torch.tensor(data[: -1]).float().view(1, num_time_steps - 1, 1)
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

    output, hidden_prev = model(x, hidden_prev)
    hidden_prev = hidden_prev.detach()  # .detach()使记录h没有grad，不参与反向传播

    loss = criterion(output, y)
    model.zero_grad()
    #model为单层网络，一个时刻点，每次使用都要清除上一次的model，因为它是add
    loss.backward()
    # for p in model.parameter()
    # print(p.grad.norm())
    # torch.nn.utils.clip_grad_norm_(p, 10)
    optimizer.step()

    if iter % 100 == 0:
        print("Iteration: {}, loss {}".format(iter, loss.item()))


start = np.random.randint(3, size=1)[0]
time_steps = np.linspace(start, start + 10, num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)  # 0-49
y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)   # 1-50

predictions = []
input = x[:, 0, :]  # 随机给一个点
for _ in range(x.shape[1]):
    input = input.view(1, 1, 1)  # 输入x1
    (pred, hidden_prev) = model(input, hidden_prev)
    input = pred  # 将预测作为下一个的输入，不断重复，并搜集每一次的预测，使他根据预测画出曲线
    predictions.append(pred.detach().numpy().ravel()[0])
    # 用ravel()方法将数组pred拉成一维数组，在取索引0的element

x = x.data.numpy().ravel()
y = y.data.numpy()
plt.scatter(time_steps[: -1], x.ravel(), s=90)  # time_steps[: -1]sin曲线描点，x轴0-10，y轴曲线的y1的值域
plt.plot(time_steps[:-1], x.ravel())  # sin曲线
plt.scatter(time_steps[1:], predictions)
plt.show()
```