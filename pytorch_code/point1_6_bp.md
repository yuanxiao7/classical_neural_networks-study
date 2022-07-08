# 感知机

- **class notes**

## 项目简介

- 这个md文件是一个手写数字识别，只用numpy实现，没有用其他任何的框架，想要深入了解bp神经网络（感知机）的工作原理小伙伴可以细品，这是一个比较完整的小demo，数据集的加载，分割，处理，传入网络，加激活函数，正则化，加优化器，输出结果，得到评判值。个人建议，先理解原理，在自己手写推一遍，然后再用代码实现出来，虽然看起来很简单，但真这么做，还是有一点点难度的，加油噢！

full_connect_numpy.py

```python
import numpy as np
import random


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)(1-sigmoid(z))



class MLP_np:
    def __init__(self, sizes):
        """

        :param sizes: [784, 30, 10]
        """
        self.sizes = sizes
        self.num_layers = len(sizes) - 1  # tow layers(not include input)

        # w: [ch_out, ch_in]
        # b: [ch_out]
        # zip 迭代器 迭代对象[]
        self.weights = [np.random.randn(ch2, ch1) for ch1, ch2 in zip(sizes[:-1], sizes[1:])]
        # [[30, 784], [10, 30]] sizes[:-1]: 784 30最后一个不迭代 sizes[1:]:30 10第一个不迭代
        self.biases = [np.random.randn(ch, 1) for ch in sizes[1:]]
        # [[30, 1], [10, 1]]

    def forward(self, x):  # 单独的forward利于后面测试
        """

        :param x: [784, 1]
        :return: [10, 1]
        """
        for b, w in zip(self.biases, self.weights):
            # w [30, 784]@[784, 1]=>[30, 1]+[30, 1]=>[30, 1] first
            z = np.dot(w, x) + b
            # [30, 1]
            x = sigmoid(z)
        return x


    def backprop(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # save activaion for every layer ,later
        activations = [x]
        # save z for every layer  ,previous
        zs = []
        activation = x

        # 1. forward
        # backward有forward为了计算梯度
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)

            zs.append(z)
            activations.append(activation)

        loss = np.power(activations[-1] - y, 2)  # 均方差

        # 2. backward
        # 2.1 compute gradient on output layer
        # [10, 1] with [10, 1] => [10, 1]  o=sigmoid(z)  z:last output
        # L对最后一层输出z求导 delta=(o-y)*o*(1-o)  (用均方差求导L=1/2*(o-y)^2)
        delta = activations[-1] * (1-activations[-1]) * (activations[-1] - y)

        # [10, 1]@ [1, 30]=> [10, 30]
        # t=activation[-2]: [30, 1]
        # L对W1求导:nabla_w[-1]=(o-y)*o*(1-o)*t
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # L对b1求导
        nabla_b[-1] = delta

        # 每反向传播一次（层），都利用上一次的梯度，大大减小计算量



        # 2.2 compute hidden gradiant
        # num_layers: add up 2 layers
        for l in range(2, self.num_layers+1):  # 从后往前，由倒数第二层的weight计算梯度
            l = -l  
            z = zs[l]  # 输出(线性)
            a = activations[l]  # 激活值(非线性)

            # delta_j
            # [10, 30]T @ [10, 1] => [30, 10]@[10, 1]=>[30, 1]*[30, 1]
            # L对隐藏层输出z求导
            delta = np.dot(self.weights[l+1].T, delta) * a * (1-a)


            # [30, 1] @ [784, 1].T => [30, 784]
            # L对W0求导（同上）
            nabla_w[l] = np.dot(delta, activations[l-1].T)
            # L对b0求导
            nabla_b[l] = delta

        return nabla_w, nabla_b, loss
        # d_[[w0],[w1]], d_[[b0],[b1]]

    def train(self, training_data, epochs, batchsz, lr, test_data):
        """

        :param training_data: list of (x,y)
        :param epochs:  1000
        :param batchsz: 10
        :param lr: 0.01
        :param test_data: list of (x,y)
        :return:
        """
        if test_data:
            n_test = len(test_data)  # 数据存在就取其长度

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+batchsz] for k in range(0, n, batchsz)]
            # 按batchsz切割每一段数据，如[[x,y],[x,y],...,[x,y],[x,y]]
            # batchsz为2，则每一个batch切成[[x,y],[x,y]]的list

            # for every batch in current catch
            for mini_batch in mini_batches:
                self.updata_mini_batch(mini_batch, lr)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
                print("Epoch {0} complete".format(j))


    def updata_mini_batch(self, batch, lr):
        """

        :param batch: list of (x, y)
        :param lr: 0.01
        :return:
        """
        # 记录最后一个batch的梯度nabla,初始为零
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        loss = 0

        # for every sample in current batch
        for x, y in batch:
            # list of every W/b gradiant
            # [w1, w2, w3]
            nabla_w_, nabla_b_, loss_ = self.backprop(x, y)
            nabla_w = [accu+cur for accu, cur in zip(nabla_w, nabla_w_)]  # 之前的和现在的
            nabla_b = [accu+cur for accu, cur in zip(nabla_b, nabla_b_)]
            loss += loss_

        # 取一个batch的平均nabla并记录  计算每一个对应元素相连的梯度之和再除于总数得到平均梯度
        nabla_w = [w/len(batch) for w in nabla_w]
        nabla_b = [b/len(batch) for b in nabla_b]
        loss = loss / len(batch)

        # w = w - lr*nabla_w 每经过一个batch更新一次weight and bias
        self.weights = [w - lr * nabla for w, nabla in zip(self.weights, nabla_w)]
        self.biases = [b - lr * nabla for b, nabla in zip(self.biases, nabla_b)]



    def evaluate(self, test_data):
        """

        :param test_data: list of (x, y)
        :return:
        """
        # 取预测的predict_max_idx和y存在一个list里 即[pred,y]
        result = [(np.argmax(self.forward(x)), y) for x, y in test_data]
        correct = sum(int(pred == y) for pred, y in result)

        return correct


def main():
    import mnist_loader
    # loading the MNIST data   # list: [[picture,label],...,[picture,label]]
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(len(training_data), training_data[0][0].shape, training_data[0][1].shape)
    print(len(test_data), test_data[0][0].shape, test_data[0][1].shape)
    # Set up a Network with 30 hidden neurons
    net = MLP_np([784, 30, 10])


    net.train(training_data, 1, 10, 0.1, test_data=test_data)


if __name__ == '__main__':
    main()
```



mnist_loader.py

```python
import pickle
import gzip
import numpy as np


def load_data():
    file = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(file, encoding='latin1')
    file.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    # tr_d:[[[28, 28], [28, 28], ..., [28, 28], [28, 28]], [1, 2, 4, ..., 9, 2, 0]] list:[pictures，labels]
    # 训练集
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]  # x为reshape的对象，(784, 1)想要reshape的形状
    training_labels = [vectorized_label(y) for y in tr_d[1]]  # label's shape:(10, 1)
    training_data = list(zip(training_inputs, training_labels))  # list: [[picture,label],...,[picture,label]]

    # 验证集
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    # 测试集
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)


def vectorized_label(j):
    e = np.zeros((10, 1))  # e=[0,0,0,0,0,0,0,0,0,0]
    e[j] = 1.0  # stick label
    # 如: j=5 => e=[0.0 ,0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    return e
```