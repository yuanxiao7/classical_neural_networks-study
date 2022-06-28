## 网课笔记整理
![image](https://user-images.githubusercontent.com/93062146/176079796-82886a5d-ee9d-495e-b10d-5f525f1986cc.png)


### [torch](https://so.csdn.net/so/search?q=torch&spm=1001.2101.3001.7020).normal()

```sql
torch.normal(means, std, out=None)
```

返回一个张量，包含从给定参数`means`,`std`的[离散](https://so.csdn.net/so/search?q=离散&spm=1001.2101.3001.7020)正态分布中抽取随机数。 均值`means`是一个张量，包含每个输出元素相关的[正态分布](https://so.csdn.net/so/search?q=正态分布&spm=1001.2101.3001.7020)的均值。 `std`是一个张量，包含每个输出元素相关的正态分布的标准差。 均值和标准差的形状不须匹配，但每个张量的元素个数须相同。

参数:

- means (Tensor) – 均值
- std (Tensor) – 标准差
- out (Tensor) – 可选的输出张量





### torch.take()

```python
torch.take(input, index)->Tensor
```

返回一个新的张量，其中的元素是输入元素在给定的索引处，将输入张量视为视为一维张量。结果[tensor](https://so.csdn.net/so/search?q=tensor&spm=1001.2101.3001.7020)的形状与索引相同。

提取特定索引像素

参数介绍：

- `input`：输入tensor。
- `indices`：索引



### 限定数值范围

使数据起伏没有这么大

```python
#37. //////  clamp gradient clipping 将数据限制在一定的范围
#float clamp(float minnumber, float maxnumber, float parameter)
grad=torch.rand(2,3)*15
print(grad)
print(grad.max())
print(grad.min())
print(grad.median())
print(grad.clamp(10))   #最小值为10
print(grad.clamp(0,10))  #最小值0，最大值10

a=torch.randn(3,4)  #打平后再取
print(a)
x=a.ge(0.5)#ge表示great equal #取出a这个Tensor中大于0.5的元素
print(x)
print(torch.masked_select(a, x))  #masked_select取出大于0.5的概率

像素提取，将复杂网络简单化，减少参数 drop
```



### 数据融合与恢复

- view / reshape lost dim information
- 当你根据网络结构改变数据集的shape时，要注意改变后要恢复原来数据对应的数值，不然会造成数据污染，直接废掉

```python
#25. //////  transpose函数，转置
# print(b.t())  #只适用于矩阵
a=torch.rand(4,3,32,32)
a1=a.transpose(1,3).contiguous().view(4,3*32*32).view(4,3,32,32)#破坏原数据
# 此处将各维度的数据调换，融合，恢复时，理应先恢复为调换后的维度，再回复调换浅的维度
print(a1.shape)
# 正确展示
a2=a.transpose(1,3).contiguous().view(4,3*32*32).view(4,32,32,3).transpose(1,3)
print(a2.shape)  #correct

#26. //////  permute
a=torch.rand(4,3,14,28)
print('a: ',a.shape)
a1=a.transpose(1,3)
print(a1.shape)
a2=a1.transpose(1,2)
print(a2.shape)
a3=a.permute(0,2,3,1)  #根据需求按照原来的索引转置，参数为原来的索引
# 用原数据的索引，放到想要改为的索引位置上，得到想要的维度
print(a3.shape)
```

### 数据扩维

```python
a = torch.rand(32)
b = torch.rand(4, 3, 14, 14)
a = a.unsqueeze(1).unsqueeze(2).unsqueeze(0)  # 先从小维扩展
print(a.shape)

 a=torch.tensor([1.2, 2.3])
 print(a.unsqueeze(-1))  #负 给内维加一维
 print(a.unsqueeze(0))   #非负 给外围加一维

# result:
# tensor([[1.2000],
#         [2.3000]])
# tensor([[1.2000, 2.3000]])

非负的在前加一维，负的在后加一维
a=torch.rand(4,3,28,28)
print(a.shape)
print(a.unsqueeze(0).shape)#在索引为0的维度之前加一维
print(a.unsqueeze(-1).shape) #在索引为-1的后面加一维
print(a.unsqueeze(4).shape)  #在索引为4的前面加一维(效果同上) 即在第四维添加一维，原第四维等一致往后挪
print(a.unsqueeze(-5).shape) #在索引为-5的后面加一维

# result:
# torch.Size([4, 3, 28, 28])
# torch.Size([1, 4, 3, 28, 28])
# torch.Size([4, 3, 28, 28, 1])
# torch.Size([4, 3, 28, 28, 1])
# torch.Size([1, 4, 3, 28, 28])
```





### 数据减维

```python
#  22. //////  squeeze  将只有一个element的dimension挤压掉
b = torch.rand(1, 32, 1, 1)
print(b.squeeze().shape)
print(b.squeeze(0).shape)
print(b.squeeze(-1).shape)
print(b.squeeze(1).shape)  # dim1有elements，不能挤压，保持不变
print(b.squeeze(-4).shape)
```





### 对应数据增减维

```python
#23. //////  expand/expand_as
a=torch.rand(4,32,14,14)
b=torch.rand(1,32,1,1)
print(a.shape)
print(b.shape)
c=b.expand(4,32,14,14).shape  #只有1才能扩张到对应的数，32扩张到32，same
print(c)
d=b.expand(-1,32,-1,14).shape  #-1是保持不变，根据需求扩张即可
print(d)
```



### repeat（）

画图展示时可以用到

```python
#24. //////  repeat()函数括号内为重复的次数
b=torch.rand(1,32,1,1)
print(b.shape)
print(b.repeat(4,32,1,1).shape)#对应维度分别重复4,32,1,1次
print(b.repeat(4,1,1,1).shape)
print(b.repeat(4,1,32,32).shape)
```



### torch.gather()

话说，第一次学时觉得还行（X  因为只看有规律的）

**此处笔记出自于**：  https://zhuanlan.zhihu.com/p/352877584

我们找个3x3的二维矩阵做个实验

```python
import torch

tensor_0 = torch.arange(3, 12).view(3, 3)
print(tensor_0)
```

输出结果

```python
tensor([[ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])
```

2.1 输入行向量index，并替换行索引(dim=0)

```python
index = torch.tensor([[2, 1, 0]])
tensor_1 = tensor_0.gather(0, index)
print(tensor_1)
```

输出结果

```python
tensor([[9, 7, 5]])
```

![image-20220626212821725](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220626212821725.png)



2.2输入行向量index，并替换列索引(dim=1)

```python
index = torch.tensor([[2, 1, 0]])
tensor_1 = tensor_0.gather(1, index)
print(tensor_1)
```

输出结果

```python
tensor([[5, 4, 3]])
```

![image-20220626213109236](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220626213109236.png)





### 二范数

```python
a=torch.tensor([[[[1.,2],[1,1]],[[1,2],[2,1]]]]) #直接创建一个tensor
print(a)
print(a.shape)
print(a.norm(2,dim=0))  # z在指定的dim的element不只有一个时取范数
print(a.norm(2,dim=1))  # 对应位置去范数
print(a.norm(2,dim=2))  
print(a.norm(2,dim=3))  # 中括号内取范数


result:
tensor([[[[1., 2.],
          [1., 1.]],
          [[1., 2.],
          [2., 1.]]]])
torch.Size([1, 2, 2, 2])
tensor([[[1., 2.],
         [1., 1.]],
         [[1., 2.],
         [2., 1.]]])
tensor([[[1.4142, 2.8284],
         [2.2361, 1.4142]]])
tensor([[[1.4142, 2.2361],
         [2.2361, 2.2361]]])
tensor([[[2.2361, 1.4142],
         [2.2361, 2.2361]]])
```



### 计算图求导

![image-20220627202639797](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220627202639797.png)







sir



