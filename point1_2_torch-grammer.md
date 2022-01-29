# pytorch语法及相关函数

### Klass notes

```python
import torch
import numpy as np

#1. ///////  创建tensor
a=torch.tensor(1.)  
print(a)
b=torch.tensor(1.3)
print(b)
print(a.shape)
print(len(a.shape))

# result:
# tensor(1.)  标量 dim=0
# tensor(1.3000)
# torch.Size([])   #一个空的list类型的size
# 0  dim=0

a=torch.tensor([1.1])
print(a)
b=torch.tensor([1.1, 2.2])
print(b)
c=torch.FloatTensor(1)  #随机生成一个dim=1的tensor
print(c)
a1=torch.FloatTensor(2)   #随机生成两个dim=2的tensor
print(a1)
data=np.ones(2)
print(data)
print(torch.from_numpy(data))

# #result
# tensor([1.1000])
# tensor([1.1000, 2.2000])
# tensor([1.8638e-34])
# tensor([1.8582e-34, 0.0000e+00])
# [1. 1.]
# tensor([1., 1.], dtype=torch.float64)



#2. //////  shape size
a=torch.randn(2,3)
print(a)
print(a.shape)
print(a.size(0))
print(a.shape[0])
print(a.size(1))
#size(0),shape[0]算第一维的length

# result:
# tensor([[ 0.7493, -1.2719, -0.8188],
#         [ 0.0652, -0.6947, -0.5670]])
# torch.Size([2, 3])
# 2
# 2
# 3



#3. //////  dimension
a=torch.randn(1,2,3)  #一个三维的tensor，dim1有一个element
print(a)              #dim2有2个element，dim3有3个element
print(a.shape)
print(a[0])    #打印第一维的数据
print(list(a.shape))

# result:
# tensor([[[ 1.3437,  2.2784, -0.3958],
#          [-0.5032, -1.7742, -1.1548]]])
# torch.Size([1, 2, 3])
# tensor([[ 1.3437,  2.2784, -0.3958],
#         [-0.5032, -1.7742, -1.1548]])
# [1, 2, 3]



#4. //////  Mixed
a=torch.randn(2,3,28,28)
print(a)
print(a.shape)
print(a.numel())  #all data=2*3*28*28=4704   data of all elements
print(a.dim())
a1=torch.tensor(1)   #变量没有dimension
print(a1.dim())

# reusul:
# torch.Size([2, 3, 28, 28])
# 4704
# 4
# 0



#5. //////  从numpy导入data
a=np.array([2,3.3])
a1=torch.from_numpy(a)
print(a1)
b=np.array([1,2])
print(torch.from_numpy(b))
a11=np.ones([2,3])
print(torch.from_numpy(a11))

# result:
# tensor([2.0000, 3.3000], dtype=torch.float64)
# tensor([1, 2], dtype=torch.int32)
# tensor([[1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64)



#6. //////  tensor与Tensor
a=torch.empty(1)   #未初始化，未赋值，系统给的，数据很奇怪（不推荐使用）
print(a)
a1=torch.Tensor(2,3)
print(a1)
a2=torch.IntTensor(2,3)
print(a2)

# result：
# tensor([1.1696e+19])
# tensor([[0.0000e+00, 0.0000e+00, 2.1019e-44],
#         [0.0000e+00, 6.8074e+22, 4.9886e-43]])
# tensor([[1702521088,      10536,  912781403],
#         [     32762,          0,          0]], dtype=torch.int32)

a=torch.tensor([2,3.2])
print(a)
print(torch.FloatTensor([2,3.2]))
print(torch.tensor([[2.,3.2],[1.,22.3]]))  #给现成的数据
b=torch.FloatTensor(1,2)  #没有初始化，易出问题
print(b)   #随机生成shape为[1,2]的tensor
print(b.shape)

# result:
# tensor([2.0000, 3.2000])
# tensor([2.0000, 3.2000])
# tensor([[ 2.0000,  3.2000],
#         [ 1.0000, 22.3000]])
# tensor([[2.0000, 3.2000]])
# torch.Size([1, 2])



#7.//////  默认数据类型FloatTensor

a=torch.tensor([1.2,3])
print(a.type())

# result:
# torch.DoubleTensor

torch.set_default_tensor_type(torch.DoubleTensor)#将数据默认类型设置为double类型
a1=torch.tensor([1.2,3])
print(a1.type())



#8. //////  rand/rand_like,randint
a=torch.rand(3,3)   #在0-1赋值
print(a)
b=torch.rand_like(a)#将a的shape读出来重新赋值
print(b)
c=torch.randint(1,10,[3,3])#随机赋值，格式（min，max，shape）
print(c)

# result:
# tensor([[0.0681, 0.4994, 0.6367],
#         [0.2846, 0.8591, 0.1015],
#         [0.6617, 0.9340, 0.0835]])
# tensor([[0.0299, 0.4987, 0.2112],
#         [0.6154, 0.3965, 0.0504],
#         [0.2901, 0.9313, 0.5074]])
# tensor([[2, 9, 5],
#         [9, 3, 7],
#         [7, 1, 5]])



#9. //////  randn
a=torch.randn(3,3) #均值为0，方差为1的正态分布赋值
print(a)
a1=torch.normal(mean=torch.full([10],0.), std=torch.arange(1,0.,-0.1))
print(a1)       #1个element=0,num=10的向量      方差由1->0减小，步长=-0.1

#result:
# tensor([[-0.0114, -0.2730, -0.5971],
#         [-0.4871,  1.7506, -0.7240],
#         [-1.3734,  0.1394,  0.8122]])
# tensor([-2.0571, -0.7358,  0.0333,  0.9240, -1.2725, -0.3112,  0.2110,  0.1629, 0.2057, -0.0117])  #使用normal会将维度打平为一维



#10. //////  full函数
a=torch.full([2,3],7)
print(a)
b=torch.full([],4)  #一个tensor类型的标量 4
print(b)
c=torch.full([1],7)
print(c)

# result:
# tensor([[7, 7, 7],
#         [7, 7, 7]])
# tensor(4)
# tensor([7])



#11. //////  range/arange
print(torch.arange(0,10))  #10取不到
print(torch.arange(0,10,2))
# print(torch.range(0,10))不建议使用，UserWarning
print(torch.range(0, 10))

# result:
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# tensor([0, 2, 4, 6, 8])
# tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
# F:/python-code/pythonProject-holiday/practice2.py:257: UserWarning:
# torch.range is deprecated and will be removed in a future release
# because its behavior is inconsistent with Python's range builtin.
# Instead, use torch.arange, which produces values in [start, end).



#12. //////  linspace/logspace
a=torch.linspace(0,10,steps=4) #闭区间[0,10]均可取,等差数列
print(a)
b=torch.linspace(0,10,steps=11)
print(b)
c=torch.logspace(0,-1,steps=10)  #由10的0次方(1.0000)降到10的-1次方(0.1000)
print(c)
e=torch.logspace(0,1,steps=10)
print(e)

# result:
# tensor([ 0.0000,  3.3333,  6.6667, 10.0000])
# tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
# tensor([1.0000, 0.7743, 0.5995, 0.4642, 0.3594, 0.2783, 0.2154, 0.1668, 0.1292,
#         0.1000])
# tensor([ 1.0000,  1.2915,  1.6681,  2.1544,  2.7826,  3.5938,  4.6416,  5.9948,
#          7.7426, 10.0000])



#13. //////  once/zeros/eye

a=torch.ones(3,3)
print(a)
a1=torch.zeros(3,3)
print(a1)
a2=torch.eye(3,4)
print(a2)
a3=torch.eye(3,3)
print(a3)
print(torch.ones_like(a3))

# result:
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])
# tensor([[1., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 1., 0.]])
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])



#14. //////  ranperm 相当于shuffle
a=torch.rand(2,3)
b=torch.rand(2,2)
print(a)
print(b)
idx=torch.randperm(2)
print(idx)  #使用同一idx，维持数值对应
a1=a[idx]
b1=b[idx]
print(a1)
print(b1)
print(a1,b1)

#result1:
# tensor([[0.9019, 0.3327, 0.0483],
#         [0.3394, 0.3753, 0.9453]])
# tensor([[0.3178, 0.0572],
#         [0.7342, 0.9782]])
# tensor([1, 0])  打乱dim=1的date，将索引0和1的data互换
# tensor([[0.3394, 0.3753, 0.9453],
#         [0.9019, 0.3327, 0.0483]])
# tensor([[0.7342, 0.9782],
#         [0.3178, 0.0572]])
# tensor([[0.3394, 0.3753, 0.9453],
#         [0.9019, 0.3327, 0.0483]]) tensor([[0.7342, 0.9782],
#         [0.3178, 0.0572]])

#result2:
# tensor([[0.5539, 0.5646, 0.7869],
#         [0.6055, 0.0690, 0.1562]])
# tensor([[0.5401, 0.3666],
#         [0.9206, 0.9676]])
# tensor([1, 0])  打乱后索引不变，即不变(因数据过少，一般不会一样)
# tensor([[0.6055, 0.0690, 0.1562],
#         [0.5539, 0.5646, 0.7869]])
# tensor([[0.9206, 0.9676],
#         [0.5401, 0.3666]])
# tensor([[0.6055, 0.0690, 0.1562],
#         [0.5539, 0.5646, 0.7869]]) tensor([[0.9206, 0.9676],
#         [0.5401, 0.3666]])



#15.//////  indexing
a=torch.rand(4,3,28,28)  #即：(bath,channel,height,weight)
# print(a)
# print(a[0])去索引为零的data即第1张图片
print(a[0].shape)
#print(a[0,0]) 第1张图片索引为0的通道，即1通道
print(a[0,0].shape)
print(a[0,0,2,4]) #第一张照片第一通道的第二行第四列的一个数值(即像素点)

# result:
# torch.Size([3, 28, 28])
# torch.Size([28, 28])
# tensor(0.0565)



#16. //////  select first/last  N
a=torch.rand(4,3,28,28)
print(a.shape)
print(a[:2].shape)  #取dim=1索引0,1的照片
print(a[:2,:1,:,:].shape)  #取dim=1索引0,1照片的R通道
print(a[:2,1:,:,:].shape)   #取0,1照片的G,B通道
print(a[:2,-1:,:,:].shape)  #-1取最后一行，即取B通道
print(a[:,:,0:28:2,0:28:2].shape) #双冒号 start:end:step 索引end取不到
print(a[:,:,::2,::2].shape)  #作用同上

#select by specific index
print(a.index_select(0,torch.tensor([0,2])).shape)  #取dim1下标0,2的照片
print(a.index_select(1,torch.tensor([1,2])).shape)  #取dim2下标为1,2的通道
print(a.index_select(2,torch.arange(28)).shape)   #取dim3下标0-27，共28
print(a.index_select(2,torch.arange(8)).shape)   #取dim3下标0-7，共8

# result:
# torch.Size([4, 3, 28, 28])
# torch.Size([2, 3, 28, 28])
# torch.Size([2, 1, 28, 28])
# torch.Size([2, 2, 28, 28])
# torch.Size([2, 1, 28, 28])
# torch.Size([4, 3, 14, 14])
# torch.Size([4, 3, 14, 14])
# torch.Size([2, 3, 28, 28])
# torch.Size([4, 2, 28, 28])
# torch.Size([4, 3, 28, 28])
# torch.Size([4, 3, 8, 28])



#17. //////  ...的使用
a=torch.rand(4,3,28,28)
print(a.shape)
print(a[...].shape)
print(a[0,...].shape)  #取第0张照片  等效a[0]
print(a[:,1,...].shape) #取所有照片的第1通道
print(a[...,:2].shape) #每张照片的每一通道，每一行只有两个元素(原来列的前2个数据)，28行2列

# result:
# torch.Size([4, 3, 28, 28])
# torch.Size([4, 3, 28, 28])
# torch.Size([3, 28, 28])
# torch.Size([4, 28, 28])
# torch.Size([4, 3, 28, 2])



#18. //////  select by mask
a=torch.randn(3,4)  #打平后再取
print(a)
x=a.ge(0.5)#ge表示great equal #取出a这个Tensor中大于0.5的元素
print(x)
print(torch.masked_select(a, x))  #masked_select取出大于0.5的概率

# result:
# tensor([[-2.1245, -0.6411, -0.1212,  0.5898],
#         [-0.3416, -0.6341,  0.0615, -1.1201],
#         [ 0.1828,  1.9103,  1.0873,  0.9475]])
# tensor([[False, False, False,  True],
#         [False, False, False, False],
#         [False,  True,  True,  True]])
# tensor([0.5898, 1.9103, 1.0873, 0.9475])


#19. //////  select by flatten index
src = torch.tensor([[4,3,5],[6,7,8]])
print(torch.take(src,torch.tensor([0,2,5])))

# result:
# 打平后再索引
# tensor([4, 5, 8])



#20. //////  view/reshape lost dim information
a=torch.rand(4,1,28,28)
print(a.shape)
b=a.view(4,28*28)
print(b)
print(b.shape)
print(a.view(4*28,28).shape)
print(a.view(4*1,28,28).shape)
print(a.view(4,784).view(4,28,28,1).shape) #logic error,data has had permute!!

# result:
# torch.Size([4, 1, 28, 28])
# tensor([[0.8042, 0.8082, 0.0096,  ..., 0.3913, 0.5276, 0.7103],
#         [0.7987, 0.8406, 0.0892,  ..., 0.9566, 0.2364, 0.5395],
#         [0.9243, 0.3281, 0.5856,  ..., 0.2182, 0.4554, 0.6917],
#         [0.9768, 0.3781, 0.6808,  ..., 0.7439, 0.6714, 0.5483]])
# torch.Size([4, 784])
# torch.Size([112, 28])
# torch.Size([4, 28, 28])
# torch.Size([4, 28, 28, 1])应该保持(b c h w) 对应的data



#21. //////  unsqueeze(position/index)
a=torch.rand(32)
b=torch.rand(4,3,14,14)
a=a.unsqueeze(1).unsqueeze(2).unsqueeze(0) #先从小维扩展
print(a.shape)

# result:
# torch.Size([1, 32, 1, 1])

 a=torch.tensor([1.2, 2.3])
 print(a.unsqueeze(-1))  #负 给内维加一维
 print(a.unsqueeze(0))   #非负 给外围加一维

# result:
# tensor([[1.2000],
#         [2.3000]])
# tensor([[1.2000, 2.3000]])

a=torch.rand(4,3,28,28)
print(a.shape)
print(a.unsqueeze(0).shape)#在索引为0的维度之前加一维
print(a.unsqueeze(-1).shape) #在索引为-1的后面加一维
print(a.unsqueeze(4).shape)  #在索引为4的前面加一维(效果同上)
print(a.unsqueeze(-5).shape) #在索引为-5的后面加一维

# result:
# torch.Size([4, 3, 28, 28])
# torch.Size([1, 4, 3, 28, 28])
# torch.Size([4, 3, 28, 28, 1])
# torch.Size([4, 3, 28, 28, 1])
# torch.Size([1, 4, 3, 28, 28])



#22. //////  squeeze  将只有一个element的dimension挤压掉
b=torch.rand(1,32,1,1)
print(b.squeeze().shape)
print(b.squeeze(0).shape)
print(b.squeeze(-1).shape)
print(b.squeeze(1).shape)  #dim1有elements，不能挤压，保持不变
print(b.squeeze(-4).shape)

# result:
# torch.Size([32])
# torch.Size([32, 1, 1])
# torch.Size([1, 32, 1])
# torch.Size([1, 32, 1, 1])
# torch.Size([32, 1, 1])



#23. //////  expand/expand_as
a=torch.rand(4,32,14,14)
b=torch.rand(1,32,1,1)
print(a.shape)
print(b.shape)
c=b.expand(4,32,14,14).shape  #只有1才能扩张到对应的数，32扩张到32，same
print(c)
d=b.expand(-1,32,-1,-1).shape  #-1是保持不变，根据需求扩张即可
print(d)

# result:
# torch.Size([4, 32, 14, 14])
# torch.Size([1, 32, 1, 1])
# torch.Size([4, 32, 14, 14])
# torch.Size([1, 32, 1, 1])



#24. //////  repeat()函数括号内为重复的次数
b=torch.rand(1,32,1,1)
print(b.shape)
print(b.repeat(4,32,1,1).shape)#对应维度分别重复4,32,1,1次
print(b.repeat(4,1,1,1).shape)
print(b.repeat(4,1,32,32).shape)

# result:
# torch.Size([1, 32, 1, 1])
# torch.Size([4, 1024, 1, 1])
# torch.Size([4, 32, 1, 1])
# torch.Size([4, 32, 32, 32])



#25. //////  transpose函数，转置
# print(b.t())  #只适用于矩阵
a=torch.rand(4,3,32,32)
a1=a.transpose(1,3).contiguous().view(4,3*32*32).view(4,3,32,32)#破坏原数据
print(a1.shape)
a2=a.transpose(1,3).contiguous().view(4,3*32*32).view(4,32,32,3).transpose(1,3)
print(a2.shape)  #correct

# result:
# torch.Size([4, 3, 32, 32])#虽然形状相同，但是原来的数据改变了，造成了数据污染
# torch.Size([4, 3, 32, 32])

# 正确做法：
b=torch.rand(1,4,2)
b1=b.transpose(1,2).contiguous()
b2=b1.view(1,4*2)
b3=b2.view(1,2,4)
b4=b3.transpose(1,2)  #正确做法，按照原来的索引对应的数据恢复，才是原数据，才不会造成数据污染
print(b)
print(b1)
print(b2)
print(b3)
print(b4)

#result:
# tensor([[[0.2525, 0.3949],
#          [0.2931, 0.7346],
#          [0.3893, 0.5196],
#          [0.3330, 0.6585]]])
# tensor([[[0.2525, 0.2931, 0.3893, 0.3330],
#          [0.3949, 0.7346, 0.5196, 0.6585]]])
# tensor([[0.2525, 0.2931, 0.3893, 0.3330, 0.3949, 0.7346, 0.5196, 0.6585]])
# tensor([[[0.2525, 0.2931, 0.3893, 0.3330],
#          [0.3949, 0.7346, 0.5196, 0.6585]]])
# tensor([[[0.2525, 0.3949],
#          [0.2931, 0.7346],
#          [0.3893, 0.5196],
#          [0.3330, 0.6585]]])

# 错误做法：可以运行 logic error
b=torch.rand(1,4,2)
b1=b.transpose(1,2).contiguous() #此函数 用于拼接打乱的数据（重新申请空间在复制）
b2=b1.view(1,4*2)
b3=b2.view(1,4,2)
print(b)
print(b1)
print(b2)
print(b3)  #这样子做以后再恢复形状破坏了原来的数据，即原来的索引对应的数据已经改变了

# result:
# tensor([[[0.5243, 0.0456],
#          [0.9872, 0.0567],
#          [0.9867, 0.4950],
#          [0.4518, 0.7733]]])
# tensor([[[0.5243, 0.9872, 0.9867, 0.4518],
#          [0.0456, 0.0567, 0.4950, 0.7733]]])
# tensor([[0.5243, 0.9872, 0.9867, 0.4518, 0.0456, 0.0567, 0.4950, 0.7733]])
# tensor([[[0.5243, 0.9872],
#          [0.9867, 0.4518],
#          [0.0456, 0.0567],
#          [0.4950, 0.7733]]])



#26. //////  permute
a=torch.rand(4,3,14,28)
print('a: ',a.shape)
a1=a.transpose(1,3)
print(a1.shape)
a2=a1.transpose(1,2)
print(a2.shape)
a3=a.permute(0,2,3,1)  #根据需求按照原来的索引转置，参数为原来的索引
print(a3.shape)

# result：
# a:  torch.Size([4, 3, 14, 28])
# torch.Size([4, 28, 14, 3])
# torch.Size([4, 14, 28, 3])
# torch.Size([4, 14, 28, 3])



#27. //////  cat 即concatnate  tensor'dimension must be consistent
a1=torch.rand(1,2,3)
a2=torch.rand(2,2,3)  #所要cat的tensor的dimension上的elements可以不一致
print(a1)
print(a2)
print(torch.cat([a1,a2],dim=0))
print(torch.cat([a1,a2],dim=0).shape)
a3=torch.rand(1,1,3)
print(a3)
print(torch.cat([a1,a3],dim=1))
print(torch.cat([a1,a3],dim=1).shape)

# result:
# tensor([[[0.7160, 0.6743, 0.9577],
#          [0.4180, 0.2082, 0.0014]]])
# tensor([[[0.9759, 0.2067, 0.8023],
#          [0.3483, 0.9587, 0.5084]],
#
#         [[0.8367, 0.6920, 0.6373],
#          [0.2452, 0.1197, 0.6556]]])
# tensor([[[0.7160, 0.6743, 0.9577],
#          [0.4180, 0.2082, 0.0014]],
#
#         [[0.9759, 0.2067, 0.8023],
#          [0.3483, 0.9587, 0.5084]],
#
#         [[0.8367, 0.6920, 0.6373],
#          [0.2452, 0.1197, 0.6556]]])
# torch.Size([3, 2, 3])
# tensor([[[0.9439, 0.0788, 0.8818]]])
# tensor([[[0.7160, 0.6743, 0.9577],
#          [0.4180, 0.2082, 0.0014],
#          [0.9439, 0.0788, 0.8818]]])
# torch.Size([1, 3, 3])

a=torch.rand(4,32,8) #假设将a的四个班，b五个班的成绩放在一处
b=torch.rand(5,32,8)
print(torch.cat([a,b], dim=0).shape)  #即将两个tensor的dim=0的元素拼接
#
# result:
# torch.Size([9, 32, 8])



#28. //////  stack  create new dimension  拼接的tensor维度必须一致，shape必须一致
a=torch.rand(3,4)
b=torch.rand(2,4)
#此处使用cat，不适用stack，因为a，b的shape不一致
print(torch.cat([a,b],dim=0).shape)

# result:
# torch.Size([5, 4])

a=torch.rand(2,3)
b=torch.rand(2,3)
print(a)
print(b)
print(torch.stack([a,b],dim=0))
print(torch.stack([a,b],dim=0).shape)

# result:
# tensor([[0.2284, 0.8609, 0.4876],
#         [0.1659, 0.2193, 0.5886]])
# tensor([[0.3868, 0.1202, 0.2529],
#         [0.7660, 0.5417, 0.9887]])
# tensor([[[0.2284, 0.8609, 0.4876],
#          [0.1659, 0.2193, 0.5886]],
#
#         [[0.3868, 0.1202, 0.2529],
#          [0.7660, 0.5417, 0.9887]]])
# torch.Size([2, 2, 3])



#29. //////  split: by len
a=torch.rand(32,8)
b=torch.rand(32,8)
print(a.shape)
c=torch.stack([a,b],dim=0)
print(c.shape)
aa,bb=c.split([1,1],dim=0)  #人为根据长度算出所需类型tensor的个数来分
print(aa.shape,bb.shape)
aa1,bb1=c.split(1,dim=0)  #按长度切割，length=1，2/1=2个
print(aa1.shape,bb1.shape)

# result:
# torch.Size([32, 8])
# torch.Size([2, 32, 8])
# torch.Size([1, 32, 8]) torch.Size([1, 32, 8])
# torch.Size([1, 32, 8]) torch.Size([1, 32, 8])



#30. //////  chunk: by num
a=torch.rand(32,8)
b=torch.rand(32,8)
print(a.shape)
c=torch.stack([a,b],dim=0)
print(c.shape)
aa,bb=c.chunk(2,dim=0) #按个数分，dim=0有两个，分成单个的
print(aa.shape,bb.shape)

# result:
# torch.Size([32, 8])
# torch.Size([2, 32, 8])
# torch.Size([1, 32, 8]) torch.Size([1, 32, 8])



#31. //////  basic calculation  #函数和运算符的作用是一样的,推荐使用运算符
a=torch.rand(3,4)
b=torch.rand(4)
print(a)
print(b)
print(torch.add(a,b))
print(a+b)
print(torch.all(torch.eq(a-b,torch.sub(a,b))))
print(torch.all(torch.eq(a*b,torch.mul(a,b))))
print(torch.all(torch.eq(a/b,torch.div(a,b))))

# result:
# tensor([[0.7733, 0.5374, 0.3961, 0.4135],
#         [0.6464, 0.0444, 0.2237, 0.9556],
#         [0.1756, 0.4167, 0.0159, 0.8139]])
# tensor([0.1389, 0.7707, 0.8128, 0.6171]) #将数据broadcast，即复制再相加
# tensor([[0.9122, 1.3081, 1.2089, 1.0306],
#         [0.7853, 0.8151, 1.0365, 1.5726],
#         [0.3145, 1.1874, 0.8287, 1.4310]])
# tensor([[0.9122, 1.3081, 1.2089, 1.0306],
#         [0.7853, 0.8151, 1.0365, 1.5726],
#         [0.3145, 1.1874, 0.8287, 1.4310]])
# tensor(True)
# tensor(True)
# tensor(True)



#32. //////  matmul 矩阵运算
a=torch.full([2,2],3.)
print(a)
b=torch.ones(2,2)
print(b)
print(torch.mm(a,b))
print(torch.matmul(a,b))
print(a@b)

# result:
# tensor([[3., 3.],
#         [3., 3.]])
# tensor([[1., 1.],
#         [1., 1.]])
# tensor([[6., 6.],
#         [6., 6.]])
# tensor([[6., 6.],
#         [6., 6.]])
# tensor([[6., 6.],
#         [6., 6.]])



#33. //////  大于二维的运算
a=torch.rand(4,3,28,64)
b=torch.rand(4,3,64,32)
#print(torch.mm(a,b).shape)  #报错RuntimeError: self must be a matrix
print(torch.matmul(a,b).shape)
c=torch.rand(4,1,64,32)  #先broadcast，再进行矩阵运算
print(torch.matmul(a,c).shape)

# result:
# torch.Size([4, 3, 28, 32])
# torch.Size([4, 3, 28, 32])



#34. //////  power
a=torch.full([2,2],3.)
print(a.pow(2))
print(a**2)
a1=a**2
print(a1.sqrt())
print(a1.rsqrt())  #开根号的倒数
print(a1**(0.5))

# result:
# tensor([[9., 9.],
#         [9., 9.]])
# tensor([[9., 9.],
#         [9., 9.]])
# tensor([[3., 3.],
#         [3., 3.]])
# tensor([[0.3333, 0.3333],
#         [0.3333, 0.3333]])
# tensor([[3., 3.],
#         [3., 3.]])



#35. //////  exp log  这里的log是以e为底的
a=torch.exp(torch.ones(2,2))
print(a)
print(torch.log(a))
#以2为底的：log2，以10为底log10

# result:
# tensor([[2.7183, 2.7183],
#         [2.7183, 2.7183]])
# tensor([[1., 1.],
#         [1., 1.]])



#36. //////  Approximation
a=torch.tensor(3.14)
print(a.floor(),a.ceil(),a.trunc(),a.frac())
#      下限      上限       整数       小数
b=torch.tensor(3.499)
print(b.round())  #四舍五入
b1=torch.tensor(3.5)
print(b1.round())

# result:
# tensor(3.) tensor(4.) tensor(3.) tensor(0.1400)
# tensor(3.)
# tensor(4.)



#37. //////  clamp gradient clipping 将数据限制在一定的范围
#float clamp(float minnumber, float maxnumber, float parameter)
grad=torch.rand(2,3)*15
print(grad)
print(grad.max())
print(grad.min())
print(grad.median())
print(grad.clamp(10))   #最小值为10
print(grad.clamp(0,10))  #最小值0，最大值10

# result:
# tensor([[13.4782, 11.7510,  7.0977],
#         [ 1.4704,  7.5331,  8.9878]])
# tensor(13.4782)
# tensor(1.4704)
# tensor(7.5331)
# tensor([[13.4782, 11.7510, 10.0000],
#         [10.0000, 10.0000, 10.0000]])
# tensor([[10.0000, 10.0000,  7.0977],
#         [ 1.4704,  7.5331,  8.9878]])




#38. //////  norm-p  范数  默认dim参数会被消去
a=torch.full([8],1.)
b=a.view(2,4)
c=a.view(2,2,2)
print(b)
print(c)
print(a.norm(1),b.norm(1),c.norm(1))  #范数1：elements的绝对值之和
print(a.norm(2),b.norm(2),c.norm(2))  #范数2：elements求绝对值之和后开根号
print(b.norm(1,dim=1)) #在dim=1处，范数1
print(b.norm(2,dim=1)) #在dim=1处，范数2
print(c.norm(1,dim=2))
print(c.norm(2,dim=2))
#
# result:
# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.]])
# tensor([[[1., 1.],
#          [1., 1.]],
#
#         [[1., 1.],
#          [1., 1.]]])
# tensor(8.) tensor(8.) tensor(8.)
# tensor(2.8284) tensor(2.8284) tensor(2.8284)
# tensor([4., 4.])
# tensor([2., 2.])
# tensor([[2., 2.],
#         [2., 2.]])
# tensor([[1.4142, 1.4142],
#         [1.4142, 1.4142]])



#39. //////   最大，最小，均值，累乘， 下标最大、最小索引
a=torch.randn(4,10)
print(a[0])
print(a.argmax())
print(a.argmax(dim=1))  #即每一张图dim=1的得到最大下标，共4张

# result:
# tensor([-1.6392,  0.6161, -0.8883,  1.2289,  1.4483, -1.0741, -0.6218,  1.0122,
#         -0.9834, -0.0848])
# tensor(19)
# tensor([4, 9, 3, 7])

a=torch.arange(8).view(2,4).float()
print(a)

print(a.max(),a.min(),a.mean(),a.prod())#prod累乘
print(a.sum(),a.argmax(),a.argmin()) #argmax()不带参数，打平一维，再取索引

# result:
# tensor([[0., 1., 2., 3.],
#         [4., 5., 6., 7.]])
# tensor(7.) tensor(0.) tensor(3.5000) tensor(0.)
# tensor(28.) tensor(7) tensor(0)




#40. //////  Top-k or k-th
a=torch.randn(4,10)
print(a.topk(3,dim=1))  #默认求最大
print(a.topk(3,dim=1,largest=False))  #largest=False求最小
print(a.kthvalue(8,dim=1)) #默认取小。只能取小，此时求的是第8小的，如果没有keepdimension，就会消去一维
print(a.kthvalue(3))

# result:
# torch.return_types.topk(
# values=tensor([[2.3989, 2.0530, 1.2924],
#         [0.3660, 0.1483, 0.1417],
#         [0.8950, 0.2022, 0.1828],
#         [1.5656, 1.5008, 1.4036]]),
# indices=tensor([[0, 5, 9],
#         [9, 5, 7],
#         [3, 1, 0],
#         [6, 3, 5]]))
# torch.return_types.topk(
# values=tensor([[-2.5597, -0.8158, -0.3092],
#         [-1.1891, -0.6543, -0.6349],
#         [-0.8313, -0.7089, -0.4556],
#         [-2.5733, -2.4882, -0.8729]]),
# indices=tensor([[6, 7, 1],
#         [1, 2, 0],
#         [5, 6, 7],
#         [8, 4, 9]]))
# torch.return_types.kthvalue(values=tensor([1.2924, 0.1417, 0.1828, 1.4036]),indices=tensor([9, 7, 0, 5]))
# torch.return_types.kthvalue(values=tensor([-0.3092, -0.6349, -0.4556, -0.8729]),indices=tensor([1, 0, 7, 9]))

a=torch.randn(4,10)
print(a)
print(a.max(dim=1))
print(a.argmax(dim=1))
print(a.max(dim=1,keepdim=True))  #keepdim=True保持原来维度
print(a.argmax(dim=1,keepdim=True))
#
# result:
# tensor([[-3.1342,  1.0709, -1.8499, -1.8785, -0.0580, -0.0042, -0.2360, -0.2132,
#           0.8451, -1.2116],
#         [ 0.3684, -1.7825,  0.6349, -1.3474, -1.7585, -1.9743, -0.1982, -2.2248,
#           0.4035, -0.6871],
#         [-2.6290, -0.2329,  1.5946,  0.1186,  0.2832, -1.5117, -0.5548, -0.1443,
#           0.2427,  1.2009],
#         [-0.7066, -2.0853,  2.2847, -0.5461, -0.9000,  0.1658,  0.6816, -0.3667,
#          -1.5578,  0.1563]])
# torch.return_types.max(
# values=tensor([1.0709, 0.6349, 1.5946, 2.2847]),
# indices=tensor([1, 2, 2, 2]))
# tensor([1, 2, 2, 2])
# torch.return_types.max(
# values=tensor([[1.0709],
#         [0.6349],
#         [1.5946],
#         [2.2847]]),
# indices=tensor([[1],
#         [2],
#         [2],
#         [2]]))
# tensor([[1],
#         [2],
#         [2],
#         [2]])



#41. //////  compare
a=torch.randn(4,10)
print(a>0)
print(torch.gt(a,0))  #gt与a>0等效
print(a!=1)
a1=torch.ones(2,3)
a2=torch.randn(2,3)
print(torch.eq(a1,a2))
print(torch.eq(a1,a1))
print(torch.equal(a1,a1))



#42. //////  where
cond=torch.rand(2,2)
print(cond)
a=torch.ones(2,2)
print(a)
b=torch.zeros(2,2)
print(b)
print(torch.where(cond>0.5,a,b)) # >0.5的数来自a，其他来自b

# # result:
# tensor([[0.0739, 0.5708],
#         [0.6397, 0.8392]])
# tensor([[1., 1.],
#         [1., 1.]])
# tensor([[0., 0.],
#         [0., 0.]])
# tensor([[0., 1.],
#         [1., 1.]])



#43. //////    gether    retrieve label
prob=torch.randn(4,10)
idx=prob.topk(dim=1,k=3)
print(idx)
b=idx[0]
print(b)
label=torch.arange(10)+100
print(label)
print(torch.gather(label.expand(4,10),dim=1,index=b.long()))

# result:
# torch.return_types.topk(
# values=tensor([[1.4218, 0.9622, 0.7820],
#         [2.8708, 1.5598, 1.1551],
#         [1.4081, 0.7554, 0.5810],
#         [1.0881, 0.9273, 0.8013]]),
# indices=tensor([[7, 3, 0],
#         [3, 7, 0],
#         [2, 7, 3],
#         [7, 9, 1]]))
# tensor([[1.4218, 0.9622, 0.7820],
#         [2.8708, 1.5598, 1.1551],
#         [1.4081, 0.7554, 0.5810],
#         [1.0881, 0.9273, 0.8013]])
# tensor([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
# tensor([[101, 100, 100],
#         [102, 101, 101],
#         [101, 100, 100],
#         [101, 100, 100]])



#44. //////  norm范数2再理解
a=torch.tensor([[[[1.,2],[1,1]],[[1,2],[2,1]]]]) #直接创建一个tensor
print(a)
print(a.shape)
print(a.norm(2,dim=0))  # z在指定的dim的element不只有一个时取范数
print(a.norm(2,dim=1))
print(a.norm(2,dim=2))
print(a.norm(2,dim=3))

# result:
# tensor([[[[1., 2.],
#           [1., 1.]],
#          [[1., 2.],
#           [2., 1.]]]])
# torch.Size([1, 2, 2, 2])
# tensor([[[1., 2.],
#          [1., 1.]],
#         [[1., 2.],
#          [2., 1.]]])
# tensor([[[1.4142, 2.8284],
#          [2.2361, 1.4142]]])
# tensor([[[1.4142, 2.2361],
#          [2.2361, 2.2361]]])
# tensor([[[2.2361, 1.4142],
#          [2.2361, 2.2361]]])

a=torch.tensor([[[1.,2],[1,1],[1,2],[2,1]]]) #直接创建一个tensor
print(a)
print(a.shape)
print(a.norm(2,dim=0))
print(a.norm(2,dim=1))
print(a.norm(2,dim=2))

# #result:
# result:  print(a.norm(2,dim=0))
# tensor([[[1., 2.],
#          [1., 1.],
#          [1., 2.],
#          [2., 1.]]])
# torch.Size([1, 4, 2])
# tensor([[1., 2.],
#         [1., 1.],
#         [1., 2.],
#         [2., 1.]])
#
# print(a.norm(2,dim=1)):
# tensor([[2.6458, 3.1623]])
#
# print(a.norm(2,dim=2)):
# tensor([[2.2361, 1.4142, 2.2361, 2.2361]])

```

