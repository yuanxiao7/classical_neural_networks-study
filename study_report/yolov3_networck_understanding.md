# yolov3_understanding

[TOC]



## yolov3_paper



### ground truth

- 参考答案  也就是指正确打标签的训练数据 或 简单来说就是有效的正确的数据。

- 在有监督学习中，数据是有标注的，以(x, t)的形式出现，其中x是输入数据，t是标注.正确的t标注是ground truth， 错误的标记则不是。（也有人将所有标注数据都叫做ground truth）由模型函数的数据则是由(x, y)的形式出现的。其中x为之前的输入数据，y为模型预测的值。标注会和模型预测的结果作比较。在损耗函数(loss function / error function)中会将y 和 t 作比较，从而计算损耗(loss / error)。

- 由于使用错误的数据，对模型的估计比实际要糟糕。另外，标记数据还被用来更新权重，错误标记的数据会导致权重更新错误。因此使用高质量的数据是很有必要的。

#### 图像中的Ground True

- 例如在一些抠图的项目中，很多人就把Alpha图叫做Ground Truth，Alpha就可以理解成是输入的原始图片对应的Alpha图，也就是原始图对应的标签，或者说是给原始图片用Alpha打了一个标签，而正确的对应于原图的Alpha图就是 Ground Truth 。



### 算子

- 算子就是映射，就是关系，就是变换。



### 边界框预测

- 网络预测每个边界框的四个坐标x, y, w,h如果中心坐标与相对位置有偏离，则网络会对它进行调整，调整的数学形式如下
- ![image-20220428154229850](C:\Users\Happy\AppData\Roaming\Typora\typora-user-images\image-20220428154229850.png)
- 训练用误差平方和来计算损失  真值与预测之差 = 梯度
- 使用逻辑回归来预测目标，一开始只计算先验框内是否有没有目标0-1判断，判断标准，只取最接近标准框的先验框为1再计算损失，即使还有其他框重叠层度超过阈值也是为0且不会计算损失。
- 使用聚类预测宽高，预测中心坐标为每个grid的左上角，偏移量借助了sigmoid函数将中心点坐标范围缩放到0-1。
- <img src="https://img-blog.csdnimg.cn/20200731170846430.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0p3ZW54dWU=,size_16,color_FFFFFF,t_70" alt="img" style="zoom: 67%;" />



### 分类预测

- 每个边界框都会用多标签来预测可能包含的类，用单独的逻辑分类器，因为softmax对网络的性能没有很大的提升，训练过程采用binary cross-entropy即二元交叉熵损失来预测类别。

  注：softmax会加强每一个框恰好有一个类的预测，但一般情况并非如此

### 跨尺度预测

- 系统使用类似于金字塔网络的相似概念，并从中提取特征,在基础特征提取器中加入卷积，最后一个卷积包含编码边界框，目标、类别预测结果的三维张量。
- yolov3 选用9类，3种尺度的先验边界框，聚类确定





## yolov3代码主体理解

### 网络架构 

#### 一、darknet53 【1 2 8 8 4 】

卷积 +（下采样 + 1残差块）+（下采样 + 2残差块）+（下采样 + 8残差块）+（下采样 + 8残差块）+ （下采样 + 8残差块）+（下采样 + 4残差块）

输入[b, c, 416, 416]

---> out3 = [b, c, 52, 52]

--->out2= [b, c, 26, 26]

--->out1 = [b, c, 13, 13]

#### 二、只改变通道数----FPN

经过网络后得到的特征图，7次卷积输出本次特征提取结果并在第5次卷积保留前面提取的特征信息的图像，与后面提取更深层次的特征进行concatenate，这样就可以在更高的感受野中获得小物体的信息，一定程度上减少的数据信息的丢失，也是yolov3相对于yolov1、2的一个很好的网络结构的优化。

X0 = out1  (7次卷积 )                    X0_out1 = out1  (5次卷积)

X1 = out2 + X0_out1  (7次卷积)    X0_out2 = out1  (5次卷积)

X2 = out3 + X0_out1  (7次卷积)    X0_out2 = out1  (5次卷积)



#### 三、yolo框框的形成

- 求出预测框（先验框）、真实框的左上角、右下角

- 1. 左上角 = （中心点横坐标 - 1/2宽 ，中心点纵坐标 - 1/2高）
  2. 右下角 = （中心点横坐标 + 1/2宽 ，中心点纵坐标 + 1/2高）

- 计算 giou
- 1. 全部框内面积 = 真框 + 测框 - 交叉部分
  2. iou = 交叉部分 / 全部框内面积
  3. giou = iou - (对角空处 - 最小涵盖框)  比iou好，易于优化
- 获得与真框相近的先验框并返回iou最大的索引



#### 四、参数调整---LOSS

- 获得 w、h、anchor 的网格比例
- 参数调整
- 1. loss_cfg = (有物体的概率 - 真实1)的和
  2. loss_框 = (预测框 - 真框) 的和
  3. loss_cls = (具体的物体概率 - 真实) 的和
- loss = loss_cfg + loss_框 + loss_cls

五、training

六、test
