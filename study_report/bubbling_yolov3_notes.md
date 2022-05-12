# bubbliiing 的yolov3

[TOC]



- 这篇笔记主要是记录up主yolov3模型的整个过程，加上自己的个人理解，并在本地成功运行该模型，模型的测试效果与up主的相差不大。以下是我对整个实现过程的理解，个文件功能的讲解，以及主要代码的详细解析。

![image-20220512183305918](E:\DL_environments\torch\yolov3\bubbliiiing\yolo3-pytorch\weights\项目文件目录.png)

## 网络

### darknet.py

darknet.py这个py文件主要是构建darknet53网络结构，由basic残差快和卷积堆叠而成，卷积为下采样，增加通道数，增大特征图感受野。最后网络输出三张检测小物体，中物体，大物体特征图，其感受野依次增大。

此处为代码展示（需要一点点科学上网）

https://github.com/bubbliiiing/yolo3-pytorch/blob/master/nets/darknet.py