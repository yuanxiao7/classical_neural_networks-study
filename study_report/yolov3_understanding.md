yolov3主体理解

网络架构 

一、darknet53 【1 2 8 8 4 】

卷积 +（下采样 + 1残差块）+（下采样 + 2残差块）+（下采样 + 8残差块）+（下采样 + 8残差块）+ （下采样 + 8残差块）+（下采样 + 4残差块）

输入[b, c, 416, 416]

---> out3 = [b, c, 52, 52]

--->out2= [b, c, 26, 26]

--->out1 = [b, c, 13, 13]

二、只改变通道数

X0 = out1  (7次卷积 )                    X0_out1 = out1  (5次卷积)

X1 = out2 + X0_out1  (7次卷积)    X0_out2 = out1  (5次卷积)

X2 = out3 + X0_out1  (7次卷积)    X0_out2 = out1  (5次卷积)



三、yolo

- 求出预测框、真实框的左上角、右下角

- 1. 左上角 = （中心点横坐标 - 1/2宽 ，中心点纵坐标 - 1/2高）
  2. 右下角 = （中心点横坐标 + 1/2宽 ，中心点纵坐标 + 1/2高）

- 计算 giou
- 1. 全部框内面积 = 真框 + 测框 - 交叉部分
  2. iou = 交叉部分 / 全部框内面积
  3. giou = iou - (对角空处 - 最小涵盖框)  比iou好，易于优化
- 获得与真框相近的先验框并返回iou最大的索引



四、参数调整

- 获得 w、h、anchor 的网格比例
- 参数调整
- 1. loss_cfg = (有物体的概率 - 真实1)的和
  2. loss_框 = (预测框 - 真框) 的和
  3. loss_cls = (具体的物体概率 - 真实) 的和
- loss = loss_cfg + loss_框 + loss_cls

五、training

六、test
