## yolov3网络主干

- #### class notes



yolov3.py

```python
from collections import OrderedDict
import torch
import torch.nn as nn
from darknet import darknet53


# explanation https://blog.csdn.net/L1778586311/article/details/112599259 参考

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size-1) // 2 if kernel_size else 0
    # If stride = 1, output the same size as input, you must use
    # pad = (kernel_size - 1) // 2.(called Same Convolution).
    # If stride = 2, pad = (kernel_size - 1) // 2, output half size.

    # conv2 返回一个nn.Sequential, 等待一个输入即可调用内部所有模块得到输出
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1))]))


def make_last_layers(filter_list, in_filters, out_filter):

    # 将所有层传化成列表便于遍历
    # 前面的卷积操作是“卷积+bn+激活
    m = nn.ModuleList([conv2d(in_filters, filter_list[0], 1),
                       conv2d(filter_list[0], filter_list[1], 3),
                       conv2d(filter_list[1], filter_list[0], 1),
                       conv2d(filter_list[0], filter_list[1], 3),
                       conv2d(filter_list[1], filter_list[0], 1),
                       conv2d(filter_list[0], filter_list[1], 3),
                       # 最后一个卷积就只是一个2D卷积（没有b_norm和激活）
                       nn.Conv2d(filter_list[1], out_filter, kernel_size=1,
                                 stride=1, padding=0, bias=True)
                       ])
    return m


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        super(YoloBody, self).__init__()

        # 生成darknet53的主干模型
        # 获得三个有效的特征层
        # [256, 52, 52] [512, 26, 26] [1024, 13, 13]


        # 创建darknet模型，但不导入预训练权重
        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(
                torch.load("model_data/darknet53_backbone_weight.pth"))

        #   out_filters : [64, 128, 256, 512, 1024]
        # 这三个数是darknet53三条输出支路的输出通道数，即yolo_head的输入通道数
        # 1024是yolo_head1的输入通道数；512是yolo_head2的，256是yolo_head3的
        # self.layers_out_filters = [256, 512, 1024]

        out_filters = self.backbone.layer_out_filters
        # yolo_head的输出通道数: 3*(5+20) = 75
        # 3是先验框的个数,5是x、y、w、h、c 等5个值
        # 20是voc数据集的类别数，80是coco数据集的类别数

        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0])*(num_classes + 5))

        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1])*(num_classes + 5))


        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[3])*(num_classes + 5))




    def forward(self, x):
        # 吧yolo_head1的第四层取出来(要传给yolo_head2)
        # 同理，把yolo_head2的第四层取出来(要传给yolo_head3)
        x2, x1, x0 = self.backbone(x)

        # 第一个特征层
        # out = [b, 255, 13, 13]
        # [1024,13,13]->[512,13,13]->[1024,13,13]->[512,13,13]->[1024,13,13]->[512,13,13]

        out0_branch = self.last_layer0[:5](x0)  # 取出信息
        out0 = self.last_layer0[5:](out0_branch)

        # [512,13,13]->[256,13,13]->[256,26,26]
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        # [256,26,26]+[512,26,26]->[768,26,26]
        x1_in = torch.cat([x1_in, x1], 1)  # 拼接

        # 第二个特征层
        # out1 = [b, 255, 26, 26]
        # [768,26,26]->[256,26,26]->[512,26,26]->[256,26,26]->[512,26,26]->[256,26,26]
        out1_branch = self.last_layer1[:5](out0_branch)
        out1 = self.last_layer1[5:](out1_branch)

        # [256,26,26]->[128,26,26]->[128,52,52]
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # [128,52,52]+[256,52,52]->[384,52,52]
        x2_in = torch.cat([x2_in, x2], 1)

        # 第三个特征层
        # out3 = [b, 255, 52, 52]
        # [384,52,52]->[128,52,52]->[256,52,52]->[128,52,52]->[256,52,52]->[128,52,52]
        out2 = self.last_layer2(x2_in)

        return out0, out1, out2
```



training

```python
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda,
                 anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(YOLOLoss, self).__init__()

        # anchors[9, 2]
        # 13*13的特征层对应的anchor是[116,90][156,198][373,326]
        # 26*26的特征层对应的anchor是[30,61][64,45][59,119]
        # 52*52的特征层对应的anchor是[10,13][16,30][33,23]

        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask

        self.giou = True
        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio = 1 * (num_classes / 80)
        self.ignore_threshold = 0.5
        self.cuda = cuda

    # clip_by_tensor作用是使数据在min到max之间，小于min的变为min，大于max的变为max
    # 括号中的表达式返回的是bool值，用float()转换为float类型数值
    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float()
        return result

    # torch.pow()功能:
    # 实现张量和标量之间逐元素求指数操作,
    # 或者在可广播的张量之间逐元素求指数操作.
    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)  # 方差

    # https://www.cnblogs.com/zhangxianrong/p/14773075.html 关于交叉熵参考
    # 这里是二分类
    def BCELoss(self, pred, target):
        epslion = 1e-7
        pred = self.clip_by_tensor(pred, epslion, 1.0 - epslion)
        output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0-pred)
        return output

    def box_giou(self, b1, b2):
        """

        :param b1: tensor, shape=(b, feat_w, fea_h, anchor_num, 4), xywh  预测
        :param b2: tensor, shape=(b, feat_w, fea_h, anchor_num, 4), xywh  真实
        :return: tensor,shape=(b, feat_w, fea_h, anchor_num, 1)
        """

        # 求出预测框左上角右下角

        b1_xy = b1[..., :2]  # central point (original point)
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half  # top left conner of box
        b1_maxes = b1_xy + b1_wh_half  # lower right conner of box

        # 求出真实框的左上角右下角

        b2_xy = b2[..., :2]
        b2_wh = b1[..., 2:4]
        b2_wh_half = b2_wh / 2
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # 求真实框和预测框所有的iou
        # 框的分布，三种情况，左上、中间、右下

        # torch.zeros_like:生成和括号内变量维度维度一致的全是零的内容
        # torch.max(a, b)在截取的两个tensor(即 a, b)中，比较对应元素，
        # 留下最大的(也就是留下所有重合矩形的左上角)，形成一个tensor并返回

        intersect_mins = torch.max(b1_mins, b2_mins)  # 重合部分的左上角
        intersect_maxes = torch.min(b1_maxes, b2_maxes)  # 重合部分的右下角
        intersect_wh = torch.max(intersect_maxes - intersect_mins,
                                 torch.zeros_like(intersect_maxes))

        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b1_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        # 真框 并 测框 = 真框 + 测框 - 重合部分
        iou = intersect_area / union_area


        # 找到包裹两个框的最小框的左上角和右下角
        # (最小框，即找到一个框框，这个框框能把真框和测框包含且为最小的那一个)

        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins,
                               torch.zeros_like(intersect_maxes))

        # 计算对角线距离
        enclose_area = enclose_wh[..., 0] * enclose_maxes[..., 1]
        giou = iou - (enclose_area - union_area) / enclose_area

        return giou
        # 一种检测的指标，用于loss计算，比iou好，易于优化 :return: tensor,shape=(b, feat_w, fea_h, anchor_num, 1)


    def forward(self, l, input, targets = None):
        """

        :param l: 当前输入进来的有效特征层，是第几个有效特征层
        :param input: shape [bs, 3*(5+num_classes), 13, 13]
                            [bs, 3*(5+num_classes), 26, 26]
                            [bs, 3*(5+num_classes), 52, 52]
        :param targets: 真实框
        :return:
        """

        # 获得图片数量，特征层的高和宽
        # 13和13

        bs = input(0)
        in_h = input(2)
        in_w = input(3)

        # 计算步长  即 比例
        # 每个特征点对应原来的图片多少个像素点
        """
        如果特征层为13*13的话，一个特征点就对应原来图片上的32个像素点
        如果特征层为26*26的话，一个特征点就对应原来图片上的16个像素点
        如果特征层为52*52的话，一个特征点就对应原来图片上的8个像素点
        stride_h = stride_w = 32, 16, 8
        stride_h和stride_w都是32
        """

        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        # 此时获得的scaled_anchors大小是相对于特征层的  共有九个
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for
                          a_w, a_h in self.anchors]

        """
        输入的input一共有三个，他们的shape分别是
        [bs,3*(5+num_classes),13,13] => [bs,3,13,13,5+num_classes]
        [bs, 3, 26, 26, 5+num_classes]
        [bs, 3, 52, 52, 5+num_classes]
        """
        predition = input.view(bs, len(self.anchors_mask[1]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()


        # 先验框的中心位置的调整参数
        x = torch.sigmoid(predition[..., 0])
        y = torch.sigmoid(predition[..., 1])

        # 先验框的宽高调整参数
        w = predition[..., 2]
        h = predition[..., 3]

        # 获得置信度，是否有物体
        conf = torch.sigmoid(predition[..., 4])

        # 种类置信度
        pre_cls = torch.sigmoid(predition[..., 5:])


        # 获得网络应有的预测结果
        # y_true: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh


        y_true, noobj_mask, box_loss_scale = self.get_target(l,
                targets, scaled_anchors, in_h, in_w)

        """
        将预测结果进行解码，判断预测结果和真实值的重合程度
        如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        作为负样本不合适
        """
        noobj_mask, pred_boxes = self.get_ignore(l, x, y, h, w,
                    targets, scaled_anchors, in_h, in_w, noobj_mask)

        if self.cuda:
            y_true = y_true.cuda()
            noobj_mask = noobj_mask.cuda()
            box_loss_scale = box_loss_scale.cuda()

        """
        box_loss_scale是真正预测框的乘积， 宽高均在0-1之间，因此乘积也在0-1之间
        2-宽高的乘积代表真实框越大，比重越小， 小框的比重更大
        """
        # y_true[..., 4]为标签的，若为1，说明有物体
        # 取出true的列（即存在物体的列）
        box_loss_scale = 2 - box_loss_scale
        loss = 0
        obj_mask = y_true[..., 4] == 1
        n = torch.sum(obj_mask)
        if n !=0:
            if self.giou:

                """
                计算预测结果和真实结果的giou
                """
                giou = self.box_giou(pred_boxes, y_true[..., :4])
                loss_loc = torch.mean((1 - giou)[obj_mask])
                # 留下y_true[..., 4]=1的列的loss 而loss = 1-giou  loss_loc=mean_loss
                # torch.shape(features["image"])[0]，后面的中括号表示取返回结果的索引值为0的值

            else:
                # 计算中心偏移的loss， 使用BCELoss(交叉熵)效果会更好一些

                loss_x = torch.mean(self.BCELoss(x[obj_mask],
                                                 y_true[..., 0][obj_mask])*box_loss_scale)
                loss_y = torch.mean(self.BCELoss(y[obj_mask],
                                                 y_true[..., 1][obj_mask])*box_loss_scale)

                # 计算宽高调整值的loss  对于确定的框框用MSELoss更好
                loss_h = torch.mean(self.MSELoss(h[obj_mask], y_true[..., 2][obj_mask])*box_loss_scale)
                loss_w = torch.mean(self.MSELoss(w[obj_mask], y_true[..., 3][obj_mask])*box_loss_scale)
                loss_loc = (loss_x + loss_y + loss_h + loss_w) * 0.1

            loss_cls = torch.mean(self.BCELoss(pre_cls[obj_mask], y_true[..., 5:][obj_mask]))
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio

        loss_conf = torch.mean(self.BCELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])
        loss += loss_conf*self.balance[1] * self.obj_ratio
        # if n != 0:
        #     print(loss_loc * self.box_ratio, loss_cls * self.cls_ratio, loss_conf * self.balance[l] * self.obj_ratio)
        return  loss

    def calculate_iou(self, _box_a, _box_b):
        #

        # 计算真实框的左上角和右下角
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2]/2, _box_a[:, 0] + _box_a[:, 2]/2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3]/2, _box_a[:, 1] + _box_a[:, 3]/2

        # 计算先验框获得的预测框的左下角和右下角
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2]/2, _box_b[:, 0] + _box_b[:, 2]/2
        b2_y1, b2_y2 = _box_b[:, 2] - _box_b[:, 3]/2, _box_b[:, 0] + _box_b[:, 3]/2

        # 将真实框和预测框都转化成左上角右下角的形式
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        # A为真实框的数量， B为先验框的数量
        A = box_a.size(0)
        B = box_b.size(0)

        # 先使A与B的dim一致
        # box_a:[A, 4] => [A, B, 4]
        # box_b:[B, 4] => [A, B, 4]
        # 再计算重合面积
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expend(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expend(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueese(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]

        # 计算预测框和真实框各自的面积  shape 同上
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueese(1).expand_as(inter)
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueese(0).expand_as(inter)

        # 求iou
        union = area_a + area_b - inter
        return inter / union

    def get_target(self, l, targets, anchors, in_h, in_w):

        # 计算一共有多少照片  targets=[bs, box_num, [x,y,h,w...]]
        bs = len(targets)

        # 用于选取那些先验框不包含物体
        noobj_mask = torch.ones(bs, len(self.anchors_mask[1]), in_h, in_w, requires_grad=False)

        # 让网络更加去关注小目标
        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[1]), in_h, in_w, requires_grad=False)

        # [b, 3, 13, 13]
        y_true = torch.zeros(bs, len(self.anchors_mask[1]), in_h, in_w, requires_grad=False)

        for b in range(bs):
            if len(targets[b])==0:  # 判断索引为b的照片有无框框
                continue

            batch_target = torch.zeros_like(targets[b])
            # 真实框的形状[box_num, 5+num_classes]

            # 计算出正样本在特征层上的中心点  # img: x, y
            # 盲猜的batch_target [box_num, 5+num_classes]
            # 即[targets_num, [x1, y1, x2, y2, conf, classes]]
            # img_x = 缩小的x * in_w
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()

            # 将真实框转换一个形式  num_true_box, 4
            # batch_target.size(0) 索引b照片的正实框的数量
            # gt_box_shape[targets_num, 4]  每一个4->[x,y,h,w]=[0,0,h,w]

            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)),
                                                   batch_target[:, 2:4]), 1))

            # 将先验框转换一个形式
            # 9个框 and 4指的是每一个框的 x, y, h, w 即每一个4->[x,y,h,w]=[0,0,h,w]
            # anchor_shapes[9, 4]

            anchor_shapes = torch.FloatTensor(torch.cat((torch.zeros((len(anchors), 2)),
                                                         torch.FloatTensor(anchors)), 1))

            # 计算交并化
            # self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 9]
            # 每一个真实框和9个先验框的重合情况
            # best_ns:
            # [每个真实框最大的重合度max_iou, 每一个真实框重合度最高的先验框的序号]
            # 返回重合度最高的框框的索引

            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)

            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[1]:
                    continue

                # 判断这个先验框是当前特征点的哪一个先验框
                k =self.anchors_mask[1].index(best_n)

                # 获得真实框属于哪个网格点  # 这里可能不需要long()
                i = torch.floor(batch_target[t, 0].long())
                j = torch.floor(batch_target[t, 1].long())

                # 取出真实框的种类
                c = batch_target[t, 4]

                # noobj_mask代表无目标的特征点
                noobj_mask[b, k, j, i] = 0

                # tx, ty 代表中心调整参数的真实值

                if not self.giou:  # self.giou = true
                    # tx, ty 代表中心调整参数的真实值
                    y_true[b, k, j, i, 0] = batch_target[t, 0] - i.float()
                    y_true[b, k, j, i, 1] = batch_target[t, 1] - j.float()
                    y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / anchors[best_n][0])
                    y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / anchors[best_n][1])

                else:
                    # tx, ty 代表中心调整参数的真实值
                    y_true[b, k, j, i, 0] = batch_target[t, 0]
                    y_true[b, k, j, i, 1] = batch_target[t, 1]
                    y_true[b, k, j, i, 2] = batch_target[t, 2]
                    y_true[b, k, j, i, 3] = batch_target[t, 3]
                    y_true[b, k, j, i, 4] = 1
                    y_true[b, k, j, i, c+5] = 1

                # 用于获得xywh的比例
                # 大目标loss权重小， 小目标loss权重大

                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3]/in_h/in_w

        return y_true, noobj_mask, box_loss_scale
```