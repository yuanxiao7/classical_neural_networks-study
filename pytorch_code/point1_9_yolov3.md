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