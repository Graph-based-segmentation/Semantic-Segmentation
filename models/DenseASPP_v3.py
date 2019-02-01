import torch
import torch.nn.functional as F

from torch import nn
from collections import OrderedDict, namedtuple
from torch.nn import BatchNorm2d as bn
# from cfgs import DenseASPP161

class DenseASPP(nn.Module):
    """
    * output_scale can only set as 8 or 16
    """
    def __init__(self, model_cfg, n_class=19, output_stride=8):
        super(DenseASPP, self).__init__()
        bn_size = model_cfg['bn_size']
        drop_rate = model_cfg['drop_rate']
        growth_rate = model_cfg['growth_rate']
        num_init_features = model_cfg['num_init_features']
        block_config = model_cfg['block_config']

        dropout0 = model_cfg['dropout0']
        dropout1 = model_cfg['dropout1']
        d_feature0 = model_cfg['d_feature0']
        d_feature1 = model_cfg['d_feature1']

        feature_size = int(output_stride / 8)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', bn(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        # block1*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[0], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % 1, block)

        # [[ why use block_config tuple?? ]]
        # If this tuple is not used, same first num_features as last num_features.
        num_features = num_features + block_config[0] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.features.add_module('transition%d' % 1, trans)
        num_features = num_features // 2

        # block2*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[1], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % 2, block)
        num_features = num_features + block_config[1] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=feature_size)
        self.features.add_module('transition%d' % 2, trans)
        num_features = num_features // 2

        # block3*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate, dilation_rate=int(2 / feature_size))
        self.features.add_module('denseblock%d' % 3, block)
        num_features = num_features + block_config[2] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=1)
        self.features.add_module('transition%d' % 3, trans)
        num_features = num_features // 2

        # block4*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate, dilation_rate=int(4 / feature_size))
        self.features.add_module('denseblock%d' % 4, block)
        num_features = num_features + block_config[3] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=1)
        self.features.add_module('transition%d' % 4, trans)

        num_features = num_features // 2
        # #################################### remove the last two pooling layers ######################################
        # #################################### remove the last classification layer ####################################
        # #################################### set the dilation rates 2 and 4 ##########################################

        # Final batch norm
        self.features.add_module('norm5', bn(num_features))
        if feature_size > 1:
            self.features.add_module('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))

        # [[ ASPP_3/6/12/18/24 : dilated convolutional layer ]]
        self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=False)

        self.ASPP_6 = _DenseAsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True)

        self.ASPP_12 = _DenseAsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0, bn_start=True)

        self.ASPP_18 = _DenseAsppBlock(input_num=num_features + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0, bn_start=True)

        self.ASPP_24 = _DenseAsppBlock(input_num=num_features + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0, bn_start=True)
        """================================================MK========================================================"""
        self.ASPP_v2_3 = _DenseAsppBlock_v3(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                            dilation_rate=2, drop_out=dropout0, bn_start=True)
        self.ASPP_v2_4 = _DenseAsppBlock_v3(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                            dilation_rate=2, drop_out=dropout0, bn_start=True)
        self.ASPP_v2_5 = _DenseAsppBlock_v3(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                            dilation_rate=2, drop_out=dropout0, bn_start=True)
        """------------------------------------------------MK--------------------------------------------------------"""
        # self.concat_conv = nn.Conv2d(in_channels=832 * 2, out_channels=832, kernel_size=1, stride=1, padding=0,
        #                              bias=False)
        self.concat_conv = nn.Conv2d(in_channels=1536, out_channels=832, kernel_size=1, stride=1, padding=0,
                                     bias=False)
        """================================================MK========================================================"""
        num_features = num_features + 5 * d_feature1

        self.classification = nn.Sequential(
            nn.Dropout2d(p=dropout1),
            nn.Conv2d(in_channels=num_features, out_channels=n_class, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=8, mode='bilinear'),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform(m.weight.data)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    """[[ In segmentation_main2.py, after outputs = net(inputs) come here ]]"""
    def forward(self, _input):
        feature = self.features(_input)                           # batch x 512 x 64 x 64

        aspp3 = self.ASPP_3(feature)                              # batch x 64 x 64 x 64
        feature_aspp3 = torch.cat((aspp3, feature), dim=1)
        # batch x 576 x 64 x 64

        aspp6 = self.ASPP_6(feature_aspp3)                        # batch x 64 x 64 x 64
        feature_aspp6 = torch.cat((aspp6, feature_aspp3), dim=1)
        # batch x 640 x 64 x 64

        aspp12 = self.ASPP_12(feature_aspp6)                      # batch x 64 x 64 x 64
        feature_aspp12 = torch.cat((aspp12, feature_aspp6), dim=1)
        # batch x 704 x 64 x 64

        aspp18 = self.ASPP_18(feature_aspp12)                      # batch x 64 x 64 x 64
        feature_aspp18 = torch.cat((aspp18, feature_aspp12), dim=1)
        # batch x 768 x 64 x 64

        aspp24 = self.ASPP_24(feature_aspp18)                      # batch x 64 x 64 x 64
        feature_aspp24 = torch.cat((aspp24, feature_aspp18), dim=1)

        """================================================MK========================================================"""
        aspp12_v2 = self.ASPP_v2_3(feature)
        feature_aspp12_v2 = torch.cat((aspp12_v2, feature), dim=1)

        aspp18_v2 = self.ASPP_v2_4(feature_aspp12_v2)
        feature_aspp18_v2 = torch.cat((aspp18_v2, feature_aspp12_v2), dim=1)

        aspp24_v2 = self.ASPP_v2_5(feature_aspp18_v2)
        feature_aspp24_v2 = torch.cat((aspp24_v2, feature_aspp18_v2), dim=1)

        concat_feature = torch.cat((feature_aspp24, feature_aspp24_v2), dim=1)
        output = self.concat_conv(concat_feature)
        """================================================MK========================================================"""

        cls = self.classification(feature_aspp24)                  # batch x num_class x 512 x 512

        return cls

class _DenseAsppBlock_v3(nn.Sequential):
    """ ConvNet block for building DenseASPP. \
    _DenseASPPBlock is performed like bottleneck layer in DenseNet"""

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):

        # the number of output feature is transferred from num1 to num2. (num1 > num2)
        super(_DenseAsppBlock_v3, self).__init__()
        if bn_start:
            self.add_module('norm.1', bn(input_num, momentum=0.0003)),

        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm.2', bn(num1, momentum=0.0003)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=5,
                                            dilation=dilation_rate, padding=dilation_rate * 2)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock_v3, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature

class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. \
    _DenseASPPBlock is performed like bottleneck layer in DenseNet"""

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):

        # the number of output feature is transferred from num1 to num2. (num1 > num2)
        super(_DenseAsppBlock, self).__init__()
        if bn_start:
            self.add_module('norm.1', bn(input_num, momentum=0.0003)),

        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm.2', bn(num1, momentum=0.0003)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature

# [[ DenseNet ]]
# [[ _DenseLayer is bottleneck layer ]]
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation_rate=1):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', bn(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', bn(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, dilation=dilation_rate, padding=dilation_rate, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dilation_rate=1):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):

            # [[ _DenseLayer is bottleneck layer ]]
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate, dilation_rate=dilation_rate)

            # [[ self.add_module is performed addition]]
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, stride=2):
        super(_Transition, self).__init__()
        self.add_module('norm', bn(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if stride == 2:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=stride))


# if __name__ == "__main__":
#     model = DenseASPP(model_cfg=DenseASPP161.Model_CFG)
#     print(model)