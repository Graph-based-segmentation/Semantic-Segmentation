import torch
import torch.nn.functional as F

from torch import nn
from collections import OrderedDict
from torch.nn import BatchNorm2d as bn

def fixed_padding(inputs, kernel_size, dilation):
    # kernel_size = 3, dilation; increasing by 2

    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))

    return padded_inputs

class depthwise_separable_conv(nn.Module):
    """At groups='in_channels', each input channel is convolved with its own set of filters
    (of size out_channels // in_channels)"""

    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(depthwise_separable_conv, self).__init__()

        if dilation == 1:
            self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding=1, dilation=dilation, groups=inplanes, bias=bias)
        else:
            self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding=dilation, dilation=dilation,
                                   groups=inplanes, bias=bias)

        self.pointwise = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)

    def forward(self, x):
        # x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])

        x = self.conv1(x)
        x = self.pointwise(x)

        return x

class DenseASPP_boundary_depthwise(nn.Module):
    """
    * output_scale can only set as 8 or 16
    """
    def __init__(self, model_cfg, n_class=19, output_stride=8):
        super(DenseASPP_boundary_depthwise, self).__init__()

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
        """=========================== MK ============================="""
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm0', bn(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),

            # ('conv1', nn.Conv2d(num_init_features, d_feature0, kernel_size=3, stride=1, padding=1, bias=False)),
            # ('bn1'  , bn(d_feature0)),
            # ('relu1', nn.ReLU(inplace=True)),

            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        """=========================== MK ============================="""

        """========================================== DenseNet121 =============================================="""
        # Each denseblock
        num_features = num_init_features

        # block1*****************************************************************************************************
        self.dense_features = nn.Sequential()
        block = _DenseBlock(num_layers=block_config[0], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

        self.dense_features.add_module('denseblock%d' % 1, block)
        num_features = num_features + block_config[0] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.dense_features.add_module('transition%d' % 1, trans)
        num_features = num_features // 2

        # block2*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[1], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.dense_features.add_module('denseblock%d' % 2, block)
        num_features = num_features + block_config[1] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=feature_size)
        self.dense_features.add_module('transition%d' % 2, trans)
        num_features = num_features // 2

        # block3*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate, dilation_rate=int(2 / feature_size))
        self.dense_features.add_module('denseblock%d' % 3, block)
        num_features = num_features + block_config[2] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=1)
        self.dense_features.add_module('transition%d' % 3, trans)
        num_features = num_features // 2

        # block4*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate, dilation_rate=int(4 / feature_size))
        self.dense_features.add_module('denseblock%d' % 4, block)
        num_features = num_features + block_config[3] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=1)
        self.dense_features.add_module('transition%d' % 4, trans)
        num_features = num_features // 2

        # Final batch norm
        self.dense_features.add_module('norm5', bn(num_features))
        if feature_size > 1:
            self.dense_features.add_module('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
        """=========================================================================================================="""

        self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=False)

        self.ASPP_6 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True)

        self.ASPP_9 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=9, drop_out=dropout0, bn_start=True)

        self.ASPP_12 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0, bn_start=True)

        self.ASPP_18 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0, bn_start=True)

        self.ASPP_24 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0, bn_start=True)

        num_features = num_features + 5 * d_feature1

        self.conv1x1 = nn.Conv2d(in_channels=d_feature1 * 6, out_channels=num_features,
                                 kernel_size=1, padding=0)

        self.classification = nn.Sequential(
            nn.Dropout2d(p=dropout1),
            # nn.Conv2d(in_channels=num_features, out_channels=n_class, kernel_size=1, padding=0),
            nn.Conv2d(in_channels=num_features, out_channels=n_class, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=8, mode='bilinear'),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform(m.weight.data)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, _input):
        """====depthwise convolution densenet====="""
        feature = self.features(_input)
        feature = self.dense_features(feature)
        """======================================="""

        mk_aspp3 = self.ASPP_3(feature)
        mk_aspp6 = self.ASPP_6(feature)
        mk_aspp9 = self.ASPP_9(feature)
        mk_aspp12 = self.ASPP_12(feature)
        mk_aspp18 = self.ASPP_18(feature)
        mk_aspp24 = self.ASPP_24(feature)
        concat_feature1 = torch.cat((mk_aspp3, mk_aspp6, mk_aspp9, mk_aspp12, mk_aspp18, mk_aspp24), dim=1)
        conv_feature1 = self.conv1x1(concat_feature1)
        # aspp3 = self.ASPP_3(feature)
        # feature = torch.cat((aspp3, feature), dim=1)
        #
        # aspp6 = self.ASPP_6(feature)
        # feature = torch.cat((aspp6, feature), dim=1)
        #
        # aspp12 = self.ASPP_12(feature)
        # feature = torch.cat((aspp12, feature), dim=1)
        #
        # aspp18 = self.ASPP_18(feature)
        # feature = torch.cat((aspp18, feature), dim=1)
        #
        # aspp24 = self.ASPP_24(feature)
        # feature = torch.cat((aspp24, feature), dim=1)

        cls = self.classification(conv_feature1)

        return cls


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        if bn_start:
            self.add_module('norm.1', bn(input_num, momentum=0.0003)),

        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),
        # self.add_module('conv.1', nn.Conv2d(in_channels=input_num, out_channels=input_num, kernel_size=1)),

        self.add_module('norm.2', bn(num1, momentum=0.0003)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        # self.add_module('conv.2', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=3,
        #                                     dilation=dilation_rate, padding=dilation_rate)),
        self.add_module('conv.2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation_rate=1):
        super(_DenseLayer, self).__init__()

        self.add_module('norm.1', bn(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        # self.add_module('conv_1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('depthwise_conv.1', depthwise_separable_conv(num_input_features, bn_size * growth_rate,
                                                                     kernel_size=3, stride=1, dilation=1))

        self.add_module('norm.2', bn(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        # self.add_module('conv_2', nn.Conv2d(bn_size * growth_rate, growth_rate,
        #                 kernel_size=3, stride=1, dilation=dilation_rate, padding=dilation_rate, bias=False)),
        self.add_module('depthwise_conv.2', depthwise_separable_conv(bn_size * growth_rate, growth_rate,
                                                                     kernel_size=3, stride=1, dilation=dilation_rate))
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
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate, dilation_rate=dilation_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, stride=2):
        super(_Transition, self).__init__()
        self.add_module('norm', bn(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if stride == 2:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=stride))