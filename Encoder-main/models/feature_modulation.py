# from  HFGI  for  feature modulation
import torch
import  numpy as np
from torch import  nn
from helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE, _upsample_add
from torch.nn import Conv2d, BatchNorm2d, PReLU
from utils import EqualLinear,ScaledLeakyReLU,EqualConv2d
from torchsummary import summary
class LocalFeatEncoder(nn.Module):
    #不如直接作为modulation部分 对fusion之后的进行modulation
    def __init__(self):
        super(LocalFeatEncoder, self).__init__()
        self.conv_layer1=nn.Sequential(Conv2d(3, 32, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(32),
                                      PReLU(32))
        self.conv_layer2 = nn.Sequential(*[bottleneck_IR(32, 48, 2), bottleneck_IR(48, 48, 1), bottleneck_IR(48, 48, 1)])

        self.conv_layer3 = nn.Sequential(*[bottleneck_IR(48, 64, 2), bottleneck_IR(64, 64, 1), bottleneck_IR(64, 64, 1)])
        # bottleneck_IR(3,48,2) 输入 是 [1,3,64,64] 的输出就会得到一个  [1,48,32,32 ]  stide 如果是 1 get 64， 64
        self.condition_scale3 = nn.Sequential(
            EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True),
            ScaledLeakyReLU(0.2),
            EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True))

        self.condition_shift3 = nn.Sequential(
            EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True),
            ScaledLeakyReLU(0.2),
            EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True))
    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it
    def forward(self, x):
        conditions = []
        feat1 = self.conv_layer1(x)
        feat2 = self.conv_layer2(feat1)
        feat3 = self.conv_layer3(feat2)

        scale = self.condition_scale3(feat3)
        scale = torch.nn.functional.interpolate(scale, size=(64,64) , mode='bilinear')
        conditions.append(scale.clone())
        shift = self.condition_shift3(feat3)
        shift = torch.nn.functional.interpolate(shift, size=(64,64) , mode='bilinear')
        conditions.append(shift.clone())
        # conditions is list  scale and shift (for affine trans)
        # 对应到模型图中
        return conditions


if __name__=='__main__':
    model=LocalFeatEncoder()
    model=model.to('cuda')
    z=torch.randn(2,3,512,512)
    z=z.to('cuda')
    summary(model,z)
    #print(model(z).size())



# 暂时启用 ， 不如直接将
class FeatEncoder(nn.Module):
    def __init__(self):
        pass

    def forward(self,x,rec_x,segm):
        '''
        :param x: origin input
        :param segm:  parsing
        y_hat： rec_x
        :return:
        '''
        pass