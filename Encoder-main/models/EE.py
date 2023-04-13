'''
efficient encoder
'''
import numpy as np
import torch
import  math
import  torch.nn as nn
from models.helpers import bottleneck_IR, bottleneck_IR_SE, get_blocks
from models.utils import EqualLinear
from torchsummary import summary
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU
# 感觉这个效果不是特别好 ， 还是换成fpn的结构把
class EBlock(nn.Module):
    def __init__(self,spatial,layer_num):
        super().__init__()
        self.model=nn.Sequential(
                              torch.nn.AdaptiveAvgPool2d((7,7)),
                              nn.Flatten(),
                              nn.Linear(spatial*7*7, 512*layer_num))
                            #这里的学习参数比较多的
    def forward(self,x):
        return self.model(x)
# 但是具体的内容还没有想好
class EE_Encoder(nn.Module):
    def __init__(self,img_size,base_size,num_layers,model='ir',input_nc=3):
        super(EE_Encoder, self).__init__()
        assert num_layers in [50,100,152],'num_layers should be 50 ,100,152'
        assert  model in ['ir','ir_se'] ,'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if model == 'ir':
            unit_module = bottleneck_IR
        elif model == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(nn.Conv2d(input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []

        for block in blocks:
            for bottleneck in block:
                last_depth=bottleneck.depth
                modules.append(unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride))
        self.out_pool=torch.nn.AdaptiveAvgPool2d((1,1))
        self.e_block = EBlock(last_depth, 1)
        self.body = nn.Sequential(*modules)
        self.linear = EqualLinear(512, 512, lr_mul=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.e_block(x)
        #print("in EE.py line 49 x size is :",x.size())
        return x



# class EE_Encoder(nn.Module):
#     def __init__(self,img_size,img_base_size,blur_kernel=[1,3,3,1],channel_multiplier=2):
#         # blur_kernel 会有类似小波基的作用
#         super(EE_Encoder,self).__init__()
#
#
#
#         self.channels = {
#             4: 512,
#             8: 512,
#             16: 512,
#             32: 512,
#             64: 256 * channel_multiplier,
#             128: 128 * channel_multiplier,
#             256: 64 * channel_multiplier,
#             512: 32 * channel_multiplier,
#             1024: 16 * channel_multiplier,
#         }
#         self.convs=nn.ModuleList()
#         log_size=int(math.log(img_size,2))
#         log_base_size=int(math.log(img_base_size,2))
#         in_channels=3
#         out_channels=self.channels[img_size]
#         #self.conv=[ResBlock()                 ,in_channels=out_channels  ,  for size in range(log_size,log_base_size,-1)]
#         modules=[]
#         for size in range(log_size,log_base_size,-1):
#             out_channels=self.channels[2**(size-1)]
#             modules.append(ResBlock(in_channels,out_channels,blur_kernel))
#             in_channels=out_channels
#         modules.append(EBlock(in_channels,1))
#         self.body=nn.Sequential(*modules)
#         self.moduleList=list(self.body)
#     def forward(self,x):
#         for i in self.moduleList:
#             x=i(x)
#         return  x


## 要不再来一个 refinement 机制的 ？



if __name__=='__main__':
    model=EE_Enccoder(512,64)
    model=model.to('cuda')

    z=torch.randn(2,3,512,512)
    z=z.to('cuda')
    summary(model,z)
    #print(model(z).size())
