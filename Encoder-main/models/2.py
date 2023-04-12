import numpy as np
import torch
import torch.nn.functional as F
from models.helpers import bottleneck_IR, bottleneck_IR_SE, get_blocks
from models.utils import EqualLinear
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, Module, PReLU, Sequential
from torchvision.models import resnet34
from torchsummary import summary


# this is for test!!
class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode="ir", opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        assert num_layers in [50, 100, 152], "num_layers should be 50,100, or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        assert 1==1
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x




if __name__=='__main__':
    model=BackboneEncoderUsingLastLayerIntoW(50)
    model=model.to('cuda')
    z=torch.randn(2,3,512,512)
    z=z.to('cuda')
    summary(model,z)
    #print(model(z).size())
