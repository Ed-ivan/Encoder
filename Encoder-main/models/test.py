
import torch
z=torch.randn(2,3,128,128)
shift = torch.nn.functional.interpolate(z, size=(64,64) , mode='bilinear')

print(shift.size())
