from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time

class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        start_time = time.time_ns()

        x = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(x)
        x = nn.ReLU(inplace=True)(x)

        x = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(x)
        x = nn.ReLU(inplace=True)(x)
        
#=======================================================================================

        x = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)(x)

        x = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(x)
        x = nn.ReLU(inplace=True)(x)

        x = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(x)
        x = nn.ReLU(inplace=True)(x)

#=======================================================================================

        x = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)(x)
   
        x = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(x)
        x = nn.ReLU(inplace=True)(x)

        x = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(x)
        x = nn.ReLU(inplace=True)(x)

        x = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(x)
        x = nn.ReLU(inplace=True)(x)

#=======================================================================================

        x = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)(x)

        x = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(x)
        x = nn.ReLU(inplace=True)(x)

        x = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(x)
        x = nn.ReLU(inplace=True)(x)

        x = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(x)
        x = nn.ReLU(inplace=True)(x)

#=======================================================================================

        x = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)(x)

        x = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(x)
        x = nn.ReLU(inplace=True)(x)

        x = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(x)
        x = nn.ReLU(inplace=True)(x)

        x = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))(x)
        x = nn.ReLU(inplace=True)(x)

#=======================================================================================

        x = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)(x)

        x = x.view(-1, 512*7*7)
        x = nn.Linear(512*7*7, 4096)(x)
        x = nn.Linear(4096, 4096)(x)
        x = nn.Linear(4096, 1024)(x)
        x = F.log_softmax(x, dim=1)

        end_time = time.time_ns()
        delay = (end_time - start_time) / (10 ** 6)
        print(f"{delay:.4f}ms")

        return x
    
model = VGG16()

data = torch.rand(1, 3, 224, 224)

model.eval()

output = model(data)

print(output.shape)
