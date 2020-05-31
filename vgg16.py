from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time

def time_ns():
    return time.time() * (10**9)

class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()

        self.features = []
        self.fc = []
        self.func_names = ["conv1-1", "relu1-1", "conv1-2", "relu1-2", "pool1",
                           "conv2-1", "relu2-1", "conv2-2", "relu2-2", "pool2",
                           "conv3-1", "relu3-1", "conv3-2", "relu3-2", "conv3-3", "relu3-3", "pool3",
                           "conv4-1", "relu4-1", "conv4-2", "relu4-2", "conv4-3", "relu4-3", "pool4",
                           "conv5-1", "relu5-1", "conv5-2", "relu5-2", "conv5-3", "relu5-3", "pool5"]

        self.features.append(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        self.features.append(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        self.features.append(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        self.features.append(nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        self.features.append(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        
        self.fc.append(nn.Linear(512*7*7, 4096))
        self.fc.append(nn.Linear(4096, 4096))
        self.fc.append(nn.Linear(4096, 1024))
        
        for i in range(len(self.features)):
            self.add_module(self.func_names[i], self.features[i])

        self.add_module('fc1', self.fc[0])
        self.add_module('fc2', self.fc[1])
        self.add_module('fc3', self.fc[2])
       
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        start_time = time_ns()
        print("name, latency(ms), size (MB)")

        for i in range(len(self.features)):
            a = time_ns()
            x = self.features[i](x)
            b = (time_ns() - a) / (10 ** 6)
            print(f"{self.func_names[i]}, {b:.4f}, {x.reshape(-1,).size()[0] * 8 / 1024 / 1024 }")

        x = x.view(-1, 512*7*7)

        for i in range(len(self.fc)):
            a = time_ns()
            x = self.fc[i](x)
            b = (time_ns() - a) / (10 ** 6)
            print(f"fc{i}, {b:.4f}, {x.reshape(-1,).size()[0] * 8 / 1024 / 1024}")

        x = F.log_softmax(x, dim=1)

        end_time = time_ns()
        delay = (end_time - start_time) / (10 ** 6)
        print(f"total delay: {delay:.4f}ms")

        return x
    
if __name__ == "__main__":
    model = VGG16()
    data = torch.rand(1, 3, 224, 224)

    if torch.cuda.is_available() is True:
        print("Use CUDA")
        model.cuda()
        data = data.cuda()

    print(f"input size: {data.size()}")

    model.eval()
    model.eval

    output = model(data)
    output = model(data)
    output = model(data)

    print(output.size())
