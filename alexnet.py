import torch
import torch.nn as nn
import time
import torch.nn.functional as F

def time_ns():
    return time.time() * (10 ** 9)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = []
        self.fc = []
        self.func_names = ["conv1", "relu1", "pool1", "conv2", "relu2", "pool2"]

        self.features.append(nn.Conv2d(1, 32, 5, padding=2))
        self.features.append(nn.ReLU(True))
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.features.append(nn.Conv2d(32, 64, 5, padding=2))
        self.features.append(nn.ReLU(True))
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        for i in range(len(self.features)):
            self.add_module(self.func_names[i], self.features[i])

        self.fc.append(nn.Linear(64*8*8, 1024))
        self.fc.append(nn.Linear(1024, 10))
    
        self.add_module('fc1', self.fc[0])
        self.add_module('fc2', self.fc[1])
        
    def forward(self, x):
        start_time = time_ns()
        for i in range(len(self.features)):
            a = time_ns()
            x = self.features[i](x)
            b = (time_ns() - a) / (10 ** 6)
            print(f"{self.func_names[i]}, {b:.4f}, {x.reshape(-1,).size()[0] * 8 / 1024 / 1024 }")

        x = x.view(-1, 64*8*8)
        
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
    model = AlexNet()
    data = torch.rand(16, 1, 32, 32)

    if torch.cuda.is_available() is True:
        model.cuda()
        data = data.cuda()

    print(f"input size: {data.size()}")

    model.eval()

    output = model(data)

    print(output.size())

