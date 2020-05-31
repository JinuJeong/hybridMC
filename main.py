import torch
import sys

if len(sys.argv) <= 1:
    print("Usage: python main.py <NN model> <enable cuda(default: 1)")
    sys.exit()

model_name = sys.argv[1]
use_cuda = 1

if (sys.argv[2] != ""):
    use_cuda = int(sys.argv[2])

model = ""
data = ""

if model_name == "vgg16":
    import vgg16
    model = vgg16.VGG16()
    data = torch.rand(1, 3, 224, 224)
elif model_name == "alexnet":
    import alexnet
    model = alexnet.AlexNet()
    data = torch.rand(1, 1, 32, 32)
else:
    print("Not found model")
    sys.exit()

if use_cuda and torch.cuda.is_available() is True:
    print("Use CUDA\n")
    model.cuda()
    data = data.cuda()

print(f"input size: {data.size()}")

model.eval()

output = model(data)
output = model(data)
output = model(data)

print(output.size())
