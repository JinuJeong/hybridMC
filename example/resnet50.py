import torch
import torchvision
import time
import sys
import getopt

def time_ns():
    return time.time() * (10 ** 9)


if __name__ == "__main__":
    model = torchvision.models.resnet50(pretrained=True)
    data = torch.rand(1, 3, 224, 224)
    warmup_step = 5
    use_gpu = False

    opts, args = getopt.getopt(sys.argv[1:], "g") 

    for o, a in opts:
        if o == "-g":
            use_gpu = True

    if use_gpu is True:
        if torch.cuda.is_available() is True:
            model.cuda()
            data = data.cuda()
            print("Use GPU")
        else:
            print("Use CPU")
    else:
        print("Use CPU")

    model.eval()

    print("Start Warmup")

    for i in range(warmup_step):
       output = model(data) 

    print("Done Warmup")

    a = time_ns()
    output = model(data)
    b = time_ns()

    delay_us = (b - a) / 1000

    print(f"Computation Time: {delay_us}us")
