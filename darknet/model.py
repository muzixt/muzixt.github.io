import sys

import torch
from torch import nn
from torchkeras import summary
from torchkeras import Model


def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels / 2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        # ?nn.LeakyReLU()
        return out


class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)
        self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)
        out = self.sf(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


def darknet53(num_classes):
    return Darknet53(DarkResidualBlock, num_classes)


class DarkNet_keras(Model):
    def __init__(self):
        super(DarkNet_keras, self).__init__()
        self.darknet = darknet53(5)

    def forward(self, x):
        return self.darknet(x)



if __name__ == '__main__':
    net = darknet53(1000)
    summary(net, (3, 416, 416))

    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    for p in net.named_children():
        if "fc" not in p:
            p[1].requires_grad_(False)

    optimizer = torch.optim.SGD(net.fc.parameters(), lr=0.001)

    # parameter = torch.load(r"E:\model\darknet\model_best.pth.tar", map_location=torch.device('cpu'))['state_dict']
    # # print(parameter.keys())
    # net.load_state_dict(parameter)

    for p in net.named_children():
        if "fc" not in p:
            p[1].requires_grad_(True)
            # params = next(p.parameters())
            # print(params.size())
            # if params.size():
            optimizer.add_param_group({"params": p[1].parameters()})

    # optimizer.SGD(optim_param, lr=1e-5)
    # for p in net.parameters():
    #     print(p.requires_grad)

    # optimizer.add_param_group({'params': net.parameters()})

    # print(net.state_dict().keys())
    # for k, v in net.state_dict().items():
    #     print(k)
    #     print(v)
    #     break

    # net.load_state_dict(torch.load(r"./darknet.pth", map_location=torch.device('cpu')))
    # var = net.forward(torch.zeros((1, 3, 224, 224))).data
    # print(var)
    # torch.save(net.state_dict(), "./darknet.pth")
