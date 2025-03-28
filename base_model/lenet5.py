import torch.nn as nn
from builder import ConvBuilder
import numpy as np
LENET_ORIGIN_DEPS = np.array([32, 64], dtype=np.int32)

class LeNet5BN(nn.Module):

    def __init__(self, builder:ConvBuilder, deps):
        super(LeNet5BN, self).__init__()
        self.bd = builder
        deps = LENET_ORIGIN_DEPS
        stem = builder.Sequential()
        stem.add_module('conv1', builder.Conv2dBNReLU(in_channels=3, out_channels=deps[0], kernel_size=5, padding=2))
        stem.add_module('maxpool1', builder.Maxpool2d(kernel_size=2))
        stem.add_module('conv2', builder.Conv2dBNReLU(in_channels=deps[0], out_channels=deps[1], kernel_size=5, padding=2))
        stem.add_module('maxpool2', builder.Maxpool2d(kernel_size=2))
        self.stem = stem
        self.flatten = builder.Flatten()
        self.linear1 = builder.IntermediateLinear(in_features=deps[1] * 64, out_features=500)
        self.relu1 = builder.ReLU()
        self.linear2 = builder.Linear(in_features=500, out_features=10)

    def forward(self, x):
        feat = self.stem(x)
        out = self.flatten(feat)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        return out,feat



class LeNet5BN_deep(nn.Module):

    def __init__(self, builder:ConvBuilder, deps):
        super(LeNet5BN_deep, self).__init__()
        self.bd = builder
        deps = LENET_ORIGIN_DEPS
        stem = builder.Sequential()
        stem.add_module('conv1', builder.Conv2dBNReLU(in_channels=3, out_channels=deps[0], kernel_size=5, padding=2))
        stem.add_module('conv2', builder.Conv2dBNReLU(in_channels=deps[0], out_channels=deps[0], kernel_size=5, padding=2))
        stem.add_module('maxpool1', builder.Maxpool2d(kernel_size=2))
        stem.add_module('conv3', builder.Conv2dBNReLU(in_channels=deps[0], out_channels=deps[1], kernel_size=5, padding=2))
        stem.add_module('conv4', builder.Conv2dBNReLU(in_channels=deps[1], out_channels=deps[1], kernel_size=5, padding=2))
        stem.add_module('maxpool2', builder.Maxpool2d(kernel_size=2))
        self.stem = stem
        self.flatten = builder.Flatten()
        self.linear1 = builder.IntermediateLinear(in_features=deps[1] * 64, out_features=500)
        self.relu1 = builder.ReLU()
        self.linear2 = builder.Linear(in_features=500, out_features=10)

    def forward(self, x):
        feat = self.stem(x)
        out = self.flatten(feat)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        return out,feat


def create_lenet5bn(cfg, builder):
    return LeNet5BN(builder=builder, deps=cfg.deps)

def create_lenet5bn_deep(cfg, builder):
    return LeNet5BN_deep(builder=builder, deps=cfg.deps)
