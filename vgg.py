'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch import Tensor
import mc_dropout
from torch.nn import functional as F
import numpy as np

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, cln):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, cln)

    def forward(self, x):
        out = self.features(x)
        emb = out.view(out.size(0), -1)
        out = self.classifier(emb)
        return out, emb

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def get_embedding_dim(self):
        return 512

class VGGDropout(mc_dropout.BayesianModule):
    def __init__(self, vgg_name, cln, init_weights=False):
        super(VGGDropout, self).__init__(10)
        self.features = self._make_layers(cfg[vgg_name])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            mc_dropout.MCDropout(),
            nn.Linear(512 * 1 * 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, cln),
        )

        if init_weights:
            self.apply(self.initialize_weights)

    def deterministic_forward_impl(self, x: Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def mc_forward_impl(self, x: Tensor):
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def get_embedding_dim(self):
        return 512

    def forward(self, x):
        out = self.features(x)
        emb = out.view(out.size(0), -1)
        out = self.classifier(emb)
        return out, emb

    def forward_old(self, x):
        k = 10
        self.k = k

        #out = self.features(x)
        #emb = out.view(out.size(0), -1)
        #out = self.classifier(emb)
        #print(out.size())

        input_B = self.deterministic_forward_impl(x)
        #print(input_B.size())


        mc_input_BK = self.mc_tensor(input_B, k)
        #print(mc_input_BK.size())
        mc_output_BK = self.mc_forward_impl(mc_input_BK)
        #print(mc_output_BK.size())
        mc_output_B_K = self.unflatten_tensor(mc_output_BK, k)
        #print(mc_output_B_K.size())
        output = mc_output_B_K[:,0,:]
        #print(mc_output_B_K2.size())

        return output, 0

def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
