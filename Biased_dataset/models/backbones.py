
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def init_layer(L):
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

# Simple ResNet Block
class SimpleBlock(nn.Module):
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim or indim == 64 or indim == 128:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)
            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x,no_relu=False):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        if no_relu:
            return out
        out = self.relu2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=True,contra_dim = 32,further_cls =False):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet, self).__init__()
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3,
                          bias=False)
        bn1 = nn.BatchNorm2d(32)
        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 32
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AdaptiveAvgPool2d((2,2))
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim * 2 * 2
        else:
            self.final_feat_dim = [indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)
        self.classifier = nn.Linear(self.final_feat_dim, 10)
        self.contra_head = nn.Linear(self.final_feat_dim, contra_dim)
        self.contra_head2 = nn.Linear(self.final_feat_dim, contra_dim)
        if further_cls:
            self.contra_cls = nn.Linear(contra_dim, 10)

    def forward(self, x, need_feature = False, need_contra_features = False, need_2nd_features = False,
                need_contra_cls = False):
        features = self.trunk(x)
        if need_2nd_features:
            features2 = self.contra_head2(features)
            return features2
        if need_contra_features:
            features = self.contra_head(features)
            return features
        if need_contra_cls:
            features = self.contra_head(features)
            return self.contra_cls(features)
        out = self.classifier(features)
        if need_feature:
            return out, features
        return out
    def get_features_contra(self, x):
        features = self.trunk(x)
        features = self.contra_head(features)
        return features


from typing import Union, List, Dict, Any, cast
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def ResNet18(args=None, flatten=True, further_cls = False):
    if args is None:
        return ResNet(SimpleBlock, [2, 2, 2, 2], [64, 128, 256, 512], flatten)
    else:
        return ResNet(SimpleBlock, [2, 2, 2, 2], [64, 128, 256, 512], flatten, contra_dim = args.contra_dim, further_cls = further_cls)




