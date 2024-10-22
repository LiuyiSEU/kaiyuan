from torch import nn
from torchsummary import summary
import torch
from Model import CBAM
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.attention = CBAM(planes * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.block = block
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv1d(2, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # Conv_block
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # identity_block
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class VGG(nn.Module):
    def __init__(self, features, num_classes=25, init_weights=True, dropout=0.5):
        # 继承
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # AdaptiveAvgPool2d使处于不同大小的图片也能进行分类
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),  # 完成4096的全连接
            nn.Linear(4096, num_classes),  # 对 num_classes的分类
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=True):  # make_layers对输入的cfg进行循环
    layers = []
    in_channels = 2
    for v in cfg:
        if v == "M":  # 对cfg进行输入循环,取第一个v
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]  # 把输入图像进行缩小
        else:
            # v = cast(int, v)
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=5)
# summary(model, (2, 256, 127), 1, 'cpu')

class Block(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.conv(x)


class VGG16(nn.Module):

    def __init__(self, classes) -> None:
        super(VGG16, self).__init__()

        self.in_channel = 2
        self.layer1 = self._make_layer(Block, 32, 1)
        self.layer2 = self._make_layer(Block, 64, 1)
        self.layer3 = self._make_layer(Block, 128, 1)
        self.layer4 = self._make_layer(Block, 256, 1)
        self.layer5 = self._make_layer(Block, 256, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 3, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, classes)
        )

    def _make_layer(self, block, out_channel, block_num):
        conv = []
        for _ in range(block_num):
            conv.append(block(self.in_channel, out_channel))
            self.in_channel = out_channel
        conv.append(CBAM(out_channel))
        return nn.Sequential(*conv)

    def forward(self, x):
        net = self.layer1(x)
        net = self.maxpool(net)
        net = self.layer2(net)
        net = self.maxpool(net)
        net = self.layer3(net)
        net = self.maxpool(net)
        net = self.layer4(net)
        net = self.maxpool(net)
        net = self.layer5(net)
        net = self.maxpool(net)
        net = net.contiguous().view(x.size()[0], -1)
        net = self.classifier(net)
        return net


class AlexNet_model(nn.Module):
    def __init__(self):
        super(AlexNet_model, self).__init__()

        self.conv = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=96, kernel_size=11, stride=4),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=3, stride=2),

                                  nn.Conv1d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=3, stride=2),

                                  nn.Conv1d(in_channels=256, out_channels=388, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=388, out_channels=388, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=388, out_channels=256, kernel_size=3, padding=1),
                                  nn.MaxPool1d(kernel_size=3, stride=2),
                                  )

        self.fc = nn.Sequential(nn.Linear(256 * 126, 4096),
                                nn.ReLU(),
                                nn.Dropout(0.5),

                                nn.Linear(4096, 4096),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(4096, 25)
                                )

    def forward(self, x):
        in_size = x.size(0)
        feature = self.conv(x)
        feature = feature.view(in_size, -1)
        output = self.fc(feature)
        return output


class Inception_block(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception_block, self).__init__()

        self.conv1x1 = nn.Conv1d(in_channels=in_channels, out_channels=c1, kernel_size=1)

        self.conv3x3 = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=c2[0], kernel_size=1),
                                     nn.ReLU(),
                                     nn.Conv1d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1),
                                     nn.ReLU())

        self.conv5x5 = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=c3[0], kernel_size=1),
                                     nn.ReLU(),
                                     nn.Conv1d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, stride=1,
                                               padding=2),
                                     nn.ReLU())

        self.pool3x3 = nn.Sequential(nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                     nn.Conv1d(in_channels=in_channels, out_channels=c4, kernel_size=1),
                                     nn.ReLU())

    def forward(self, x):
        output1 = F.relu(self.conv1x1(x))
        output2 = self.conv3x3(x)
        output3 = self.conv5x5(x)
        output4 = self.pool3x3(x)
        output = [output1, output2, output3, output4]
        return torch.cat(output, dim=1)


class GoogLeNet_Model(nn.Module):
    def __init__(self):
        super(GoogLeNet_Model, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, stride=2, padding=3),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm1d(192),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.inceptionA = nn.Sequential(Inception_block(192, 64, (96, 128), (16, 32), 32),
                                        Inception_block(256, 128, (128, 192), (32, 96), 64),
                                        nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.inceptionB = nn.Sequential(Inception_block(480, 192, (96, 208), (16, 48), 64),
                                        Inception_block(512, 160, (112, 224), (24, 64), 64),
                                        Inception_block(512, 128, (128, 256), (24, 64), 64),
                                        Inception_block(512, 112, (144, 288), (32, 64), 64),
                                        Inception_block(528, 256, (160, 320), (32, 128), 128),
                                        nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.inceptionC = nn.Sequential(Inception_block(832, 256, (160, 320), (32, 128), 128),
                                        Inception_block(832, 384, (192, 384), (48, 128), 128),
                                        nn.AdaptiveAvgPool1d(1),
                                        nn.Dropout(0.4))

        self.fc = nn.Sequential(nn.Linear(1024, 25))

    def forward(self, x):
        in_size = x.size(0)
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.inceptionA(output)
        output = self.inceptionB(output)
        output = self.inceptionC(output)
        output = output.view(in_size, -1)
        output = self.fc(output)
        return output


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
    }
model = ResNet(Bottleneck, [3, 4, 6, 3], 25).to('cuda')
summary(model, (2, 4096))
