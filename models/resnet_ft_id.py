import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from resnet_ft import ResNet

# MedicalNet-Tencent 
# https://github.com/Tencent/MedicalNet/blob/master/models/resnet.py
# https://github.com/Tencent/MedicalNet/blob/master/model.py

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_id(ResNet):
    
    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda = False,
                 num_features = 128):

        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], shortcut_type)        
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.layer_maxpool = nn.AdaptiveMaxPool3d(1)
        
        self.fc1 = nn.Sequential(
            nn.Linear(64 * block.expansion, num_features),
            nn.Dropout(0.5)
        )
        self.cls1 = nn.Sequential(
            nn.Linear(num_features,2)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128 * block.expansion, num_features),
            nn.Dropout(0.5)
        )
        self.cls2 = nn.Sequential(
            nn.Linear(num_features,2)
        )
           
        self.fc3 = nn.Sequential(
            nn.Linear(256 * block.expansion,num_features),
            nn.Dropout(0.5)
        )
        self.cls3 = nn.Sequential(
            nn.Linear(num_features,2)
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion,num_features),
            nn.Dropout(0.5)
        )
        self.cls = nn.Sequential(
            nn.Linear(num_features,2)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward_once(self, x):
        x = self.conv1(x)  
        x = self.bn1(x)    
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) 
        x1 = self.layer_maxpool(x)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1(x1)
        
        x = self.layer2(x)
        x2 = self.layer_maxpool(x)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc2(x2)
        
        x = self.layer3(x)
        x3 = self.layer_maxpool(x)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc3(x3)
        
        x = self.layer4(x)
        x4 = self.layer_maxpool(x)
        x4 = x4.view(x4.size(0), -1)
        x4 = self.fc(x4)

        return x1,x2,x3,x4

    def forward(self,x0,x1):
        out01, out02, out03, out04 = self.forward_once(x0)
        out11, out12, out13, out14 = self.forward_once(x1)

        y01, y11 = self.cls1(out01), self.cls1(out11)
        y02, y12 = self.cls2(out02), self.cls2(out12)
        y03, y13 = self.cls3(out03), self.cls3(out13)
        y04 = self.cls(out04)
        y14 = self.cls(out14)

        results = {'id0':[y01,y02,y03,y04], 'id1':[y11,y12,y13,y14],
                'feature0':[out01, out02, out03, out04],
                'feature1':[out11, out12, out13, out14]}
        return results


def resnet18_id(pretrained=False,**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet_id(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained == True:
        # MedicalNet的模型在多GPU训练，dict的key多了module
        model = nn.DataParallel(model)
        pretrained_dict = torch.load('./weight/MedicalNet/pretrain/resnet_18.pth')['state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def resnet50_id(pretrained=False,**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet_id(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained == True:
        # MedicalNet的模型在多GPU训练，dict的key多了module
        model = nn.DataParallel(model)
        pretrained_dict = torch.load('./weight/MedicalNet/pretrain/resnet_50.pth')['state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    # model = resnet50_id(sample_input_D=49,
    #                 sample_input_H = 59,
    #                 sample_input_W = 47,
    #                 shortcut_type= 'B',
    #                 no_cuda= False,
    #                 num_seg_classes=2,  # 源代码default=2
    #                 num_features=128,
    #                 pretrained=True)

    model = resnet18_id(sample_input_D=49,
                sample_input_H = 59,
                sample_input_W = 47,
                shortcut_type= 'B',
                no_cuda= False,
                num_seg_classes=2,  # 源代码default=2
                num_features=128,
                pretrained=True)

    inputs0 = torch.rand(8, 1, 49, 58, 47)
    inputs1 = torch.rand(8, 1, 49, 58, 47)
    output = model(inputs0, inputs1)
    print(output['id0'][1].shape)  # float32
    print("ok")