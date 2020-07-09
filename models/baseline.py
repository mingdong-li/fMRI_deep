import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial


__all__ = ['baseline', 'baseline4D']


class Baseline(nn.Module):
    def __init__(self, num_feature=128):
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8,
                kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(8)
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=32,
                kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32)
        )

        self.gap = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(32, num_feature),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_feature, 2)
        )

    def forward_once(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        concatenated = torch.cat((output1,output2),0)
        output = self.classifier(concatenated)

        return output1, output2, output


def initialize_model(model_name, num_feature):
    model_ft = None
    input_size = []

    if model_name == 'baseline':
        model = Baseline(num_feature=128)
    
    return model, input_size
