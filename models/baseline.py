import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial


__all__ = ['baseline', 'baseline4D']


class Baseline(nn.Module):
    def __init__(self):
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv3d(in_channels=1, out_channels=8,
                kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(8)
        )

        self.layer2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv3d(in_channels=8, out_channels=32,
                kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32)
        )
        
        self.gap = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(inplace=True)
        )

    def forward_once(self, x):
        output = self.layer1(x)
        output = self.layer2(x)
        output = self.gap(x)
        output = self.fc(x)
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


def initialize_model(model_name, num_classes):
    model_ft = None
    input_size = []

    if model_name == 'baseline':
        model = Baseline()
    
    return model, input_size
