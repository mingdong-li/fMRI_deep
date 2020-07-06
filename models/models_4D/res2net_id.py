import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv3d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool3d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv3d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
            bns.append(nn.BatchNorm3d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv3d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out.shape = ([8, 104, 13, 15, 13])
        # self.width = 26
        spx = torch.split(out, self.width, 1)
        # spx = (a,b,c,d)  [8, 26, 13, 15, 13]  4个这样的tensor， 按照channel分割

        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Res2Net(nn.Module):
    # 可以筛选后卷积层load MedicalNet中res2net和res2net相同层参数?
    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=1000):
        # num_class是feature的维数
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.layer_maxpool = nn.AdaptiveMaxPool3d(1)
        # self.layer_maxpool = nn.MaxPool3d(kernel_size=3,stride=2, padding=1)

        # self.avgpool = nn.AvgPool3d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * block.expansion, num_classes*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(num_classes*2, num_classes)
        )
        self.cls1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(num_classes,2)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128 * block.expansion, num_classes*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(num_classes*2, num_classes),
        )
        self.cls2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(num_classes,2)
        )
           
        self.fc3 = nn.Sequential(
            nn.Linear(256 * block.expansion, num_classes*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(num_classes*2, num_classes)
        )
        self.cls3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(num_classes,2)
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, num_classes*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(num_classes*2, num_classes)
        )
        self.cls = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(num_classes,2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward_once(self, x):
        # old for res2net not res2net_id
        # need to be overwrite
        x = self.conv1(x)  # [bs, 64, 26, 30, 25]
        x = self.bn1(x)    # 
        x = self.relu(x)
        x = self.maxpool(x)# [8, 64, 13, 15, 13]

        x = self.layer1(x) # [8, 256, 13, 15, 13]
        x = self.layer2(x) # [8, 512, 7, 8, 7]
        x = self.layer3(x) # [8, 1024, 4, 4, 4]
        x = self.layer4(x) # [8, 2048, 2, 2, 2]

        x = self.avgpool(x)# [8, 2048, 1, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward(self,x1,x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)

        return output1,output2


class test_Res2Net(Res2Net):
    def forward_once(self, x):
        x = self.conv1(x)  # [bs, 64, 26, 30, 25]
        x = self.bn1(x)    # 
        x = self.relu(x)
        x = self.maxpool(x)# [8, 64, 13, 15, 13]

        x = self.layer1(x) # [8, 256, 13, 15, 13]
        x1 = self.layer_maxpool(x)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1(x1)
        
        x = self.layer2(x) # [8, 512, 7, 8, 7]
        x2 = self.layer_maxpool(x)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc2(x2)
        
        x = self.layer3(x) # [8, 1024, 4, 4, 4]
        x3 = self.layer_maxpool(x)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc3(x3)
        
        x4 = self.layer4(x) # [8, 2048, 2, 2, 2]
        x = self.avgpool(x4)# [8, 2048, 1, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x1,x2,x3,x

    def forward(self,x1,x2):
        out01, out02, out03, out04 = self.forward_once(x1)
        out11, out12, out13, out14 = self.forward_once(x2)

        y01, y11 = self.cls1(out01), self.cls1(out11)
        y02, y12 = self.cls2(out02), self.cls2(out12)
        y03, y13 = self.cls3(out03), self.cls3(out13)
        y04 = self.cls(out04)
        y14 = self.cls(out14)

        results = {'id0':[y01,y02,y03,y04], 'id1':[y11,y12,y13,y14],
                'feature0':[out01, out02, out03, out04],
                'feature1':[out11, out12, out13, out14]}
        return results

def test_res2net50(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = test_Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model

if __name__ == '__main__':
    inputs0 = torch.rand(8, 1, 49, 58, 47)
    inputs1 = torch.rand(8, 1, 49, 58, 47)
    import os
    reho_list = os.listdir('./data/siam/train_reho')
    import random
    random_rc = random.choice(reho_list)
    import nilearn as nil
    from nilearn.image import new_img_like, load_img
    test = load_img(os.path.join('./data/siam/train_reho/',random_rc))
    test = torch.tensor(test.get_data())
    inputs0[0,0,:,:,:] = test
    inputs1[0,0,:,:,:] = test


    model = test_res2net50(num_classes=512)
    
    output = model(inputs0, inputs1)
    print(output['id0'][1].shape)  # float32
    label = torch.tensor([1,0,1,0,1,1,0,1]).type(torch.int64)

    # test预测accuracy用
    # _, pred1=torch.max(c1.data,1)
    # _, pred2=torch.max(c2.data,1)
    # print(pred1.shape)

    # with torch.set_grad_enabled(False):
    #     loss = nn.CrossEntropyLoss()
    #     print(loss(c1, label))

    # similarity = torch.reshape(F.pairwise_distance(output1,output2), (-1,1))
    # print(similarity)
