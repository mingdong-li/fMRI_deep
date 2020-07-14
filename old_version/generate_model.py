import torch
import torch.nn as nn
import torch.nn.functional as F

from models import resnet_ft
from models import resnet_ft_id
from models import resnet 
from models import res2net
from models import res2net_id

# Contrastive LOSS 
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        # 0: same   1: diff
        # loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
        #                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        # 0: diff   1: same
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        # loss里的euclidean_distance 和 margin可能有问题，导致过拟合
        # loss_contrastive = torch.sum((label) * torch.pow(euclidean_distance, 2) +
        #                               (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def generate_model(backbone, env, GPUs=False):
    """generate different model from backbones
    
    Arguments:
        backbone {str} -- backbone name 
    
    Returns:
        nn.module -- model Network
    """    
    if backbone not in ['resnet_ft', 'resnet_id', 'resnet', 'res2net', 'res2net_id','unet']:
        raise Exception('unknow backbone Network')

    else:
        if backbone == 'resnet_ft':
            model = resnet_ft.resnet18(
                    sample_input_D = 49,
                    sample_input_H = 59,
                    sample_input_W = 47,
                    shortcut_type= 'B',
                    no_cuda= False,
                    num_seg_classes=2,  # 源代码default=2
                    num_features=128)

            model = nn.DataParallel(model)  # MedicalNet的模型在多GPU训练，dict的key多了module
            pretrained_dict = torch.load('./deep/base_siam/weights/MedicalNet/pretrain/resnet_18.pth')['state_dict']
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
        elif backbone == 'res2net':
            model = res2net.res2net50(num_classes=512)
            # pretrained_dict = torch.load('./deep/base_siam/weights/1124res2net/last_epoch99_%s.pth'%env)
            # model.load_state_dict(pretrained_dict)
            if torch.cuda.device_count()>1:
                model = nn.DataParallel(model,device_ids=[0,1])

        elif backbone == 'res2net_id':
            model = res2net_id.test_res2net50(num_classes=512)
            if torch.cuda.device_count()>1:
                model = nn.DataParallel(model,device_ids=[0,1])
            
        elif backbone == 'resnet':
            model, _ = resnet.initialize_model('resnet18',512)
            if torch.cuda.device_count()>1:
                model = nn.DataParallel(model,device_ids=[0,1])

        elif backbone == 'resnet_id':
            model = resnet_ft_id.resnet50_id(sample_input_D=49,
                sample_input_H = 59,
                sample_input_W = 47,
                shortcut_type= 'B',
                no_cuda= False,
                num_seg_classes=2,  # 源代码default=2
                num_features=256,
                pretrained=True)

        elif backbone == 'baseline':
            model = SiameseNetwork()


    model = model.cuda()
    return model


def weights_init_kaiming(net):
    """init weights kaiming for baseline
    
    Arguments:
        net {nn.module} -- Net
    
    Returns:
        nn.module -- Net
    """    
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.normal_(m.weight.data, 0.0)
            nn.init.constant_(m.bias.data, 0.0)

    return net

class SiameseNetwork(nn.Module):
    # this is the baseline for code test
    def __init__(self):
        """
        Arguments:
            nn {[type]} -- [description]
            input batch_data: size = [batch_size, 1, 61, 73, 61]
        """
        super(SiameseNetwork, self).__init__()
        
        self.layer1 = self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        


        self.cnn1 = nn.Sequential(
            nn.AvgPool3d(3),
            nn.Conv3d(in_channels=1,out_channels=4,kernel_size=(3,3,3),
                 stride=1,padding=0,dilation=1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(4),
            
            nn.Conv3d(in_channels=4,out_channels=8,kernel_size=(3,3,3),
                 stride=1,padding=0,dilation=1,groups=1,bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(8),

            nn.Conv3d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(8),
            nn.AvgPool3d(2)
            # [b_size,960]
        )

        self.fc1 = nn.Sequential(
            nn.Linear(960, 500),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            # 激活层怎么用，FC中怎么加ReLU和Dropout
            nn.Linear(500, 128))

        self.classifer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(128,2)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output_f = self.fc1(output)
        return output_f

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # 这样的话loss也要修改，不能单纯的是contrast，不然classifier参数得不到训练
        # distance = torch.abs(output1-output2)
        # pred = self.classifer(distance)

        return output1, output2


if __name__ == '__main__':
    net = generate_model('res2net_id', 'reho')
  
    # net = SiameseNetwork()
    loss = ContrastiveLoss()
    print(net.named_modules)
    print(loss)
    print("OK")

    inputs = torch.rand(8, 1, 49, 58, 47)
    inputs = inputs.cuda()
    a,b,c,d = net(inputs,inputs)
    print(a.shape)
    print(c.shape)

    
    # weights_init_kaiming(net)
    # print(net(inputs,inputs))
    # 怎么能获得每层输出的维度呢?
    # size mismatch, m1: [8 x 5120], m2: [57024 x 2048] at ..\aten\src\TH/generic/THTensorMath.cpp:961
    print("test")

