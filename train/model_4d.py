import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from models_4D import resnet_ft
from models_4D import resnet_ft_id
from models_4D import resnet 
from models_4D import res2net
from models_4D import res2net_id
# 基于4-D数据的模型

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


def generate_model(backbone, GPUs=False):
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
                    num_features=128,
                    seq_len=16)

            model = nn.DataParallel(model)  # MedicalNet的模型在多GPU训练，dict的key多了module
            pretrained_dict = torch.load('./deep/base_siam/weights/MedicalNet/pretrain/resnet_18.pth')['state_dict']
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    model = model.cuda()
    return model


if __name__ == '__main__':
    import numpy as np
    import nilearn as nil
    from nilearn.image import new_img_like, load_img
    from nilearn import image,plotting

    # ab_dir = 'e:/master_project/fMRI/data/siam/train_bold'
    # bold_list = os.listdir(ab_dir)
    # b = load_img(os.path.join(ab_dir,bold_list[0]))
    # print(b.shape)
    # data = b.get_fdata()

    c_loss = ContrastiveLoss()
    model = generate_model('resnet_ft')
    inputs = torch.randn(4,1,49,58,47,16)
    inputs = inputs.cuda()
    # p0 = inputs[:,:,:,:,:,0:16]
    # p1 = inputs[:,:,:,:,:,16:32]
    o0,o1 = model(inputs, inputs)
    print(c_loss(o0[1].cpu(),o1[1].cpu(), torch.Tensor([[0],[1],[0],[1]])))
    print('ok')

