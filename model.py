import torch
from torch import nn
from models import resnet, baseline, resnet_ft_id

def generate_model(opt):
    # 目前读入的数据是[1, 49, 58, 47, 231]
    # 但现在模型是处理[1, 49, 58, 47]
    # 方案一：加lstm层，建立时序网络
    # 方案二：选择231中的某一时刻，或者平均值？

    assert opt.model in [
        'baseline',
        'baseline4D',
        'resnet',
        'resnet_ft', 
        'resnet_id', 
        'unet']

    if opt.model == 'baseline':
        model, _ = baseline.initialize_model('baseline', 512)

    if opt.model == 'resnet':
        # assert opt.model_depth in [10, 18, 34, 50, 101]
        model, _ = resnet.initialize_model('resnet18',512)
            if torch.cuda.device_count()>1:
                model = nn.DataParallel(model,device_ids=[0,1])

    if opt.model == 'resnet_ft':
        model = resnet_ft.resnet18(
                    sample_input_D = 49,
                    sample_input_H = 59,
                    sample_input_W = 47,
                    shortcut_type= 'B',
                    no_cuda= False,
                    num_seg_classes=2,  # 源代码default=2
                    num_features=128)

        model = nn.DataParallel(model)  # MedicalNet的模型在多GPU训练，dict的key多了module
        pretrained_dict = torch.load('./weight/MedicalNet/pretrain/resnet_18.pth')['state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if opt.model == 'resnet_id'；
        model = resnet_ft_id.resnet18_id(sample_input_D=49,
        sample_input_H = 59,
        sample_input_W = 47,
        shortcut_type= 'B',
        no_cuda= False,
        num_seg_classes=2,  # 源代码default=2,实际用不到分割的内容
        num_features=256,
        pretrained=True)


    # ------------------------
    # if not opt.no_cuda:
    #     if len(opt.gpu_id) > 1:
    #         model = model.cuda() 
    #         model = nn.DataParallel(model, device_ids=opt.gpu_id)
    #         net_dict = model.state_dict() 
    #     else:
    #         import os
    #         os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
    #         model = model.cuda() 
    #         model = nn.DataParallel(model, device_ids=None)
    #         net_dict = model.state_dict()


    return model
