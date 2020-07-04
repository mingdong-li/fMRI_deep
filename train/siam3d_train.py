import sys, os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.utils

import numpy as np
import pandas as pd
import random
import nilearn as nil

sys.path.append("./deep/base_siam")
from config import Config
from utils import imshow, show_plot
from model import SiameseNetwork, weights_init_kaiming
from model import ContrastiveLoss, generate_model
from niiDataset import CreateNiiDataset, ValNiiDataset

import time
import argparse
import copy


# same_label:  0: not same  1：same 
# label:       0: TD        1: ADHD
# Conv3d的规定输入数据格式为(batch, channel, Depth, Height, Width)

# def preds2label(pred_same, img1):
#     """to detect the img0 label
    
#     Arguments:
#         pred_same {tensor bs * 1} -- same_label of predict
#         img1 {tensor bs * 1} -- img1_label 训练集中用于val过程，进行对比的样本
    
#     Returns:
#         tensor bs*1 -- predict DX
#     """   
#     img1_inverse = torch.where((img1!=0), torch.full_like(img1,0) , torch.full_like(img1,1))
#     img1 = torch.where((img1==0),img1, torch.full_like(img1,1))
#     predict_labels = torch.where((pred_same==1),img1, img1_inverse)
#     return predict_labels


def main(args):
    """train the SiameseNet for ADHD detection
    
    Arguments:
        args {parser.parse_args()} -- for cmd run, add some arguments
    """
    fmri_datasets = {}
    fmri_datasets['train'] = CreateNiiDataset(args.train_dir, args.train_pheno_dir, args.train_csv_dir)
    # fmri_datasets['val'] =  CreateNiiDataset(args.val_dir, args.val_pheno_dir, args.val_csv_dir)
    
    # val*1_vs_train*5
    fmri_datasets['val'] =  ValNiiDataset(args.train_dir, args.train_pheno_dir,
                            args.val_dir, args.val_pheno_dir, args.val_train_dir)

    dataloaders = {x: torch.utils.data.DataLoader(fmri_datasets[x], batch_size=args.batch_size,
                shuffle=True, num_workers=0) # more workers may work faster
                for x in ['train','val']}

    # train_dataloader = torch.utils.data.DataLoader(fmri_datasets['train'], 
    #                     batch_size = args.batch_size,
    #                     shuffle=True, num_workers=0)
    # val_dataloader = torch.utils.data.DataLoader(fmri_datasets['val'], 
    #                     batch_size=10,
    #                     shuffle=True, num_workers=0)
    # dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # baseline test
    # model = SiameseNetwork().cuda()
    # model = weights_init_kaiming(model)
        
    model = generate_model(args.backbone, args.env)

    # check parameters (visualization)
    # paras = dict(model.named_parameters())
    # for k, v in paras.items():
    #     print(k.ljust(30), str(v.shape).ljust(30), 'bias:', v.requires_grad)

    dataset_sizes = {x: len(fmri_datasets[x]) for x in ['train', 'val']}
    criterion1 = ContrastiveLoss()
    criterion2 = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(),lr = args.lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(),lr = args.lr)

    
    # set lr according to the layers
    # paras_new = []
    # for k, v in paras.items():
    #     if 'layer' in k or 'conv1' in k or 'bn1' in k:
    #         paras_new += [{'params': [v], 'lr': 0.001, 'weight_decay': 0}]
    #     else:
    #         paras_new += [{'params': [v], 'lr': 0.01, 'weight_decay': 0.00004}]
    # optimizer = optim.SGD(paras_new, momentum=0.9, lr = args.lr)

    # Decay LR by a factor of 0.1 every 20 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # train Iterate over data.
    counter = []
    loss_history = {'train':[], 'val':[]}
    accuracy = {'train':[], 'val':[]}

    best_acc = 0.0
    epoch_acc = 0.0
    num_epochs = args.train_num_epochs

    # num_epochs = 20
    for epoch in range(num_epochs):
        counter.append(epoch)
        since = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0
            # phase = 'val' # 测试val_accuracy用
            for i, data in enumerate(dataloaders[phase]):
                img0 = data['input0']['values']
                img0_label = data['input0']['labels']
                img1 = data['input1']['values']
                img1_label = data['input1']['labels']
                same_label = data['same']

                img0, img1 , same_label = img0.cuda(), img1.cuda() , same_label.cuda()
                img0_label = img0_label.cuda()
                img1_label = img1_label.cuda()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    out_f0,out_f1, out_id0, out_id1 = model(img0,img1)
                    loss_contrastive = criterion1(out_f0,out_f1,same_label)
                    loss_nll_0 = criterion2(out_id0, img0_label)
                    loss_nll_1 = criterion2(out_id1, img1_label)
                    alpha = 1 # shrink parameter to balance loss
                    loss = alpha * loss_contrastive + loss_nll_0 + loss_nll_1
                    
                    print("Epoch number {}, batch_num {}\n Current loss {}\n".format
                        (epoch, i, loss.item()))
                
                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                running_loss +=loss.item()

                if phase == 'val':
                    # 计算accuracy
                    labels = data['input0']['labels']
                    labels = labels.cuda()
                    _, _, out_id0, out_id1 = model(img0,img1)
                    _, preds = torch.max(out_id0, 1)
                    running_correct += (preds == labels).sum().item()

            epoch_loss = running_loss/len(dataloaders[phase])
            print("------------------------\n {:s} \n Epoch number {}\n Current loss {}\n".format
                (phase, epoch, epoch_loss))
            
            loss_history[phase].append(epoch_loss)
            
            time_elapsed = time.time() - since
            print('{:s} : Epoch {:d} complete in {:.0f}m {:.0f}s'.format(
                phase, epoch,time_elapsed // 60, time_elapsed % 60))

            if phase == 'train':
                exp_lr_scheduler.step()

            if phase == 'val':
                epoch_acc = float(running_correct)/len(fmri_datasets[phase])
                accuracy[phase].append(epoch_acc)
                print('epoch {:d} val Acc: {:4f}'.format(epoch, epoch_acc))
                # if epoch_acc > best_acc:
                    # best_acc = epoch_acc
                    # best_model_wts = copy.deepcopy(model.state_dict())
                    # torch.save(model.state_dict(), './deep/base_siam/weights/best_base_siam.pth')
    
    show_plot(counter,accuracy, args.env)
    show_plot(counter,loss_history, args.env)
    # show_plot(counter,accuracy)
    torch.save(model.state_dict(), './deep/base_siam/weights/last_epoch%s_%s.pth'%(epoch, args.env))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    myconfig = Config('falff')
    # add some arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=myconfig.env)
    parser.add_argument('--backbone', type=str, default = myconfig.backbone)

    parser.add_argument("--train_dir", type=str, default = myconfig.train_dir)
    parser.add_argument("--val_dir", type=str, default = myconfig.val_dir)

    parser.add_argument("--train_pheno_dir", type=str, default = myconfig.train_pheno_dir)
    parser.add_argument("--val_pheno_dir", type=str, default = myconfig.val_pheno_dir)

    parser.add_argument("--train_csv_dir", type=str, default = myconfig.train_csv)
    parser.add_argument("--val_csv_dir", type=str, default = myconfig.val_csv)
    parser.add_argument("--val_train_dir", type=str, default = myconfig.val1_csv)

    parser.add_argument("--batch_size", type=int, default = myconfig.batch_size)
    parser.add_argument("--lr", type=float, default = myconfig.lr)
    parser.add_argument('--train_num_epochs', type=int, default=myconfig.train_num_epochs)

    args = parser.parse_args()

    # main running 
    main(args)