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
from utils.utils import imshow, show_plot

from model_4d import ContrastiveLoss, generate_model
from niiDataset import CreateNiiDataset, ValNiiDataset

import time
import argparse
import copy


# same_label:  0: not same  1：same 
# label:       0: TD        1: ADHD
# Conv3d的规定输入数据格式为(batch, channel, Depth, Height, Width)

def main(args):
    """train the SiameseNet for ADHD detection
    
    Arguments:
        args {parser.parse_args()} -- for cmd run, add some arguments
    """
    fmri_datasets = {}
    fmri_datasets['train'] = CreateNiiDataset(args.train_dir, args.train_pheno_dir, args.train_csv_dir)
    # fmri_datasets['val'] =  CreateNiiDataset(args.val_dir, args.val_pheno_dir, args.val_csv_dir)
    
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
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda" if use_cuda else "cpu")
        
    model = generate_model('resnet_ft') 
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

    # Decay LR by a factor of 0.1 every step_size epoch
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # train Iterate over data.
    counter = []
    loss_history = {'train':[], 'val':[]}
    accuracy = {'train':[], 'val':[]}

    best_acc = 0.0
    epoch_acc = 0.0
    num_epochs = args.train_num_epochs

    for epoch in range(num_epochs):
        # 需要加入对于时间序列的分割输入！
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0
            train_correct = 0
            val_correct = 0
            # phase = 'val' # 测试val_accuracy用
            for i, data in enumerate(dataloaders[phase]):
                img0 = data['input0']['values'][:,:,:,:,:,100:116]
                img0_label = data['input0']['labels']
                img1 = data['input1']['values'][:,:,:,:,:,100:116]
                img1_label = data['input1']['labels']
                same_label = data['same']

                img0, img1 , same_label = img0.cuda(), img1.cuda() , same_label.cuda()
                img0_label = img0_label.cuda()
                img1_label = img1_label.cuda()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    output0, output1 = model(img0,img1)
                    f0, c0 = output0
                    f1, c1 = output1
                    
                    loss_contrastive = 0.0
                    loss_nll_0 = 0.0
                    loss_nll_1 = 0.0
                    loss_contrastive = criterion1(f0, f1, same_label)
                    loss_nll_0 = criterion2(c0, img0_label)
                    loss_nll_1 = criterion2(c1, img1_label)
                    
                    alpha = 0.5 # shrink parameter to balance loss
                    loss = alpha * loss_contrastive + loss_nll_0 + loss_nll_1
                    _, preds = torch.max(c0, 1)
                    running_correct = (preds == img0_label).sum().item()
                    print("{}: Epoch number {}, batch_num {}\n Current loss {} \n Current acc {}".format
                        (phase, epoch, i, loss.item(), float(running_correct)/(len(img0_label))))
                
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                        train_correct += running_correct

                    if phase == 'val':
                        val_correct += running_correct

                running_loss +=loss.item()
                # if phase == 'val':
                #     # 计算accuracy
                #     # 怎么把代码改成每隔K个iterate输出一次验证集
                #     labels = data['input0']['labels']
                #     labels = labels.cuda()
                #     out0, _ = model(img0,img1)
                #     _, preds = torch.max(out0[1], 1)
                #     val_correct += (preds == labels).sum().item()

            epoch_loss = running_loss/len(dataloaders[phase])
            print("------------------------\n {:s} \n Epoch number {}\n LOSS {}\n".format
                (phase, epoch, epoch_loss))
            
            loss_history[phase].append(epoch_loss)
            time_elapsed = time.time() - since
            print('{:s} : Epoch {:d} complete in {:.0f}m {:.0f}s'.format(
                phase, epoch,time_elapsed // 60, time_elapsed % 60))

            if phase == 'train':
                exp_lr_scheduler.step()
                epoch_acc = float(train_correct)/len(fmri_datasets[phase])
                accuracy[phase].append(epoch_acc)
                print('train epoch {:d} train Acc: {:4f}'.format(epoch, epoch_acc))

            if phase == 'val':
                epoch_acc = float(val_correct)/len(fmri_datasets[phase])
                counter.append(epoch)
                accuracy[phase].append(epoch_acc)
                print('val epoch {:d} val Acc: {:4f}'.format(epoch, epoch_acc))
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), './deep/base_siam/weights/best_%s.pth'%(args.backbone))
    
    show_plot(counter,accuracy, '%s_val_acc_%s'%(args.env, args.backbone))
    show_plot(counter,loss_history, '%s_loss_%s'%(args.env, args.backbone))
    torch.save(model.state_dict(), './deep/base_siam/weights/last_epoch%s_%s_%s.pth'%(epoch, args.env, args.backbone))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    myconfig = Config('bold')
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