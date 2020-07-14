import os, sys 

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

import time
import argparse
import copy

from utils.niiDataset_bold import CreateNiiDataset, ValNiiDataset
from utils.contrastive_loss import ContrastiveLoss
from utils.utils import show_plot
from setting import parse_opts
from model import generate_model

# same_label:  0: not same  1：same 
# label:       0: TD        1: ADHD
# Conv3d的规定输入数据格式为(batch, channel, Depth, Height, Width)




def train(args):
    """train the SiameseNet for ADHD detection
    
    Arguments:
        args {parser.parse_args()} -- for cmd run, add some arguments
    """

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    
    fmri_datasets = {}
    fmri_datasets['train'] = CreateNiiDataset(args.train_data_dir, args.train_pheno, args.train_list)

    # val*1_vs_train*2
    fmri_datasets['val'] =  ValNiiDataset(args.train_data_dir, args.train_pheno,
                            args.val_data_dir, args.val_pheno, args.val_list)

    dataloaders = {x: torch.utils.data.DataLoader(fmri_datasets[x], batch_size=args.batch_size,
                shuffle=True, num_workers=0) # more workers may work faster
                for x in ['train','val']}

    model = generate_model(args)
    dataset_sizes = {x: len(fmri_datasets[x]) for x in ['train', 'val']}
    
    criterion_con = ContrastiveLoss()
    criterion_nll = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    # # Decay LR by a factor of 0.1 every 20 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # train Iterate over data.
    counter = []
    nll_loss_history = {'train':[], 'val':[]}
    con_loss_history = {'train':[], 'val':[]}
    loss_history = {'train':[], 'val':[]}
    acc_history = {'train':[], 'val':[]}

    best_acc = 0.0
    epoch_acc = 0.0
    num_epochs = args.epoch_num


    for epoch in range(num_epochs):
        counter.append(epoch)
        since = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_nllloss = 0.0
            running_conloss = 0.0
            running_loss = 0.0
            running_correct = 0

            for i, data in enumerate(dataloaders[phase]):
                assert args.dimension in [3,4]
                if args.dimension == 3:
                    img0 = torch.sum(data['input0']['values'], 5)/args.input_time
                    img1 = torch.sum(data['input1']['values'], 5)/args.input_time

                elif args.dimension == 4:
                    img0 = data['input0']['values']
                    img1 = data['input1']['values']
                
                same_label = data['same']
                img0, img1 = img0.to(device), img1.to(device)
                same_label = same_label.view((-1,))
                same_label = same_label.to(device)

                # img0_label = data['input0']['labels']
                # img1_label = data['input1']['labels']
                # img0_label, img1_label = img0_label.to(device), img1_label.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase=='train'):
                    out_f0,out_f1, out_same = model(img0,img1)
                    _, preds = torch.max(out_same, 1)

                    loss_con = criterion_con(out_f0,out_f1,same_label)
                    loss_nll = criterion_nll(out_same, same_label.long())
                    alpha = 1 # shrink parameter to balance loss
                    loss = alpha * loss_con + loss_nll

                    print("Epoch number {}, batch_num {}\nCurrent loss {}\n".format
                        (epoch, i, loss.item()))
                
                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                running_conloss += loss_con.item()
                running_nllloss += loss_nll.item()
                running_loss += loss.item()
                running_correct += torch.sum(preds==same_label.long())



            # epoch结束
            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_nllloss = running_nllloss/dataset_sizes[phase]
            epoch_conloss = running_conloss/dataset_sizes[phase]
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_correct.double()/dataset_sizes[phase]
            
            con_loss_history[phase].append(epoch_conloss)
            nll_loss_history[phase].append(epoch_nllloss)
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc)
            print("------------------------\n {:s} \n Epoch number {}\n Current loss {}\n".format
                (phase, epoch, epoch_loss)) 
            print("------------------------\n {:s} \n Epoch number {}\n Current contrast accuracy {}\n".format
                (phase, epoch, epoch_acc))
            
            time_elapsed = time.time() - since
            print('{:s} : Epoch {:d} complete in {:.0f}m {:.0f}s'.format(
                phase, epoch,time_elapsed // 60, time_elapsed % 60))

            # save every epoch
            torch.save(model.state_dict(), './weight/save_train/epoch%s_%s.pth'%(epoch, args.env))
            # save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), './weight/save_train/best_model.pth')


        show_plot(counter,acc_history, args.env, 'ACCURACY')
        show_plot(counter,con_loss_history, args.env, 'loss_con')
        show_plot(counter,nll_loss_history, args.env, 'loss_nll')
        show_plot(counter,loss_history, args.env, 'LOSS')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    # myconfig = Config('bold')
    args = parse_opts('bold')
    train(args)

