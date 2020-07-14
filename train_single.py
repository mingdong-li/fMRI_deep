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
from sklearn import metrics

import time
import argparse
import copy

# traditional net based on MedicalNet transfer learning 
from utils.dataset_bold import CreateDataset
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
    fmri_datasets['train'] = CreateDataset(args.train_data_dir, args.train_pheno)
    fmri_datasets['val'] =  CreateDataset(args.val_data_dir, args.val_pheno)

<<<<<<< HEAD
    dataloaders = {x: torch.utils.data.DataLoader(fmri_datasets[x], batch_size=args.batch_size,
                shuffle=True, num_workers=0, drop_last= True) # more workers may work faster
                for x in ['train','val']}
=======
    train_loader = torch.utils.data.DataLoader(fmri_datasets['train'], batch_size=args.batch_size,
                shuffle=True, num_workers=0, drop_last= True)
    val_loader = torch.utils.data.DataLoader(fmri_datasets['val'], batch_size=args.batch_size,
                shuffle=True, num_workers=0, drop_last= False)
    dataloaders = {'train': train_loader, 'val': val_loader}
>>>>>>> 89a79bb5f6412aa9387d7ad9d2db8a28aea8865c
    
    args.model = "resnet_ft"
    args.use_siam = False

    model = generate_model(args)
    dataset_sizes = {x: len(fmri_datasets[x]) for x in ['train', 'val']}
    
    criterion_nll = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    # # Decay LR by a factor of 0.1 every 20 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # train Iterate over data.
    counter = []

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

            running_loss = 0.0
            running_correct = 0
            pred_all = []
            label_all = []

            for i, data in enumerate(dataloaders[phase]):
                assert args.dimension in [3,4]
                if args.dimension == 3:
                    img0 = torch.sum(data['input0']['values'], 5)/args.input_time

                elif args.dimension == 4:
                    img0 = data['input0']['values']
                
                img0_label = data['input0']['labels']
                img0= img0.to(device)
                img0_label = img0_label.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase=='train'):
                    output = model(img0)
                    _, preds = torch.max(output, 1)
                    loss = criterion_nll(output, img0_label)

                    print("{}: Epoch number {}, batch_num {}\nCurrent loss {}\n".format
                        (phase, epoch, i, loss.item()))
                
                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                running_correct += torch.sum(preds==img0_label)

                pred_all.extend(preds.cpu().numpy().tolist())
                label_all.extend(img0_label.cpu().numpy().tolist())

            # epoch结束
            if phase == 'train':
                exp_lr_scheduler.step()


            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_correct.double()/dataset_sizes[phase]
            
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc)
            print("------------------------\n {:s} \n Epoch number {}\n loss {}\n".format
                (phase, epoch, epoch_loss)) 
            print("------------------------\n {:s} \n Epoch number {}\n |accuracy | precision | recall|\n|{} | {} | {}|\n".format
                (phase, epoch, epoch_acc,metrics.precision_score(pred_all,label_all),metrics.recall_score(pred_all,label_all)))

            
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
        show_plot(counter,loss_history, args.env, 'LOSS')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    # myconfig = Config('bold')
    args = parse_opts('bold')
    train(args)

