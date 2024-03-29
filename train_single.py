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
from torch.utils.tensorboard import SummaryWriter       

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
    writer = SummaryWriter("./tensorboard/runs")


    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    
    fmri_datasets = {}
    fmri_datasets['train'] = CreateDataset(args.train_data_dir, args.train_pheno, dimension=3)
    fmri_datasets['val'] =  CreateDataset(args.val_data_dir, args.val_pheno, dimension=3)

    train_loader = torch.utils.data.DataLoader(fmri_datasets['train'], batch_size=args.batch_size,
                shuffle=True, num_workers=0, drop_last= False)
    val_loader = torch.utils.data.DataLoader(fmri_datasets['val'], batch_size=args.batch_size,
                shuffle=True, num_workers=0, drop_last= False)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    args.model = "resnet_ft"
    args.use_siam = False

    model = generate_model(args)
    dataset_sizes = {x: len(fmri_datasets[x]) for x in ['train', 'val']}
    
    criterion_nll = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    # # Decay LR by a factor of 0.1 every 20 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    # train Iterate over data.
    counter = []

    loss_history = {'train':[], 'val':[]}
    acc_history = {'train':[], 'val':[]}
    pre_history = {'train':[], 'val':[]} # precision
    rec_history = {'train':[], 'val':[]} # recall


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

            # 不同机构测量的时间不一样
            for i, data in enumerate(dataloaders[phase]):
                assert args.dimension in [3,4]

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
                writer.add_scalar("Loss/train", running_loss, epoch)
                writer.add_scalar("Accuracuy/train", metrics.accuracy_score(pred_all,label_all), epoch)
                writer.add_scalar("Precision/train", metrics.precision_score(pred_all,label_all), epoch)
                writer.add_scalar("Recall/train", metrics.recall_score(pred_all,label_all), epoch)

            if phase == 'val':
                writer.add_scalar("Loss/val", running_loss, epoch)
                writer.add_scalar("Accuracuy/val", metrics.accuracy_score(pred_all,label_all), epoch)
                writer.add_scalar("Precision/val", metrics.precision_score(pred_all,label_all), epoch)
                writer.add_scalar("Recall/val", metrics.recall_score(pred_all,label_all), epoch)

            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = metrics.precision_score(pred_all,label_all)
            epoch_pre = metrics.accuracy_score(pred_all,label_all)
            epoch_rec = metrics.recall_score(pred_all,label_all)

            # 保存记录
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc)
            pre_history[phase].append(epoch_pre)
            rec_history[phase].append(epoch_rec)

            
            print("------------------------\n {:s} \nEpoch number {}\n loss {}\n".format
                (phase, epoch, epoch_loss)) 
            print("------------------------\n {:s} \nEpoch number {}\n |accuracy | precision | recall|\n|{} | {} | {}|\n".format
                (phase, epoch, epoch_acc,
                metrics.precision_score(pred_all,label_all), metrics.recall_score(pred_all,label_all)))

            
            time_elapsed = time.time() - since
            print('{:s} : Epoch {:d} complete in {:.0f}m {:.0f}s'.format(
                phase, epoch,time_elapsed // 60, time_elapsed % 60))

            # save every epoch
            if epoch%20 ==0:
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

    args = parse_opts('bold')
    train(args)

