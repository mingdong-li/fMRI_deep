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

    # pretrained_dict = torch.load('./deep/base_siam/weights/best_res2net_id.pth')
    # model_dict = model.state_dict()
    # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

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
    accuracy = {'train':[], 'val4':[], 'val3':[], 'val2':[],'val1':[], 'avg_acc':[]}

    best_acc = 0.0
    iter_acc = 0.0
    num_epochs = args.train_num_epochs

    # num_epochs = 20
    for epoch in range(num_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
   

        running_loss = 0.0
        for i, data in enumerate(dataloaders['train']):
            model.train()
            

            img0 = data['input0']['values']
            img0_label = data['input0']['labels']
            img1 = data['input1']['values']
            img1_label = data['input1']['labels']
            same_label = data['same']

            img0, img1 , same_label = img0.cuda(), img1.cuda() , same_label.cuda()
            img0_label = img0_label.cuda()
            img1_label = img1_label.cuda()

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output = model(img0,img1)
                id0_list = output['id0']
                id1_list = output['id1']
                out_f0_list = output['feature0']
                out_f1_list = output['feature1']
                
                loss_contrastive = 0.0
                loss_nll_0 = 0.0
                loss_nll_1 = 0.0
                for j in range(len(out_f0_list)):
                    loss_contrastive += criterion1(out_f0_list[j],out_f1_list[j],same_label)
                    loss_nll_0 += criterion2(id0_list[j], img0_label)
                    loss_nll_1 += criterion2(id1_list[j], img1_label)
                
                alpha = 0.1 # shrink parameter to balance loss
                loss = alpha * loss_contrastive + loss_nll_0 + loss_nll_1
                print("Epoch number {}, batch_num {}\n Current loss {}\n".format
                    (epoch, i, loss.item()))
            
                loss.backward()
                optimizer.step()

            running_loss +=loss.item()

            if i%100 == 1:
                model.eval()
                val_loss= 0.0
                running_correct4 = 0
                running_correct3 = 0
                running_correct2 = 0
                running_correct1 = 0
                for v_index, data in enumerate(dataloaders['val']):
                    img0 = data['input0']['values']
                    img0_label = data['input0']['labels']
                    img1 = data['input1']['values']
                    img1_label = data['input1']['labels']
                    same_label = data['same']

                    img0, img1 , same_label = img0.cuda(), img1.cuda() , same_label.cuda()
                    img0_label = img0_label.cuda()
                    img1_label = img1_label.cuda()
                    with torch.set_grad_enabled(False):
                        output = model(img0,img1)
                        id0_list = output['id0']
                        id1_list = output['id1']
                        out_f0_list = output['feature0']
                        out_f1_list = output['feature1']

                        loss_contrastive = 0.0
                        loss_nll_0 = 0.0
                        loss_nll_1 = 0.0
                        for v_j in range(len(out_f0_list)):
                            loss_contrastive += criterion1(out_f0_list[v_j],out_f1_list[v_j],same_label)
                            loss_nll_0 += criterion2(id0_list[v_j], img0_label)
                            loss_nll_1 += criterion2(id1_list[v_j], img1_label)
                        
                        # alpha = 0.1 # shrink parameter to balance loss
                        loss_v = alpha * loss_contrastive + loss_nll_0 + loss_nll_1
                        print("VAL: Epoch number {}, batch_num {}\n Current loss {}\n".format
                            (epoch, v_index, loss_v.item()))
                        val_loss +=loss_v.item()

                    # 计算accuracy
                    labels = data['input0']['labels']
                    labels = labels.cuda()
                    out = model(img0,img1)
                    _, preds4 = torch.max(out['id0'][3], 1)
                    _, preds3 = torch.max(out['id0'][2], 1)
                    _, preds2 = torch.max(out['id0'][1], 1)
                    _, preds1 = torch.max(out['id0'][0], 1)
                    running_correct4 += (preds4 == labels).sum().item()
                    running_correct3 += (preds3 == labels).sum().item()
                    running_correct2 += (preds2 == labels).sum().item()
                    running_correct1 += (preds1 == labels).sum().item()
                
                iter_acc4 = float(running_correct4)/len(fmri_datasets['val'])
                iter_acc3 = float(running_correct3)/len(fmri_datasets['val'])
                iter_acc2 = float(running_correct2)/len(fmri_datasets['val'])
                iter_acc1 = float(running_correct1)/len(fmri_datasets['val'])
                acc_list = [iter_acc1,iter_acc2,iter_acc3,iter_acc4]
                
                counter.append(i)
                loss_history['train'].append(loss.item())
                loss_history['val'].append(val_loss/len(dataloaders['val']))

                
                
                for k, key in enumerate(['val1','val2','val3','val4']):
                    accuracy[key].append(acc_list[k])

                iter_acc = sum(acc_list)/4
                accuracy['avg_acc'].append(iter_acc)
                print('epoch {:d} val Acc: {:4f}'.format(epoch, iter_acc))

                if iter_acc > best_acc:
                    best_acc = iter_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), './deep/base_siam/weights/best_%s.pth'%(args.backbone))
            
    
    show_plot(range(len(accuracy['val1'])),accuracy, '%s_val_acc_%s'%(args.env, args.backbone))
    show_plot(range(len(accuracy['val1'])),loss_history, '%s_loss_%s'%(args.env, args.backbone))
    torch.save(model.state_dict(), './deep/base_siam/weights/last_epoch%s_%s_%s.pth'%(epoch, args.env, args.backbone))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    myconfig = Config('reho')
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