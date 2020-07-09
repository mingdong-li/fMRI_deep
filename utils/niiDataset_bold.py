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

sys.path.append('./')
import setting


class CreateNiiDataset(Dataset):
    def __init__(self, data_dir, pheno_dir, csv_dir,transform = None, target_transform = None):
        self.root_path = data_dir # parameter passing
        self.pheno_dir = pheno_dir
        self.csv = pd.read_csv(csv_dir, header=0)

        lines = os.listdir(self.root_path)
        lines.sort()

        imgs = []
        for line in lines:
            imgs.append(line)

        self.imgs = imgs  # 所有文件的路径
        self.pheno = pd.read_csv(self.pheno_dir, header=0)
        self.transform = transform
        self.target_transform = target_transform

        # for i in self.imgs:
        #     print(self.pheno[self.pheno['ScanDir ID']==int(i.split('.')[0])]['DX'].values)

    def __getitem__(self, idx):
        # 
        img0_path = os.path.join(self.root_path, self.csv.iat[idx,1])
        img1_path = os.path.join(self.root_path, self.csv.iat[idx,2])

        img0_id  = int(self.csv.iat[idx,1].split('.')[0][-7:])
        img1_id  = int(self.csv.iat[idx,2].split('.')[0][-7:])
        img0_class = self.pheno[self.pheno['ScanDir ID']==img0_id]['DX'].values
        img1_class = self.pheno[self.pheno['ScanDir ID']==img1_id]['DX'].values

        img0_class = 0 if img0_class==0 else 1
        img1_class = 0 if img1_class==0 else 1
        same_label = int(self.csv.iat[idx,3])

        img0 = nil.image.load_img(img0_path)
        img1 = nil.image.load_img(img1_path)
        
        data0 = img0.get_fdata()  # np.memmap
        data1 = img1.get_fdata()  # np.memmap

        # transform可以加入resample等内容
        if self.transform is not None:
            data0 = self.transform(data0)
            data1 = self.transform(data1)

        data = {}
        input0 = {}
        input1 = {}
        
        data0 = data0[np.newaxis, :, :] # add channel axis
        data0_tensor = torch.from_numpy(data0)
        data0_tensor = data0_tensor.type(torch.FloatTensor)
        input0['values'] = data0_tensor # should be a tensor in Float Tensor Type
        # input0['paths'] = [img0_path] # should be a list, with path inside

        label_tensor = torch.tensor(img0_class).type(torch.int64)
        input0['labels'] = label_tensor

        data1 = data1[np.newaxis, :, :] # add channel axis
        data1_tensor = torch.from_numpy(data1)
        data1_tensor = data1_tensor.type(torch.FloatTensor)
        input1['values'] = data1_tensor # should be a tensor in Float Tensor Type
        # input1['paths'] = [img1_path] # should be a list, with path inside

        label_tensor = torch.tensor(img1_class).type(torch.int64)
        input1['labels'] = label_tensor

        data['input0'] = input0
        data['input1'] = input1
        same_label = torch.from_numpy(np.array([same_label]))
        same_label = same_label.type(torch.FloatTensor)
        data['same'] = same_label

        return data

    def load_data(self):
        return self

    def __len__(self):
        # 是不是决定了__getitem__(self,idx)的idx范围
        return self.csv.values.shape[0]



class ValNiiDataset(Dataset):
    def __init__(self, train_dir, train_pheno_dir, 
                    val_dir, val_pheno_dir,    
                    csv_dir,transform = None, target_transform = None):
        """[summary]
        
        Arguments:
            train_dir {str} -- [nii file packs]
            train_pheno_dir {str} -- [no use]
            val_dir {str} -- [nii file packs]
            val_pheno_dir {str} -- [no use]
            csv_dir {str} -- [val_csv]
        
        Keyword Arguments:
            transform {[type]} -- [description] (default: {None})
            target_transform {[type]} -- [description] (default: {None})
        """                    

        # img0
        self.val_root_path = val_dir 
        self.val_pheno_dir = val_pheno_dir
        self.val_pheno = pd.read_csv(self.val_pheno_dir, header=0)

        # img1
        self.train_root_path = train_dir # parameter passing
        self.train_pheno_dir = train_pheno_dir
        self.train_pheno = pd.read_csv(self.train_pheno_dir, header=0)
        

        self.csv = pd.read_csv(csv_dir, header=0)

        train_lines = os.listdir(self.train_root_path)
        val_lines = os.listdir(self.val_root_path)
        
        train_lines.sort()
        val_lines.sort()
        # imgs = []

        # for line in lines:
        #     imgs.append(line)

        # self.imgs = imgs  # 所有文件的路径
        
        self.transform = transform
        self.target_transform = target_transform

        # for i in self.imgs:
        #     print(self.pheno[self.pheno['ScanDir ID']==int(i.split('.')[0])]['DX'].values)

    def __getitem__(self, idx):
        # 
        img0_path = os.path.join(self.val_root_path, self.csv.iat[idx,1])
        img1_path = os.path.join(self.train_root_path, self.csv.iat[idx,2])

        img0_id  = int(self.csv.iat[idx,1].split('.')[0][-7:])
        img1_id  = int(self.csv.iat[idx,2].split('.')[0][-7:])
        img0_class = self.val_pheno[self.val_pheno['ScanDir ID']==img0_id]['DX'].values
        img1_class = self.train_pheno[self.train_pheno['ScanDir ID']==img1_id]['DX'].values

        img0_class = 0 if img0_class==0 else 1
        img1_class = 0 if img1_class==0 else 1

        same_label = int(self.csv.iat[idx,3])

        img0 = nil.image.load_img(img0_path)
        img1 = nil.image.load_img(img1_path)
        
        data0 = img0.get_fdata()  # np.memmap
        data1 = img1.get_fdata()  # np.memmap

        # transform可以加入resample等内容
        if self.transform is not None:
            data0 = self.transform(data0)
            data1 = self.transform(data1)

        data = {}
        input0 = {}
        input1 = {}
        
        data0 = data0[np.newaxis, :, :] # add channel axis
        data0_tensor = torch.from_numpy(data0)
        data0_tensor = data0_tensor.type(torch.FloatTensor)
        input0['values'] = data0_tensor # should be a tensor in Float Tensor Type
        input0['name'] = self.csv.iat[idx,1]
        label_tensor = torch.tensor(img0_class).type(torch.int64)
        input0['labels'] = label_tensor

        data1 = data1[np.newaxis, :, :] # add channel axis
        data1_tensor = torch.from_numpy(data1)
        data1_tensor = data1_tensor.type(torch.FloatTensor)
        input1['values'] = data1_tensor # should be a tensor in Float Tensor Type
        input1['name'] = self.csv.iat[idx,2] # should be a list, with path inside
        label_tensor = torch.tensor(img1_class).type(torch.int64)
        input1['labels'] = label_tensor

        data['input0'] = input0
        data['input1'] = input1
        same_label = torch.from_numpy(np.array([same_label]))
        same_label = same_label.type(torch.FloatTensor)
        data['same'] = same_label

        return data

    def load_data(self):
        return self

    def __len__(self):
        # 是不是决定了__getitem__(self,idx)的idx范围
        return self.csv.values.shape[0]


if __name__ == '__main__':
    # opt = setting.parse_opts('bold')
    # print(opt.epoch_num)


    my_config = setting.Config('bold')

    fmri_datasets = {}

    # train_csv是针对几个机构的数据集生成的
    # 目前的bold只有peking数据，建立Dataset有没有影响（目前代码是可以运行的）# 有bug!!!有些nii没有匹配到

    fmri_datasets['train'] = CreateNiiDataset(my_config.train_dir, my_config.train_pheno_dir, my_config.train_csv)
    fmri_datasets['val'] =  ValNiiDataset(my_config.train_dir, my_config.train_pheno_dir,
                            my_config.val_dir, my_config.val_pheno_dir, my_config.val_csv)

    # # visualization the Dataloader 
    # dataloaders = {x: torch.utils.data.DataLoader(fmri_datasets[x], batch_size=my_config.batch_size,
    #             shuffle=True) # more workers may work faster
    #             for x in ['train', 'val']}


    
    train_dataloader = torch.utils.data.DataLoader(fmri_datasets['train'], 
                        batch_size = my_config.batch_size,
                        shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(fmri_datasets['val'], 
                        batch_size=10,
                        shuffle=False, num_workers=0)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    
    # visualization the Dataloader 
    for epoch in range(2):
        for i, batch_data in enumerate(dataloaders['val']):
            print(epoch, i, "inputs", batch_data['input0']['values'].data.size(), "labels", batch_data['input0']['labels'].data.size())
            # BOLD data [10, 1, 49, 58, 47, 231];fMRI影像：[1, 49, 58, 47, 231]
            
            print(batch_data['same'].shape)
            print(batch_data['input0']['name'])
            print(batch_data['input1']['name'])
            print(batch_data['input0']['labels'])
            print(batch_data['input1']['labels'])
            print(batch_data['same'])

    print('ok')