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


class CreateDataset(Dataset):
    def __init__(self, data_dir, pheno_dir, dimension=3):
        self.root_path = data_dir # parameter passing
        self.pheno_dir = pheno_dir

        lines = os.listdir(self.root_path)
        lines.sort()

        imgs = []
        for line in lines:
            imgs.append(line)

        self.imgs = imgs  # 所有文件的路径
        self.pheno = pd.read_csv(self.pheno_dir, header=0)
        self.dimension  = dimension

        # for i in self.imgs:
        #     print(self.pheno[self.pheno['ScanDir ID']==int(i.split('.')[0])]['DX'].values)

    def __getitem__(self, idx):
        # 
        img0_path = os.path.join(self.root_path, self.imgs[idx])
        img0_id = int(self.imgs[idx].split('.')[0])
        img0_class = self.pheno[self.pheno['ScanDir ID']==img0_id]['DX'].values

        img0_class = 0 if img0_class==0 else 1

        img0 = nil.image.load_img(img0_path)
        data0 = img0.get_fdata()  # np.memmap
        input_time = data0.shape[-1]

        data = {}
        input0 = {}
        # input1 = {}
        
        data0 = data0[np.newaxis, :] # add channel axis
        data0_tensor = torch.from_numpy(data0)
        data0_tensor = data0_tensor.type(torch.FloatTensor)
        input0['name'] = img0_id
        input0['values'] = data0_tensor # should be a tensor in Float Tensor Type
        label_tensor = torch.tensor(img0_class).type(torch.int64)
        input0['labels'] = label_tensor

        data['input0'] = input0

        if self.dimension == 3:
            data['input0']['values'] = torch.sum(data['input0']['values'], -1)/input_time

        elif self.dimension == 4:
            data['input0']['values'] = data['input0']['values']

        return data

    def load_data(self):
        return self

    def __len__(self):
        # 是不是决定了__getitem__(self,idx)的idx范围
        return len(self.imgs)


if __name__ == '__main__':
    # opt = setting.parse_opts('bold')
    # print(opt.epoch_num)


    my_config = setting.Config('bold')

    fmri_datasets = {}

    # train_csv是针对几个机构的数据集生成的
    # 目前的bold只有peking数据，建立Dataset有没有影响（目前代码是可以运行的）# 有bug!!!有些nii没有匹配到

    fmri_datasets['train'] = CreateDataset(my_config.train_dir, my_config.train_pheno)
    fmri_datasets['val'] =  CreateDataset(my_config.val_dir, my_config.val_pheno)

    
    train_dataloader = torch.utils.data.DataLoader(fmri_datasets['train'], 
                        batch_size = my_config.batch_size,
                        shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(fmri_datasets['val'], 
                        batch_size= my_config.batch_size,
                        shuffle=True, num_workers=0)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    
    # visualization the Dataloader 
    for epoch in range(2):
        for i, batch_data in enumerate(dataloaders['val']):
            print(epoch, i) 
            print("inputs", batch_data['input0']['values'].data.size()) 
            print("labels", batch_data['input0']['labels'].data.size())
            # BOLD data [bs, 1, 49, 58, 47, 231];fMRI影像：[1, 49, 58, 47, 231]
            print(batch_data['input0']['name'])
            print(batch_data['input0']['labels'])

    print('ok')