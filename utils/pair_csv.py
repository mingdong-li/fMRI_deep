import os, sys
import numpy as np
import pandas as pd
import random


class pair_csv():
    def __init__(self,file_dir,pheno_dir):
        self.files = os.listdir(file_dir)
        self.pheno = pd.read_csv(pheno_dir)
        self.pair = np.zeros([0,5])
        self.count = 0

    def form(self):
        for i, img in enumerate(self.files):
            img0_dir = img
            img0_label = self.pheno[self.pheno['ScanDir ID']==int(img0_dir.split('.')[0])]['DX'].values
            img0_label = 0 if img0_label == 0 else 1
            if i <len(self.files)-1:
                for j in self.files[i+1:]:
                    print(self.count)
                    self.count += 1
                    img1_dir = j
                    img1_label = self.pheno[self.pheno['ScanDir ID']==int(img1_dir.split('.')[0])]['DX'].values
                    img1_label = 0 if img1_label == 0 else 1

                    same_label=-1
                    if img0_label == img1_label or img0_label*img1_label == 3:
                        same_label = 1
                    else:
                        same_label = 0
                    
                    line = np.array([img0_dir,img1_dir,same_label, img0_label, img1_label]).reshape(1,5)
                    self.pair = np.concatenate([self.pair,line],axis=0)
                    # print(img0_dir, 'vs', img1_dir)
        
        self.pair_df = pd.DataFrame(self.pair,columns=['img0','img1','same', 'img0_label', 'img1_label'])
        self.pair_df.to_csv("./data/siam/%s.csv"%mode)
        print("ok")


class val_pair_csv():
    
    def __init__(self, train_dir,val_dir, train_pheno_dir, val_pheno_dir):
        """init and form a val csv to read nii
        
        Arguments:
            train_dir {str} -- [description]
            val_dir {str} -- [description]
            pheno_dir {[type]} -- [description]
        """        
        self.train_files = os.listdir(train_dir)
        self.val_files = os.listdir(val_dir)

        self.pheno_train = pd.read_csv(train_pheno_dir)
        self.pheno_val = pd.read_csv(val_pheno_dir)

        self.pair = np.zeros([0,5])

    def form(self, support_num=2):
        for img0 in self.val_files:
            print(img0)
            img0_list = []
            img0_label = self.pheno_val[self.pheno_val['ScanDir ID']==int(img0.split('.')[0])]['DX'].values
            img0_label = 0 if img0_label == 0 else 1

            pos = []
            neg = []
            pos_labels = []
            neg_labels = []
            
            while len(pos)< support_num or len(neg) < support_num:
                img1 = random.choice(self.train_files)
                img1_label =  self.pheno_train[self.pheno_train['ScanDir ID']==int(img1.split('.')[0])]['DX'].values
                img1_label = 0 if img1_label == 0 else 1
                
                if img1_label == img0_label and len(pos)<support_num:
                    pos.append(img1)
                    pos_labels.append(img1_label)
                elif img1_label != img0_label and len(neg)<support_num:
                    neg.append(img1)
                    neg_labels.append(img1_label)

            sup_set = {'neg':neg, 'pos':pos}
            img1_label_set = {'neg':neg_labels, 'pos':pos_labels}
           
            for i, phase in enumerate(['neg', 'pos']):
                same_label_col = np.full((support_num,1), i)
                img0_col = np.full((support_num,1), img0)
                img1_col = np.array(sup_set[phase]).reshape(-1,1)

                img0_label_col = np.full((support_num,1), img0_label)
                img1_label_col = np.array(img1_label_set[phase]).reshape(-1,1)

                lines = np.concatenate([img0_col, img1_col, same_label_col, 
                                    img0_label_col, img1_label_col],axis=1)

                self.pair = np.concatenate([self.pair,lines],axis=0)
        
        self.pair_df = pd.DataFrame(self.pair,columns=['img0','img1','same', 
                                    'img0_label', 'img1_label'])
        self.pair_df.to_csv("./data/siam/val_vs_train.csv")
        print("ok")


if __name__ == '__main__':
    data = {'train':'./data/siam/train_bold', 'val':'./data/siam/val_bold'}
    pheno = {'train':'./data/siam/adhd200_pheno.csv', 
            'val':'./data/siam/adhd200_pheno.csv'}
    
    for mode in ['train', 'val']:
        if mode == 'train':
            test = pair_csv(data[mode], pheno[mode])
            test.form()
        elif mode == 'val':
            test = val_pair_csv(data['train'], data['val'], pheno['train'], pheno['val'])
            test.form()

