'''
Configs for training & testing
'''

import argparse


class Config():
    def __init__(self, data_type):
        if data_type not in ['falff', 'reho', 'bold']:
            raise Exception("Unknown input")
        # cwd = os.getcwd()

        self.env = data_type
        self.backbone = 'resnet_ft'

        self.train_dir ='./datasets/siam/train_%s'%(self.env)
        self.val_dir = './datasets/siam/val_%s'%(self.env)

        self.train_pheno_dir ='./datasets/siam/adhd200_pheno.csv'
        self.val_pheno_dir ='./datasets/siam/adhd200_pheno.csv'

        self.train_csv = './datasets/siam/train.csv'
        self.val_csv = './datasets/siam/val.csv'
        self.val1_csv = './datasets/siam/val_vs_train.csv'

        # train hyperparameters
        self.batch_size = 8
        self.train_num_epochs = 20
        self.lr = 0.001

def parse_opts(env):
    assert env in ['falff', 'reho', 'bold']
    parser = argparse.ArgumentParser()


    # train hyperparameters
    self.batch_size = 8
    self.train_num_epochs = 20
    self.lr = 0.001

     parser.add_argument('--batch_size',
        default=32,
        type=int,
        help='batch size')

    parser.add_argument('--epoch_num',
        default=32,
        type=int,
        help='epoch num')

    parser.add_argument('--lr',
        default=0.001,
        type=float,
        help='learning rate')

    parser.add_argument('--train_data_root',
        default='./datasets/siam/train_%s'%(env),
        type=str,
        help='Root directory path of train data')

    parser.add_argument('--val_data_root',
        default='./datasets/siam/train_%s'%(env),
        type=str,
        help='Root directory path of val data')

    parser.add_argument('--train_pheno',
        default='./datasets/siam/adhd200_pheno.csv',
        type=str,
        help='csv file of ADHD200_train_pheno')

    parser.add_argument('--val_pheno',
        default='./datasets/siam/adhd200_pheno.csv',
        type=str,
        help='csv file of ADHD200_val_pheno')
    
    parser.add_argument('--train_list',
        default='./datasets/siam/train.csv',
        type=str,
        help='csv file of train_list')

    parser.add_argument('--val_list',
        default='./datasets/siam/val.csv',
        type=str,
        help='csv file of val_list')

    return parser

    

