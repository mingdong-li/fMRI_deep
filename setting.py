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

        self.train_pheno ='./datasets/siam/adhd200_pheno.csv'
        self.val_pheno ='./datasets/siam/adhd200_pheno.csv'

        self.train_csv = './datasets/siam/train_bold.csv'
        self.val_csv = './datasets/siam/val_bold.csv'

        # train hyperparameters
        self.batch_size = 8
        self.train_num_epochs = 20
        self.lr = 0.001

def parse_opts(env):
    assert env in ['falff', 'reho', 'bold']
    parser = argparse.ArgumentParser()

    parser.add_argument("--env",
        default = env,
        type = str,
        help='data env')

    parser.add_argument("--model",
        default='baseline',
        type=str,
        help='backbone select')

    parser.add_argument("--use_siam",
        default=False,
        type=bool,
        help='backbone select')

    parser.add_argument('--dimension',
        default=3,
        type=int,
        help='data dimension')

    # CUDA
    # parser.add_argument(
    #     '--no_cuda', action='store_true', help='If true, cuda is not used.')
    # parser.set_defaults(no_cuda=False)
    # parser.add_argument(
    #     '--gpu_id',
    #     nargs='+',
    #     type=int,              
    #     help='Gpu id lists')


    # train hyperparameters
    parser.add_argument("--batch_size",
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

    # img info
    # [D, H, W, time]
    parser.add_argument('--input_D',
        default=49,
        type=int,
        help='sample_input_D')

    parser.add_argument('--input_H',
        default=53,
        type=int,
        help='sample_input_H')

    parser.add_argument('--input_W',
        default=47,
        type=int,
        help='sample_input_W')

    parser.add_argument('--input_time',
        default=231,
        type=int,
        help='sample_input_time')

    # data path
    parser.add_argument('--train_data_dir',
        default='./datasets/siam/train_%s'%(env),
        type=str,
        help='Root directory path of train data')

    parser.add_argument('--val_data_dir',
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
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = parse_opts('bold')
    print('ok')

    

