import os

class Config():
    def __init__(self, data_type):
        if data_type not in ['falff', 'reho', 'bold']:
            raise Exception("Unknown input")
        cwd = os.getcwd()

        self.env = data_type
        self.backbone = 'resnet_ft'

        self.train_dir =cwd + '/data/siam/train_%s'%(self.env)
        self.val_dir = cwd + '/data/siam/val_%s'%(self.env)

        self.train_pheno_dir =cwd+ '/data/siam/adhd200_pheno.csv'
        self.val_pheno_dir =cwd+ '/data/siam/adhd200_pheno.csv'

        self.train_csv = cwd + '/data/siam/train.csv'
        self.val_csv = cwd + '/data/siam/val.csv'
        self.val1_csv = cwd + '/data/siam/val_vs_train.csv'

        # train hyperparameters
        self.batch_size = 8
        self.train_num_epochs = 20
        self.lr = 0.001


if __name__ == '__main__':
    cwd = os.getcwd()
    print(cwd)