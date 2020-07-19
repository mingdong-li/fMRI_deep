import os, shutil
'''
    # move the 'sfnwmrda*.nii.gz' to other folders
    # prepare for model train and validation


    # Peking:sfnwmrda1068505_session_1_rest_1.nii
    ./train/KKI/sub_id/sfnwmrdaXXXX.gz
'''

root_dir = 'F:/research_data'
target_train_dir = 'F:/research_data/adhd200_selected_train'
target_val_dir = 'F:/research_data/adhd200_selected_val'


def train_select(ins):
    ins_dir = root_dir + '/' + 'train' + '/' + ins
    subjects = os.listdirins_dir
    for sub in subjects:
        for f in os.listdir(ins_dir + '/' + sub):
            if 'sfnwmrda' in f and '.gz' in f:
                shutil.copyfile(os.path.join(ins_dir,sub,f), os.path.join(target_train_dir, ins,sub+'.nii.gz'))

def val_select(ins):
    ins_dir = root_dir + '/' + 'val' + '/' + ins
    subjects = os.listdir(ins_dir)
    for sub in subjects:
        for f in os.listdir(ins_dir '/' + f):
            if 'sfnwmrda' in f and '.gz' in f:
                shutil.copyfile(os.path.join(ins_dir,sub,f), os.path.join(target_train_dir, ins,sub+'.nii.gz'))

if __name__ == '__main__':
    institutions = ['KKI', 'NYU']
    i = institutions[0]
    train_select(i)
    val_select(i)