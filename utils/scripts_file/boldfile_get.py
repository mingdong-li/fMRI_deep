import os, shutil
'''
    # move the 'sfnwmrda*.nii.gz' to other folders
    # prepare for model train and validation


    # Peking:sfnwmrda1068505_session_1_rest_1.nii
    ./train/KKI/sub_id/sfnwmrdaXXXX.gz
'''

root_dir = 'F:/mingdong/research_data/adhd200'
target_train_dir = 'F:/mingdong/research_data/adhd200/adhd200_selected_train'
target_val_dir = 'F:/mingdong/research_data/adhd200/adhd200_selected_val'


def train_select(ins):
    ins_dir = root_dir + '/' + 'train' + '/' + ins
    subjects = os.listdir(ins_dir)
    for sub in subjects:
        for f in os.listdir(ins_dir + '/' + sub):
            if 'sfnwmrda' in f and '.gz' in f:
                shutil.copyfile(os.path.join(ins_dir,sub,f), os.path.join(target_train_dir, ins,sub+'.nii.gz'))
                print('%s_%s is ok'%(ins, sub))

def val_select(ins):
    ins_dir = root_dir + '/' + 'val' + '/' + ins
    subjects = os.listdir(ins_dir)
    for sub in subjects:
        for f in os.listdir(ins_dir + '/' + sub):
            if 'sfnwmrda' in f and '.gz' in f:
                shutil.copyfile(os.path.join(ins_dir,sub,f), os.path.join(target_val_dir, ins,sub+'.nii.gz'))
                print('%s_%s is ok'%(ins, sub))

if __name__ == '__main__':
    institutions = ['KKI','NeuroIMAGE', 'NYU', 'OHSU']
    for index in [0,1,2,3]:
        i = institutions[index]
        train_select(i)
        val_select(i)