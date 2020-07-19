import os, shutil

def move(env,old_train,old_test):
    if env not in ['falff', 'reho']:
        raise InterruptedError

    # old_train = './data/ADHD200/falff_reho/%s_train'%env
    # old_test = './data/ADHD200/falff_reho/%s_test/Peking_1'%env

    new_train = './data/siam/train_%s'%env
    new_test = './data/siam/val_%s' %env

    for file_path in [(old_train,new_train), (old_test,new_test)]:
        subs = os.listdir(file_path[0])
        for sub in subs:
            for j in os.listdir(os.path.join(file_path[0],sub)):
                if j.split('.')[-1] == 'gz' and 'rest_1' in j:
                    shutil.copyfile(os.path.join(file_path[0],sub,j),os.path.join(file_path[1],sub+'.nii.gz'))

def move_bold(env):
    """
    move BOLD into a folder
    """
    print("test")
    if env not in ['bold']:
        raise InterruptedError

    old_train = './data/ADHD200/bold/%s_pk_train'%env
    old_test = './data/ADHD200/bold/%s_pk_test'%env

    new_train = './data/siam/train_%s'%env
    new_test = './data/siam/val_%s' %env

    for file_path in [(old_train,new_train), (old_test,new_test)]:
        subs = os.listdir(file_path[0])
        for sub in subs:
            for j in os.listdir(os.path.join(file_path[0],sub)):
                # sfnwmrdaXXXXXXX_session_1_rest_1.nii.gz
                if j.split('.')[-1] == 'gz' and j.split('.')[0][0:2]=='sf':
                    shutil.copyfile(os.path.join(file_path[0],sub,j),os.path.join(file_path[1],sub+'.nii.gz'))


if __name__ == '__main__':
    # ['Peking', 'KKI'. 'OHSU', 'NYU', 'NeuroIMAGE']
    institute = 'NYU'
    # ['falff', 'reho', 'bold']
    env = 'reho'
    old_train = './data/ADHD200/falff_reho/%s_train/%s'%(env, institute)
    old_test = './data/ADHD200/falff_reho/%s_test/%s'%(env, institute)
    
    # move_bold('bold')
    move(env, old_train, old_test)
