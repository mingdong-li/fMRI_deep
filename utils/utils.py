import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import PIL.ImageOps    


def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    


fig_format = {'train':'r.-', 'val': 'b.--', 
            'val4':'b^--', 'val3':'mx--', 'val2':'co--', 'val1':'k.--',
            'avg_acc':'rD-'}
    
def show_plot(counter, y, env, factor):
    """plot loss curve
    
    Arguments:
        counter {list} -- [0,1,2,......]
        y {dict} -- {'train':list, 'val':list}
        env {str} -- 'reho' or 'falff' or 'bold'
        factor {str} -- 'X_LOSS' or 'ACCURACY'
    """
    for key, v_list in y.items():
        if len(v_list) != 0:
            plt.plot(counter, v_list, fig_format[key], label = key) 
    plt.legend()
    plt.savefig('./result/%s_%s.jpg'%(env, factor))
    plt.show()

def dict_save(dict,name):
    """save dict

    Args:
        dict (dict): to be saved dict
        name (str): name.txt
    """
    f = open("./result/%s.txt"%name,"w")
    f.write( str(dict) )
    f.close()

if __name__ == '__main__':
    
    x = [1,2,3,4,5]
    y = {'train':[2,4,3,1,3],'val':[4,2,1,5,3]}

    dict_save(y, 'test')

    xx = [1,2,3,4,5]
    yy = {'train':[],'val4':[4,2,1,5,3],'val3':[1,2,5,1,3],'val2':[2,5,1,6,2], 'avg_acc':[1,2,3,2,1]}

    show_plot(x,y,'reho')
    show_plot(xx,yy,'reho')