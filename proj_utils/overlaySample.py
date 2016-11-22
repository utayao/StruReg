from skimage import data, color, io, img_as_float
import numpy as np
import matplotlib.pyplot as plt


import time
from scipy import ndimage
import scipy.io as sio
import numpy as np
import deepdish as dd
from  skimage.feature import peak_local_max
from PIL import Image
import matplotlib.pyplot as plt
from local_utils import *                    
from recurrentUtils import pad_vector_grid, pad_vector_grid_sequence
from scipy.io import loadmat 
import shutil

def overlayImg(img, mask, savepath,alpha = 0.9):
    #img = img_as_float(data.camera())
    rows, cols = img.shape    
    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    color_mask[mask == 1] = [5,119,72] 
    #color_mask[170:270, 40:120] = [0, 1, 0] # Green block
    #color_mask[200:350, 200:350] = [0, 0, 1] # Blue block
    
    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))
    
    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)
    
    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    
    img_masked = color.hsv2rgb(img_hsv)
    
    # Display the output
    f, (ax0, ax1, ax2) = plt.subplots(1, 3,
                                      subplot_kw={'xticks': [], 'yticks': []})
    ax0.imshow(img, cmap=plt.cm.gray)
    ax1.imshow(color_mask)
    ax2.imshow(img_masked)
    plt.show()
    
    img_masked = np.asarray((img_masked/np.max(img_masked) ) * 255, dtype = np.uint8)
    im = Image.fromarray(img_masked)
    im.save(savepath)



def overPath(imgpath, maskpath, savepath):
    assert os.path.isfile(imgpath), 'image does not exist!'
    thisimg = np.asarray(Image.open(imgpath))    
    mask_org = np.asarray(Image.open(maskpath))
    mask = np.asarray(process_mask_paramisum(mask_org), dtype = thisimg.dtype)
    
    thisimg = RGB2GRAY(thisimg)
    thisimg = np.asarray(thisimg, dtype= np.uint8)
    mask = np.asarray(mask, dtype= np.uint8)    
    overlayImg(img = thisimg,  mask = mask, savepath = savepath, alpha = 0.85)  
    
rootfolder = 'D:\Workstation\OwnCloud\MICCAI_2016_RNNseg\MakeFigures\zizhao_Figure'
imgpath = os.path.join(rootfolder, 'HC_12_168_02.bmp')  
maskpath = os.path.join(rootfolder, 'HC_12_168_02_mask.bmp') 
savepath = os.path.join(rootfolder, 'overlaied.bmp') 
overPath(imgpath, maskpath, savepath)
         
               