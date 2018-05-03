
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import imageio
import sys
import cv2
from skimage import color


# In[2]:


def rescale_im(directory):
    dirs = os.listdir(directory)
    size_summary_raw = pd.read_csv(directory+'/summary.csv')
    size_summary = size_summary_raw.iloc[:,1:3]
    min_size = size_summary.apply( max, axis=1 ).min()
    min_unit = 2**np.ceil(np.log2(min_size))
    for imid in dirs:
        if len(imid)<np.max([len(i) for i in dirs]):
            continue
        ## raw image, both in gray channel or RGB channel
        im = imageio.imread(directory+imid+'/images/'+imid+'.png')
        ## convert RGB to gray
        if len(im.shape) == 3:
            im = im[:, :, :3]
        im = color.rgb2gray(im)
        im = im / im.max() * 255
        tosize = np.ceil(np.max(im.shape)/min_unit)*min_unit
        if size_summary_raw.loc[size_summary_raw['image_id'] == imid, 'hsv_cluster'].values == 1:
            im = 255 - im
        im_c = (im - im.mean())
        im_c[im_c < 0] = 0
        im = (im_c / im_c.max() * 255).astype(np.uint8)

        if tosize == im.shape[0] == im.shape[1]:
            im_pad = np.empty((int(tosize), int(tosize), 3))
            im_pad[:,:,0] = im_pad[:,:,1] = im_pad[:,:,2] = im
        else:
            im_pad = np.empty((int(tosize), int(tosize), 3))
            row_copy = int(np.ceil((tosize / im.shape[0] - 1)/2))
            col_copy = int(np.ceil((tosize / im.shape[1] - 1)/2))
            #print(im.shape)
            #print(row_copy, col_copy)
            for i in range(3):
                im_i = im
                top_left = top_right = bottom_left = bottom_right = np.rot90(np.rot90(im_i))
                mid_left = mid_right = np.fliplr(im_i)
                top_mid = bottom_mid = np.flipud(im_i)
                for j in range(col_copy):
                    top_mid = np.concatenate((top_left, top_mid, top_right), axis=1)
                    im_i = np.concatenate((mid_left, im_i, mid_right), axis=1)
                    bottom_mid = np.concatenate((bottom_left, bottom_mid, bottom_right), axis=1)
                    top_left = top_right = bottom_left = bottom_right = np.fliplr(top_right)
                    mid_left = mid_right = np.fliplr(mid_right)
                for k in range(row_copy):
                    im_i = np.concatenate((top_mid, im_i, bottom_mid), axis=0)
                    top_mid = bottom_mid = np.flipud(top_mid)
                #print(im_i.shape)
                row_size_left = int((im_i.shape[0] - tosize) // 2)
                row_size_right = int((im_i.shape[0] - tosize) // 2 + (im_i.shape[0] - tosize) % 2)
                col_size_left = int((im_i.shape[1] - tosize) // 2)
                col_size_right = int((im_i.shape[1] - tosize) // 2 + (im_i.shape[1] - tosize) % 2)
                if row_size_right == 0 and col_size_right == 0:
                    im_i = im_i[row_size_left:, col_size_left:]
                elif row_size_right == 0:
                    im_i = im_i[row_size_left:, col_size_left:-col_size_right]
                elif col_size_right == 0:
                    im_i = im_i[row_size_left:-row_size_right, col_size_left:]
                else:
                    im_i = im_i[row_size_left:-row_size_right, col_size_left:-col_size_right]
                im_pad[:,:,i] = im_i
        imageio.imsave(directory+imid+'/images/'+imid+'_pad.png', im_pad.astype(np.uint8))
        
        
        ## labels
        try:
            im = imageio.imread(directory+imid+'/label/'+'Combined.png')
            tosize = np.ceil(np.max(im.shape)/min_unit)*min_unit
            if tosize == im.shape[0] == im.shape[1]:
                pass
            else:
                row_copy = int(np.ceil((tosize / im.shape[0] - 1) / 2))
                col_copy = int(np.ceil((tosize / im.shape[1] - 1) / 2))
                top_left = top_right = bottom_left = bottom_right = np.rot90(np.rot90(im))
                mid_left = mid_right = np.fliplr(im)
                top_mid = bottom_mid = np.flipud(im)
                for j in range(col_copy):
                    top_mid = np.concatenate((top_left, top_mid, top_right), axis=1)
                    im = np.concatenate((mid_left, im, mid_right), axis=1)
                    bottom_mid = np.concatenate((bottom_left, bottom_mid, bottom_right), axis=1)
                    top_left = top_right = bottom_left = bottom_right = np.fliplr(top_right)
                    mid_left = mid_right = np.fliplr(mid_right)
                for k in range(row_copy):
                    im = np.concatenate((top_mid, im, bottom_mid), axis=0)
                    top_mid = bottom_mid = np.flipud(top_mid)
                row_size_left = int((im.shape[0] - tosize) // 2)
                row_size_right = int((im.shape[0] - tosize) // 2 + (im.shape[0] - tosize) % 2)
                col_size_left = int((im.shape[1] - tosize) // 2)
                col_size_right = int((im.shape[1] - tosize) // 2 + (im.shape[1] - tosize) % 2)
                if row_size_right == 0 and col_size_right == 0:
                    im = im[row_size_left:, col_size_left:]
                elif row_size_right == 0:
                    im = im[row_size_left:, col_size_left:-col_size_right]
                elif col_size_right == 0:
                    im = im[row_size_left:-row_size_right, col_size_left:]
                else:
                    im = im[row_size_left:-row_size_right, col_size_left:-col_size_right]

            imageio.imsave(directory+imid+'/label/'+'Combined_pad.png', im.astype(np.uint8))
        except:
            continue



# In[ ]:


dir_path = sys.argv[1]
rescale_im(dir_path)

