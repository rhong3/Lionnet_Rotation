import tifffile
import os
import numpy as np


# Get all images in the root directory
def image_ids_in(root_dir, ignore=['.DS_Store', 'dict.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            print(id.split('.ls')[0])
            ids.append(id.split('.ls')[0])
    return ids


images = image_ids_in('20190607/')
os.mkdir('20190607/tiffs')

for i in images:
    im = tifffile.imread('20190607/{}.lsm'.format(i))
    print(i)
    if np.shape(im)[1] == 1:
        tif = im[0, 0, :, :, :]
        print(np.shape(tif))
    else:
        tif = im[0, :, 0, :, :]
        print(np.shape(tif))
    tifffile.imsave('20190607/tiffs/{}.tif'.format(i), tif)

