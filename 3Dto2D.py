import numpy as np
import skimage as ski
import os


# Get all images in the root directory
def image_ids_in(root_dir, ignore=['.DS_Store', 'dict.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            print(id.split('.t')[0])
            ids.append(id.split('.t')[0])
    return ids


images = image_ids_in('Images/')

for i in images:
    im = ski.io.imread('Images/{}.tif'.format(i))
    os.mkdir('Images/{}'.format(i))
    for a in range(np.shape(im)[0]):
        print(a)
        sgim = im[a,:,:,:]
        ski.io.imsave('Images/{}/{}_{}.tif'.format(i, str(a), i), sgim)
        os.rename('Images/{}/{}_{}.tif'.format(i, str(a), i), 'Images/{}/{}_{}.png'.format(i, str(a), i))
print('done!')

