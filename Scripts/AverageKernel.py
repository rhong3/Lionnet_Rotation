# After prediction, for each prediction pixel, take average with its 6 nearby pixels (up, down, left, right, front, back) then reconstruct the binary image.
import numpy as np  # linear algebra
from skimage import io
import sys
import os
import pickle
from scipy import ndimage

mode = sys.argv[1]

def image_ids_in(root_dir, ignore=['.DS_Store', 'trainset_summary.csv', 'stage2_train_labels.csv', 'samples.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids


def dataloader(mode):
    try:
        with open(mode + '/out.pickle', 'rb') as f:
            images = pickle.load(f)
    except:
        handles = image_ids_in(mode)
        images = {}
        images['Image'] = []
        images['ID'] = []
        # images['Dim'] = []
        for i in handles:
            im = io.imread(mode + '/'+i)
            images['Image'].append(im)
            j = i.split('.')[0]
            # io.imsave('Images/' +'norm_'+ j + '.tif', im)
            images['ID'].append(j)

        with open(mode + '/out.pickle', 'wb') as f:
            pickle.dump(images, f)
        with open(mode + '/out.pickle', 'rb') as f:
            images = pickle.load(f)
    return images


def average(img, output, teid):
    result = ndimage.generic_filter(img, np.nanmean, size=3, mode='constant', cval=np.NaN)
    out = result>230
    out = out.astype(np.uint8)
    io.imsave(output + '/' + teid + '.tif', ((out / out.max()) * 255).astype(np.uint8))


def test(tesample, mode):
    if not os.path.exists(mode + '/AVE'):
        os.makedirs(mode + '/AVE')
    for itr in range(len(tesample['ID'])):
        teim = tesample['Image'][itr]
        teid = tesample['ID'][itr]
        average(teim, mode + '/AVE', teid)

sample = dataloader(mode)

test(sample, mode)