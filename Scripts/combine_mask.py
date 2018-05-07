import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.io
import os
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from textwrap import wrap
np.random.seed(1234)
import scipy.misc
import matplotlib.cm as cm
from imageio import imread



STAGE1_TRAIN = "inputs/stage_1_train"
STAGE1_TRAIN_IMAGE_PATTERN = "%s/{}/images/{}.png" % STAGE1_TRAIN
STAGE1_TRAIN_MASK_PATTERN = "%s/{}/masks/*.png" % STAGE1_TRAIN

# Get image names
def image_ids_in(root_dir, ignore=['.DS_Store', 'summary.csv', 'stage1_train_labels.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids


# read in images
def read_image(image_id, space="rgb"):
    print(image_id)
    image_file = STAGE1_TRAIN_IMAGE_PATTERN.format(image_id, image_id)
    image = skimage.io.imread(image_file)
    # Drop alpha which is not used
    image = image[:, :, :3]
    if space == "hsv":
        image = skimage.color.rgb2hsv(image)
    return image

# Get image width, height and combine masks available.
def read_image_labels(image_id, space="rgb"):
    image = read_image(image_id, space = space)
    mask_file = STAGE1_TRAIN_MASK_PATTERN.format(image_id)
    masks = skimage.io.imread_collection(mask_file).concatenate()
    height, width, _ = image.shape
    num_masks = masks.shape[0]
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = 1
    try:
        os.mkdir(STAGE1_TRAIN+'/'+image_id+'/label')
    except:
        pass
    print(np.max(labels))
    scipy.misc.imsave(STAGE1_TRAIN+'/'+image_id+'/label/Combined.png', labels)
    a = imread(STAGE1_TRAIN+'/'+image_id+'/label/Combined.png')
    a = a/255

    print(np.max(a))
    return labels


train_image_ids = image_ids_in(STAGE1_TRAIN)

for im in train_image_ids:
    read_image_labels(im)

