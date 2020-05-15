import numpy as np # linear algebra
import skimage.io
import os
import sys
np.random.seed(1234)
import scipy.misc
import skimage.morphology as mph
from skimage import color

dd = sys.argv[1]

STAGE1_TRAIN = "../inputs/"+dd
STAGE1_TRAIN_IMAGE_PATTERN = "%s/{}/images/{}.png" % STAGE1_TRAIN
STAGE1_TRAIN_MASK_PATTERN = "%s/{}/masks/*.png" % STAGE1_TRAIN

# Get image names
def image_ids_in(root_dir, ignore=['.DS_Store', 'summary.csv', 'stage1_train_labels.csv', 'vsamples.csv', 'stage1_solution.csv', 'samples.csv']):
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
    mkk = []
    for i in masks:
        mask = i/255
        selem = mph.disk(1)
        mask = mph.erosion(mask, selem)
        mkk.append(mask)
    mkk = np.asarray(mkk)
    height, width, _ = image.shape
    num_masks = masks.shape[0]
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[mkk[index] > 0] = 1
    try:
        os.mkdir(STAGE1_TRAIN+'/'+image_id+'/label')
    except:
        pass
    scipy.misc.imsave(STAGE1_TRAIN+'/'+image_id+'/label/ER_Combined.png', labels)
    return labels

train_image_ids = image_ids_in(STAGE1_TRAIN)

for im in train_image_ids:
    read_image_labels(im)

