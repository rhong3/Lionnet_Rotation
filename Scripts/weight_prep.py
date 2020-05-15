# Calculate weight, emphasize boundary weight
import skimage.io
import scipy
import scipy.misc
import scipy.ndimage
from skimage import color
import os
import numpy as np

STAGE1_TRAIN = "../stage_1_train"
STAGE1_TRAIN_IMAGE_PATTERN = "%s/{}/images/{}.png" % STAGE1_TRAIN
STAGE1_TRAIN_MASK_PATTERN = "%s/{}/masks/*.png" % STAGE1_TRAIN


# Get image names
def image_ids_in(root_dir, ignore=['.DS_Store', 'summary.csv', 'stage1_train_labels.csv', 'vsamples.csv', 'stage1_solution.csv', 'samples.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore or '.csv' in id:
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


def get_weight(image_id, space="rgb"):
    image = read_image(image_id, space=space)
    height, width, _ = image.shape
    mask_file = STAGE1_TRAIN_MASK_PATTERN.format(image_id)
    masks = skimage.io.imread_collection(mask_file).concatenate()
    massls = []
    for i in masks:
        mask = i/255
        labeled_array, num_features = scipy.ndimage.label(mask)
        if num_features == 1:
            mass = scipy.ndimage.measurements.center_of_mass(mask)
            massls.append([int(mass[0]), int(mass[1])])
        else:
            print(image_id)
            print(num_features)
    massmp = np.ones((height, width), np.uint16)
    for index in range(0, len(massls)):
        massmp[massls[index][0], massls[index][1]] = 0
    weightmap = scipy.ndimage.morphology.distance_transform_edt(massmp)
    weightmap = (weightmap-np.max(weightmap))**2
    scipy.misc.imsave(STAGE1_TRAIN + '/' + image_id + '/label/WT_Combined.png', weightmap)


train_image_ids = image_ids_in(STAGE1_TRAIN)

for im in train_image_ids:
    get_weight(im)

