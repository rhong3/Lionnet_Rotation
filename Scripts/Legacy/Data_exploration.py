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

# This code is used to get initial summary information of images and using kmeans to classify them into
# different categories.

STAGE1_TRAIN = "inputs/stage_2_test"
STAGE1_TRAIN_IMAGE_PATTERN = "%s/{}/images/{}.png" % STAGE1_TRAIN
STAGE1_TRAIN_MASK_PATTERN = "%s/{}/masks/*.png" % STAGE1_TRAIN
IMAGE_ID = "image_id"
IMAGE_WIDTH = "width"
IMAGE_WEIGHT = "height"
HSV_CLUSTER = "hsv_cluster"
HSV_DOMINANT = "hsv_dominant"
TOTAL_MASK = "total_masks"


# Get image names
def image_ids_in(root_dir, ignore=['.DS_Store', 'trainset_summary.csv', 'stage2_train_labels.csv', 'samples.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids

# read image
def read_image(image_id, space="rgb"):

    image_file = STAGE1_TRAIN_IMAGE_PATTERN.format(image_id, image_id)
    image = skimage.io.imread(image_file)
    print(image.shape)
    # Drop alpha which is not used
    if len(image.shape) != 3:
        image = np.resize(image, (image.shape[0], image.shape[1], 3))
    else:
        image = image[:, :, :3]
    print(image.shape)
    if space == "hsv":
        image = skimage.color.rgb2hsv(image)
    return image


# Get image width, height.
def read_image_labels(image_id, space="rgb"):
    image = read_image(image_id, space = space)
    height, width, _ = image.shape
    labels = np.zeros((height, width), np.uint16)
    return image, labels


# Load image identifiers.
train_image_ids = image_ids_in(STAGE1_TRAIN)


def get_domimant_colors(img, top_colors=2):
    img_l = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    clt = KMeans(n_clusters = top_colors)
    clt.fit(img_l)
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    return clt.cluster_centers_, hist

# Get image size, HSV values
def get_images_details(image_ids):
    details = []
    for image_id in image_ids:
        print(image_id)
        image_hsv, labels = read_image_labels(image_id, space="hsv")
        height, width, l = image_hsv.shape
        dominant_colors_hsv, dominant_rates_hsv = get_domimant_colors(image_hsv, top_colors=1)
        dominant_colors_hsv = dominant_colors_hsv.reshape(1, dominant_colors_hsv.shape[0] * dominant_colors_hsv.shape[1])
        info = (image_id, width, height, dominant_colors_hsv.squeeze())
        details.append(info)
    return details


META_COLS = [IMAGE_ID, IMAGE_WIDTH, IMAGE_WEIGHT]
COLS = META_COLS + [HSV_DOMINANT]


details = get_images_details(train_image_ids)


trainPD = pd.DataFrame(details, columns=COLS)
X = (pd.DataFrame(trainPD[HSV_DOMINANT].values.tolist())).as_matrix()
# Kmeans to classify images based on HSV values
kmeans = KMeans(n_clusters=3).fit(X)
clusters = kmeans.predict(X)
trainPD[HSV_CLUSTER] = clusters

# Save to a summary csv
trainPD.to_csv(STAGE1_TRAIN+'/summary.csv', header = True, index = False)

# plot image
def plot_images(images, images_rows, images_cols, imname):
    f, axarr = plt.subplots(images_rows,images_cols,figsize=(16,images_rows*2))
    for row in range(images_rows):
        for col in range(images_cols):
            image_id = images[row*images_cols + col]
            image = read_image(image_id)
            height, width, l = image.shape
            ax = axarr[row,col]
            ax.axis('off')
            ax.set_title("%dx%d"%(width, height))
            ax.imshow(image)
    f.savefig(imname)


plot_images(trainPD[trainPD[HSV_CLUSTER] == 0][IMAGE_ID].values, 7, 8)


plot_images(trainPD[trainPD[HSV_CLUSTER] == 1][IMAGE_ID].values, 2, 8)


plot_images(trainPD[trainPD[HSV_CLUSTER] == 2][IMAGE_ID].values, 2, 8)


P = trainPD.groupby(HSV_CLUSTER)[IMAGE_ID].count().reset_index()
# Get each group percentage
P['Percentage'] = 100*P[IMAGE_ID]/P[IMAGE_ID].sum()


f, ax = plt.subplots(1,1,figsize=(16,5))
r = trainPD.plot(kind="hist", bins=300, y = TOTAL_MASK, ax=ax, grid=True, title="Masks Histogram")


