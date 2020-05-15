import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(1234)
import sys

# This code is used to generate CSV files that contain full path to images, image sizes, and image classes.
dir_path = sys.argv[1]

# ROOT = dir_path
# ROOT_IMAGE_PATTERN = "%s/{}/images/{}.png" % ROOT
# ROOT_IMAGEPAD_PATTERN = "%s/{}/images/{}_pad.png" % ROOT
# ROOT_LABEL_PATTERN = "%s/{}/label/Combined.png" % ROOT
# ROOT_LABELPAD_PATTERN = "%s/{}/label/Combined_pad.png" % ROOT
ROOTT = dir_path
# ROOTT_IMAGE_PATTERN = "%s/{}/images/{}.png" % ROOTT
ROOTT_IMAGEPAD_PATTERN = "%s/{}/images/{}_pad.png" % ROOTT
ROOTT_GAP_PATTERN = "%s/{}/label/Gap_pad.png" % ROOTT
ROOTT_LABELPAD_PATTERN = "%s/{}/label/MU_Combined_pad.png" % ROOTT


def read_lite(summary, mode, root, root_IMAGEPAD_PATTERN, root_LABELPAD_PATTERN, root_GAP_PATTERN):
    sample = []
    for index, row in summary.iterrows():
        ls = []
        color = row['hsv_cluster']
        W = row['width']
        H = row['height']
        id = row['image_id']
        if color == 0:
            type = 'fluorescence'
        elif color == 1:
            type = 'histology'
        elif color == 2:
            type = 'light'
        image_path = root_IMAGEPAD_PATTERN.format(id, id)
        label_path = root_LABELPAD_PATTERN.format(id)
        gap_path = root_GAP_PATTERN.format(id)
        ls.append(type)
        ls.append(image_path)
        if mode == 'train':
            ls.append(label_path)
            ls.append(gap_path)
        ls.append(W)
        ls.append(H)
        ls.append(id)
        sample.append(ls)
    if mode == 'train':
        df = pd.DataFrame(np.array(sample), columns=['Type', 'Image', 'Label', 'Gap', 'Width', 'Height', 'ID'])
    else:
        df = pd.DataFrame(np.array(sample), columns=['Type', 'Image', 'Width', 'Height', 'ID'])
    return df



# train = pd.read_csv('../inputs/stage_1_train/summary.csv', header = 0)
# trsample = read_lite(train, 'train', ROOT, ROOT_IMAGEPAD_PATTERN, ROOT_LABELPAD_PATTERN)
# Sample = shuffle(trsample)
# trsample, vasample = np.split(Sample.sample(frac=1), [int(0.8*len(Sample))])
# trsample.to_csv('../inputs/stage_1_train/trsamples.csv', index = False, header = True)
# vasample.to_csv('../inputs/stage_1_train/vasamples.csv', index = False, header = True)


test = pd.read_csv(ROOTT+'/summary.csv', header = 0)
tesample = read_lite(test, 'train', ROOTT, ROOTT_IMAGEPAD_PATTERN, ROOTT_LABELPAD_PATTERN, ROOTT_GAP_PATTERN)
tesample.to_csv(ROOTT+'/musamples.csv', index=False, header=True)