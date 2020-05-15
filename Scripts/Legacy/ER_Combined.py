import os
import pandas as pd
import numpy as np
import imageio
import sys


# This code is used to normalize, correct contrast, and mirroring images.

def rescale_im(directory):
    ## load image summary table which contains image handle path, image size and RGB channel info
    dirs = os.listdir(directory)
    size_summary_raw = pd.read_csv(directory + '/summary.csv')
    ## get image size
    size_summary = size_summary_raw.iloc[:, 2:4]
    min_size = size_summary.apply(max, axis=1).min()
    ## determine the smallest size unit for all images; has to be a power of 2.
    min_unit = 2 ** np.ceil(np.log2(min_size))
    for imid in dirs:
        ## labels
        try:
            im = imageio.imread(directory + imid + '/label/' + 'ER_Combined.png')
            tosize = np.ceil(np.max(im.shape) / min_unit) * min_unit
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

            imageio.imsave(directory + imid + '/label/' + 'ER_Combined_pad.png', im.astype(np.uint8))
        except:
            continue


dir_path = sys.argv[1]
rescale_im(dir_path)

