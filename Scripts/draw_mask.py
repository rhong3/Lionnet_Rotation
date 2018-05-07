import pandas as pd
import numpy as np
import imageio
import sys
import os

# This code is used to generate validation labels from ground truth csv file Kaggle provided

def draw_mask(csvfile, directory):
    '''Take in a run-length csv file where each line is a mask in one image and outputs a label image of all masks
    combined within each image.'''
    label_csv = pd.read_csv(csvfile)
    ims = label_csv.groupby('ImageId')["ImageId"].unique()
    ## each image
    for i in ims:
        coords = []
        im_label = label_csv.loc[label_csv['ImageId'] == i[0]].reset_index(drop=True)
        im_id = im_label['ImageId'][0]
        masks = im_label['EncodedPixels']
        rows = im_label["Height"][0]
        cols = im_label["Width"][0]
        ## each mask
        for j in masks:
            ## disect a run-length pixel encoding into individual pixels
            if not os.path.exists(directory+'/'+im_id+'/masks/'):
                os.makedirs(directory+'/'+im_id+'/masks/')
            coords_ea = []
            mask = j.split(' ')
            starts = [int(mask[2 * k]) for k in range(int(len(mask) / 2))]
            stretches = [int(mask[2 * k + 1]) for k in range(int(len(mask) / 2 - 1))]
            for k in range(len(stretches)):
                row = starts[k] % rows - 1
                col = starts[k] // rows
                coords.append([row, col])
                coords_ea.append([row, col])
                for stretch in range(stretches[k]):
                    row += 1
                    if row >= rows:
                        row -= rows
                        col += 1
                    coords.append([row, col])
                    coords_ea.append([row, col])
            ## single pixel masks with no stretches
            if len(starts) > len(stretches):
                for k in range(len(stretches), len(starts)):
                    row = starts[k] % rows - 1
                    col = starts[k] // rows
                    coords.append([row, col])
            im_mask_ea = np.zeros((rows, cols), dtype='uint8')
            for coord in coords_ea:
                im_mask_ea[coord[0], coord[1]] = 255
            ## outputs images with each single mask
            imageio.imsave(directory+'/'+im_id+'/masks/mask_'+str(starts[0])+'.png', im_mask_ea)

        im_mask = np.zeros((rows, cols), dtype='uint8')
        for coord in coords:
            im_mask[coord[0], coord[1]] = 255
        if not os.path.exists(directory+'/'+im_id+'/label/'):
            os.makedirs(directory+'/'+im_id+'/label/')
        ## outputs images with all masks combined
        imageio.imsave(directory+'/'+im_id+'/label/Combined.png', im_mask)



CSVfile = sys.argv[1]
Directory = sys.argv[2]
draw_mask(CSVfile, Directory)



