# Prepare manual labeled images as training data

import skimage.io as io
import numpy as np
import os
import skimage.morphology as mph
import pandas as pd

sum = pd.read_csv("../stage_1_train/summary.csv",
                  usecols=["image_id", "width", "height", "total_masks", "hsv_dominant", "hsv_cluster"])
new = []

for i in range(1, 8):
    image = io.imread("../C1-FB323A_CSC_Rd1 #0{}.tif".format(str(i)))
    label = io.imread("../ManualNuclei_C2-FB323A_CSC_Rd1 #0{}.tif".format(str(i)))
    for j in range(np.shape(image)[0]):
        try:
            os.mkdir('../stage_1_train/C1-FB323A_CSC_Rd1_{}_{}'.format(str(i), str(j+1)))
        except FileExistsError:
            pass
        try:
            os.mkdir('../stage_1_train/C1-FB323A_CSC_Rd1_{}_{}/images'.format(str(i), str(j+1)))
        except FileExistsError:
            pass
        try:
            os.mkdir('../stage_1_train/C1-FB323A_CSC_Rd1_{}_{}/label'.format(str(i), str(j+1)))
        except FileExistsError:
            pass

        io.imsave('../stage_1_train/C1-FB323A_CSC_Rd1_{}_{}/images/C1-FB323A_CSC_Rd1_{}_{}.png'.
                  format(str(i), str(j+1), str(i), str(j+1)), image[j, :, :])
        io.imsave('../stage_1_train/C1-FB323A_CSC_Rd1_{}_{}/label/Combined_pad.png'.
                  format(str(i), str(j+1), str(i), str(j+1)), label[j, :, :])

        selem = mph.disk(2)
        erlabel = mph.erosion(label[j, :, :], selem)
        io.imsave('../stage_1_train/C1-FB323A_CSC_Rd1_{}_{}/label/ER_Combined_pad.png'.
                  format(str(i), str(j + 1), str(i), str(j + 1)), erlabel)

        selem = mph.disk(2)
        dllabel = mph.dilation(label[j, :, :], selem)
        io.imsave('../stage_1_train/C1-FB323A_CSC_Rd1_{}_{}/label/DL_Combined.png'.
                  format(str(i), str(j + 1), str(i), str(j + 1)), dllabel)

        gap = dllabel - erlabel
        io.imsave('../stage_1_train/C1-FB323A_CSC_Rd1_{}_{}/label/Gap_pad.png'.
                  format(str(i), str(j + 1), str(i), str(j + 1)), gap)

        new.append(['C1-FB323A_CSC_Rd1_{}_{}'.format(str(i), str(j + 1)), 1024, 1024, 0, 0, 0])


newpd = pd.DataFrame(new, columns=["image_id", "width", "height", "total_masks", "hsv_dominant", "hsv_cluster"])
newpd = pd.concat([sum, newpd])
newpd.to_csv("../stage_1_train/summary.csv", index=True)
