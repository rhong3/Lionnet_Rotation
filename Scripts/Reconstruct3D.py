# Reconstruct manual labeled images as 3D training data


import skimage.io as io
import numpy as np
import os
import pandas as pd

pdls = []
try:
    os.mkdir('../train3D')
except FileExistsError:
    pass

for i in range(1, 24):
    im = []
    lb = []
    gp = []
    for j in range(1, 8):
        image = io.imread("../stage_1_train/C1-FB323A_CSC_Rd1_{}_{}/images/C1-FB323A_CSC_Rd1_{}_{}_pad.png".format(i,j,i,j))
        im.append(image[:, :, 0])
        label = io.imread("../stage_1_train/C1-FB323A_CSC_Rd1_{}_{}/label/Combined_pad.png".format(i,j))
        lb.append(label)
        gap = io.imread("../stage_1_train/C1-FB323A_CSC_Rd1_{}_{}/label/Gap_pad.png".format(i,j))
        gp.append(gap)
    try:
        os.mkdir("../train3D/C1-FB323A_CSC_Rd1_{}".format(i))
    except FileExistsError:
        pass
    io.imsave("../train3D/C1-FB323A_CSC_Rd1_{}/image.tif".format(i), np.asarray(im).astype(np.uint8))
    io.imsave("../train3D/C1-FB323A_CSC_Rd1_{}/label.tif".format(i), np.asarray(lb).astype(np.uint8))
    io.imsave("../train3D/C1-FB323A_CSC_Rd1_{}/gap.tif".format(i), np.asarray(gp).astype(np.uint8))
    pdls.append(['fluorescence', '../inputs/train3D/C1-FB323A_CSC_Rd1_{}/image.tif'.format(i),
                 '../inputs/train3D/C1-FB323A_CSC_Rd1_{}/label.tif'.format(i),
                 '../inputs/train3D/C1-FB323A_CSC_Rd1_{}/gap.tif'.format(i),
                 1024, 1024, 7, 'C1-FB323A_CSC_Rd1_{}'.format(i)])

pdsum = pd.DataFrame(pdls, columns=['Type', 'Image', 'Label', 'Gap', 'Width', 'Height', 'Depth', 'ID'])
pdsum.to_csv('../train3D/samples.csv', index=False)



