# Cycle-GAN 3D preparation
import glob
import random
import os
import numpy as np
from skimage import io
from torch.utils.data import Dataset
import torch


def construct(root):
    imlist = random.sample(glob.glob(root + '/*/'), 9)
    ct = 1
    imcombined = None
    lbcombined = None
    for i in imlist:
        im = io.imread(i+'image.tif')
        lb = io.imread(i+'label.tif')
        if ct == 1:
            imcombined = im
            lbcombined = lb
        else:
            imcombined = np.concatenate((imcombined, im), axis=1)
            lbcombined = np.concatenate((lbcombined, lb), axis=1)
        ct += 1
    ic = np.concatenate((imcombined[:, :3072, :], imcombined[:, 3072:6144, :], imcombined[:, 6144:, :]), axis=2)
    il = np.concatenate((lbcombined[:, :3072, :], lbcombined[:, 3072:6144, :], lbcombined[:, 6144:, :]), axis=2)

    return ic, il


def sampling(img, lb, bt, dir, rand_num=56):
    # original images
    for i in range(12):
        for j in range(12):
            ic = img[:, 256*i:256*(i+1), 256*j:256*(j+1)]
            ic = ic / ic.max() * 255
            ic[ic < 30] = 0
            ic[ic > 80] = ic[ic > 80] * (1+ic[ic > 80]/255)
            ic = np.clip(ic, 0, 255)
            cutim = [ic]
            cutlb = [lb[:, 256*i:256*(i+1), 256*j:256*(j+1)]]
            io.imsave(dir+'/{}_{}_{}_im.tif'.format(bt, i*256, j*256), np.asarray(cutim).astype(np.uint8))
            io.imsave(dir + '/{}_{}_{}_lb.tif'.format(bt, i*256, j*256), np.asarray(cutlb).astype(np.uint8))
    # random
    for m in range(rand_num):
        ht = random.randint(0, 2048)
        wt = random.randint(0, 2048)
        ic = img[:, ht:ht+256, wt:wt+256]
        ic = ic / ic.max() * 255
        ic[ic < 30] = 0
        ic[ic > 80] = ic[ic > 80] * (1+ic[ic > 80]/255)
        ic = np.clip(ic, 0, 255)
        cutim = [ic]
        cutlb = [lb[:, ht:ht+256, wt:wt+256]]
        io.imsave(dir + '/{}_{}_{}_im.tif'.format(bt, ht, wt), np.asarray(cutim).astype(np.uint8))
        io.imsave(dir + '/{}_{}_{}_lb.tif'.format(bt, ht, wt), np.asarray(cutlb).astype(np.uint8))


class ImageDataset(Dataset):
    def __init__(self, root, unaligned=False):
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root + '/data/*_im.tif')))
        self.files_B = sorted(glob.glob(os.path.join(root + '/data/*_lb.tif')))

    def __getitem__(self, index):
        item_A = torch.from_numpy(io.imread(self.files_A[index % len(self.files_A)])).long()
        if self.unaligned:
            item_B = torch.from_numpy(io.imread(self.files_B[random.randint(0, len(self.files_B) - 1)])).long()
        else:
            item_B = torch.from_numpy(io.imread(self.files_B[index % len(self.files_B)])).long()

        return {'Fl': item_A, 'Bn': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

