# Cycle-GAN 3D preparation
import glob
import random
import os
import numpy as np
from skimage import io
from torch.utils.data import Dataset
import torch
import skimage.morphology as mph


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
            cutim = np.clip(ic, 0, 255)
            cutlb = lb[:, 256*i:256*(i+1), 256*j:256*(j+1)]
            combined = [np.concatenate((cutim, cutlb), axis=0)]
            io.imsave(dir+'/{}_{}_{}.tif'.format(bt, i*256, j*256), np.asarray(combined).astype(np.uint8))
    # random
    for m in range(rand_num):
        ht = random.randint(0, 768)
        wt = random.randint(0, 768)
        mula = 1024*random.randint(0, 2)
        mulb = 1024*random.randint(0, 2)
        ic = img[:, mula+ht:mula+ht+256, mulb+wt:mulb+wt+256]
        ic = ic / ic.max() * 255
        ic[ic < 30] = 0
        ic[ic > 80] = ic[ic > 80] * (1+ic[ic > 80]/255)
        cutim = np.clip(ic, 0, 255)
        cutlb = lb[:, mula+ht:mula+ht+256, mulb+wt:mulb+wt+256]
        combined = [np.concatenate((cutim, cutlb), axis=0)]
        io.imsave(dir + '/{}_{}_{}.tif'.format(bt, mula+ht, mulb+wt), np.asarray(combined).astype(np.uint8))


def test_sampling(root, dir):
    imlist = sorted(glob.glob(os.path.join(root + '/*.tif')))
    for m in imlist:
        im = io.imread(m)
        for i in range(4):
            for j in range(4):
                ic = im[:, 256 * i:256 * (i + 1), 256 * j:256 * (j + 1)]
                ic = ic / ic.max() * 255
                ic[ic < 30] = 0
                ic[ic > 80] = ic[ic > 80] * (1 + ic[ic > 80] / 255)
                cutim = np.clip(ic, 0, 255)
                io.imsave(dir + '/{}_{}_{}'.format(i, j, m.split('/')[-1]), np.asarray(cutim).astype(np.uint8))


def test_reassemble(dir):
    former = sorted(glob.glob(os.path.join(dir + '/*.tif')))
    imlist = []
    for ii in former:
        imlist.append((ii.split('/')[-1])[4:])
    for im in imlist:
        imf1 = np.cacatenate((io.imread(str('0_0_' + im)), io.imread(str('1_0_' + im)), io.imread(str('2_0_' + im))),
                             axis=1)
        imf2 = np.cacatenate((io.imread(str('0_1_' + im)), io.imread(str('1_1_' + im)), io.imread(str('2_1_' + im))),
                             axis=1)
        imf3 = np.cacatenate((io.imread(str('0_2_' + im)), io.imread(str('1_2_' + im)), io.imread(str('2_2_' + im))),
                             axis=1)
        imf = np.cacatenate((imf1, imf2, imf3), axis=2)
        imf = mph.remove_small_objects(imf/255, min_size=500, connectivity=2)
        imf = mph.remove_small_holes(imf, min_size=500, connectivity=2)*255
        io.imsave(dir + '/' + im, np.asarray(imf).astype(np.uint8))
    for ii in former:
        os.remove(ii)


class ImageDataset(Dataset):
    def __init__(self, root, stack):
        self.files = sorted(glob.glob(os.path.join(root + '/data/*.tif')))
        self.stack = stack

    def __getitem__(self, index):
        item_A = torch.from_numpy(io.imread(self.files[index % len(self.files)])[:, :self.stack, :, :]/255).float()
        item_B = torch.from_numpy(io.imread(self.files[index % len(self.files)])[:, self.stack:, :, :]/255).float()

        return {'Fl': item_A, 'Bn': item_B}

    def __len__(self):
        return len(self.files)


class TestDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root + '/data/test/*.tif')))

    def __getitem__(self, index):
        item = torch.from_numpy(io.imread(self.files[index % len(self.files)])/255).float()
        return {'Fl': item, 'name': self.files[index % len(self.files)]}

    def __len__(self):
        return len(self.files)