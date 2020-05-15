import matplotlib
matplotlib.use('Agg')
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# This code is used to generate distribution of positive/negative ratio of training set and validation set

np.random.seed(1234)
import torch
from torch.autograd import Variable
from imageio import imread, imsave
from torch.nn import functional as F
from torch.nn import init
from skimage.morphology import label
import matplotlib.pyplot as plt
import sys
import os
import pickle
import time

# Output folder
output = sys.argv[1]

if not os.path.exists('../' + output):
    os.makedirs('../' + output)

with open('../inputs/mirror_roll/train.pickle', 'rb') as f:
    images = pickle.load(f)

list1 = []

# get label positive and negative ratio
for itr in range(len(images['Label'])):
    trla = images['Label'][itr]
    trlaa = trla[0:1, :, :, :]
    label_ratio = (trlaa > 0).sum() / (trlaa.shape[1] * trlaa.shape[2] * trlaa.shape[3] - (trlaa > 0).sum())
    list1.append(label_ratio)
list2 = np.sort(list1)
print('mean: ',np.mean(list2))
print('max: ',np.max(list2))
print('min: ', np.min(list2))
print('20%: ', np.percentile(list2,20))
print('80%: ', np.percentile(list2,80))

# plot
plt.hist(list2,bins=100)
plt.title('label ratio distribution')
plt.savefig('../' + output + '/dis.png')
df = pd.DataFrame(list2)
df.to_csv('../' + output + '/dis.csv',index_label = False)

