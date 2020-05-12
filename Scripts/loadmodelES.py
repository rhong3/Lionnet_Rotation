import numpy as np  # linear algebra
import torch
from torch.autograd import Variable
from skimage import io
from torch.nn import functional as F
import skimage.morphology as mph
import sys
import os
import pickle
import scipy.misc

output = sys.argv[1]
md = sys.argv[2]
gp = sys.argv[3]

# Use cuda or not
USE_CUDA = 1

def Cuda(obj):
    if USE_CUDA:
        if isinstance(obj, tuple):
            return tuple(cuda(o) for o in obj)
        elif isinstance(obj, list):
            return list(cuda(o) for o in obj)
        elif hasattr(obj, 'cuda'):
            return obj.cuda()
    return obj

class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class UNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
        self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = torch.nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_block1 = UNet_down_block(3, 16, False)
        self.down_block2 = UNet_down_block(16, 32, True)
        self.down_block3 = UNet_down_block(32, 64, True)
        self.down_block4 = UNet_down_block(64, 128, True)
        self.down_block5 = UNet_down_block(128, 256, True)
        self.down_block6 = UNet_down_block(256, 512, True)
        self.down_block7 = UNet_down_block(512, 1024, True)

        self.mid_conv1 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.mid_conv2 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(1024)
        self.mid_conv3 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(1024)

        self.up_block1 = UNet_up_block(512, 1024, 512)
        self.up_block2 = UNet_up_block(256, 512, 256)
        self.up_block3 = UNet_up_block(128, 256, 128)
        self.up_block4 = UNet_up_block(64, 128, 64)
        self.up_block5 = UNet_up_block(32, 64, 32)
        self.up_block6 = UNet_up_block(16, 32, 16)

        self.last_conv1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = torch.nn.BatchNorm2d(16)
        self.last_conv2 = torch.nn.Conv2d(16, 1, 1, padding=0)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.down_block6(self.x5)
        self.x7 = self.down_block7(self.x6)
        self.x7 = self.relu(self.bn1(self.mid_conv1(self.x7)))
        self.x7 = self.relu(self.bn2(self.mid_conv2(self.x7)))
        self.x7 = self.relu(self.bn3(self.mid_conv3(self.x7)))
        x = self.up_block1(self.x6, self.x7)
        x = self.up_block2(self.x5, x)
        x = self.up_block3(self.x4, x)
        x = self.up_block4(self.x3, x)
        x = self.up_block5(self.x2, x)
        x = self.up_block6(self.x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x


def image_ids_in(root_dir, ignore=['.DS_Store', 'trainset_summary.csv', 'stage2_train_labels.csv', 'samples.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids


def dataloader(mode='test'):
    try:
        with open(mode + '_norm2.pickle', 'rb') as f:
            images = pickle.load(f)
    except:
        images = {}
        images['Image'] = []
        images['ID'] = []
        for ww in ['Rd1', 'Rd2', 'Rd3']:
            handles = image_ids_in(ww)
            # images['Dim'] = []
            for i in handles:
                im = io.imread(ww + '/'+i)
                # print(np.shape(im))
                # sigma_est = estimate_sigma(im, multichannel=False, average_sigmas=True)
                # print("Estimated Gaussian noise standard deviation = {}".format(sigma_est))
                # im = denoise_tv_chambolle(im, weight=0.1, multichannel=False)
                im = im / im.max() * 255
                im = 255 - im
                im_c = (im - im.mean())
                im_c[im_c < 0] = 0
                # im = (im_c / im_c.max() * 255)
                im = np.invert(im.astype(np.uint8))
                image = np.empty((im.shape[0], 3, im.shape[1], im.shape[2]), dtype='float32')
                for j in range(im.shape[0]):
                    for k in range(3):
                        image[j,k,:,:] = im[j,:,:]
                images['Image'].append(image)
                j = i.split('.')[0]
                # io.imsave('Images/' +'norm_'+ j + '.tif', im)
                images['ID'].append(ww + '_' + j)

        with open(mode + '_norm2.pickle', 'wb') as f:
            pickle.dump(images, f)
        with open(mode + '_norm2.pickle', 'rb') as f:
            images = pickle.load(f)
    return images


def test(tesample, model, mode):
    if not os.path.exists(output):
        os.makedirs(output)
    for itr in range(len(tesample['ID'])):
        teim = tesample['Image'][itr]
        # print(np.shape(teim))
        teid = tesample['ID'][itr]
        Da = teim.shape[2]
        Db = teim.shape[3]
        if teim.shape[2] != 1024 or teim.shape[3] != 1024:
            qq = np.empty((teim.shape[0],teim.shape[1],1024,1024))
            for j in range(teim.shape[0]):
                for k in range(3):
                    qq[j,k,:,:] = scipy.misc.imresize(teim[j,k,:,:], (1024, 1024))
            teim = qq
        ott = np.empty((teim.shape[0], Da, Db))
        for itt in range(teim.shape[0]):
            xt = Cuda(Variable(torch.from_numpy(teim[itt:itt+1, :, :, :]).type(torch.FloatTensor)))
            pred_mask = model(xt)
            pred_np = (F.sigmoid(pred_mask) > 0.625).cpu().data.numpy().astype(np.uint8)
            pred_np = scipy.misc.imresize(pred_np[0,0,:,:], (Da, Db))
            pred_np = mph.remove_small_objects(pred_np.astype(bool), min_size=500, connectivity=2).astype(np.uint8)
            if mode == 'nuke':
                pred_np = mph.remove_small_holes(pred_np, min_size=500, connectivity=2)
            ott[itt,:,:] = pred_np
        io.imsave(output + '/' + teid + mode + '.tif', ((ott/ott.max())*255).astype(np.uint8))


def cbtest(tesample):
    for itr in range(len(tesample['ID'])):
        teid = tesample['ID'][itr]
        a = io.imread(output + '/' + teid + 'nuke.tif')
        b = io.imread(output + '/' + teid + 'gap.tif')
        pred = np.clip(a - b, 0, None)
        pred = mph.remove_small_objects(pred.astype(bool), min_size=500, connectivity=2).astype(np.uint8)
        pred = mph.remove_small_holes(pred, min_size=3000, connectivity=2)
        io.imsave(output + '/' + teid + '_pred.tif', ((pred / pred.max()) * 255).astype(np.uint8))


if __name__ == '__main__':
    sample = dataloader('test')

    model = Cuda(UNet())
    a = torch.load(md)
    model.load_state_dict(a['state_dict'])

    test(sample, model, 'nuke')

    model = Cuda(UNet())
    a = torch.load(gp)
    model.load_state_dict(a['state_dict'])

    test(sample, model, 'gap')

    cbtest(sample)

