import matplotlib
matplotlib.use('Agg')
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(1234)
import torch
import torch.nn as nn
from torch.autograd import Variable
import skimage.io as io
from torch.nn import functional as F
from torch.nn import init
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from skimage.morphology import label
import matplotlib.pyplot as plt
import sys
import os
import pickle
import time
from PIL import Image
import skimage.morphology as mph
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)


# Type in output folder, epoch number, and initial learning rate
output = sys.argv[1]
eps = sys.argv[2]
LR = sys.argv[3]

# Make directory if not exist
if not os.path.exists('../' + output):
    os.makedirs('../' + output)

# Use cuda or not
USE_CUDA = 1

train_transformer = transforms.Compose([
    transforms.ColorJitter(brightness=0.35, contrast=0.5, saturation=0.5, hue=0.35),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation((-180, 180), resample=False, expand=False, center=None),
    transforms.ToTensor(),
])

val_transformer = transforms.Compose([
    transforms.ToTensor(),
])


class DataSet(Dataset):
    def __init__(self, datadir, transform=None):
        self.data_dir = datadir
        self.transform = transform
        self.imglist = pd.read_csv(self.data_dir, header=0).values.tolist()

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.open(self.imglist[idx][1])
        label = Image.open(self.imglist[idx][2])
        gap = Image.open(self.imglist[idx][3])

        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label': label,
                  'gap': gap}
        return sample


# UNet model
class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv3d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(output_channel)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(output_channel)
        self.max_pool = torch.nn.MaxPool3d(2, 2)
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
        self.conv1 = torch.nn.Conv3d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(output_channel)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(output_channel)
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

        self.mid_conv1 = torch.nn.Conv3d(1024, 1024, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(1024)
        self.mid_conv2 = torch.nn.Conv3d(1024, 1024, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(1024)
        self.mid_conv3 = torch.nn.Conv3d(1024, 1024, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(1024)

        self.up_block1 = UNet_up_block(512, 1024, 512)
        self.up_block2 = UNet_up_block(256, 512, 256)
        self.up_block3 = UNet_up_block(128, 256, 128)
        self.up_block4 = UNet_up_block(64, 128, 64)
        self.up_block5 = UNet_up_block(32, 64, 32)
        self.up_block6 = UNet_up_block(16, 32, 16)

        self.last_conv1 = torch.nn.Conv3d(16, 16, 3, padding=1)
        self.last_bn = torch.nn.BatchNorm3d(16)
        self.last_conv2 = torch.nn.Conv3d(16, 1, 1, padding=0)
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


# Initial weights
def init_weights(module):
    for name, param in module.named_parameters():
        if name.find('weight') != -1:
            if len(param.size()) == 1:
                Cuda(init.uniform(param.data, 1).type(torch.DoubleTensor))
            else:
                Cuda(init.xavier_uniform(param.data).type(torch.DoubleTensor))
        elif name.find('bias') != -1:
            Cuda(init.constant(param.data, 0).type(torch.DoubleTensor))

# Cuda
def Cuda(obj):
    if USE_CUDA:
        if isinstance(obj, tuple):
            return tuple(cuda(o) for o in obj)
        elif isinstance(obj, list):
            return list(cuda(o) for o in obj)
        elif hasattr(obj, 'cuda'):
            return obj.cuda()
    return obj

# Early stop function
def losscp (list):
    newlist = np.sort(list)
    if np.array_equal(np.array(list), np.array(newlist)):
        return 1
    else:
        return 0


# Cut predicted test image back to original size
def back_scale(model_im, im_shape):
    temp = np.reshape(model_im, [model_im.shape[-2], model_im.shape[-1]])
    row_size_left = (temp.shape[0] - im_shape[0][1]) // 2
    row_size_right = (temp.shape[0] - im_shape[0][1]) // 2 + (temp.shape[0] - im_shape[0][1]) % 2
    col_size_left = (temp.shape[1] - im_shape[0][0]) // 2
    col_size_right = (temp.shape[1] - im_shape[0][0]) // 2 + (temp.shape[1] - im_shape[0][0]) % 2
    if row_size_right == 0 and col_size_right == 0:
        new_im = temp[row_size_left:, col_size_left:]
    elif row_size_right == 0:
        new_im = temp[row_size_left:, col_size_left:-col_size_right]
    elif col_size_right == 0:
        new_im = temp[row_size_left:-row_size_right, col_size_left:]
    else:
        new_im = temp[row_size_left:-row_size_right, col_size_left:-col_size_right]
    return new_im

# Vectorize predicted test images
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

# Vectorize predicted test images
def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

# Dice loss function
def dice_loss(input, target):
    smooth = 1.
    iflat = input.view(-1).cpu()
    tflat = target.view(-1).cpu()
    intersection = (iflat * tflat).sum()
    return 1.0 - (((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)))


# F1 score
def Fscore(y_pred, target):
    pred = Cuda((y_pred.view(-1) > 0.5).type(torch.FloatTensor))
    target_vec = Cuda(target.view(-1).type(torch.FloatTensor))
    label = target_vec.sum().cpu().data.numpy()
    tp = (pred * target_vec).sum().cpu().data.numpy()
    predicted = pred.sum().cpu().data.numpy()
    recall = tp / predicted
    precision = tp / label
    F = 2 * precision * recall / (precision + recall)
    return F


# PPV metric function
def metric(y_pred, target):
    pred = Cuda((y_pred.view(-1) > 0.5).type(torch.FloatTensor))
    target_vec = Cuda(target.view(-1).type(torch.FloatTensor))
    label = target_vec.sum().cpu().data.numpy()
    tp = (pred * target_vec).sum().cpu().data.numpy()
    predicted = pred.sum().cpu().data.numpy()
    ppv = (tp) / (predicted + label - tp)
    return ppv


# Training and validation method
def train(tr_dir, va_dir, bs, ep, ilr, mode):

    try:
        trs = DataSet(str(tr_dir + '/samples.csv'), transform=train_transformer)
        vas = DataSet(str(va_dir + '/samples.csv'), transform=val_transformer)
    except FileNotFoundError:
        trs = DataSet(str(tr_dir + '/samples.csv'), transform=train_transformer)
        vas = DataSet(str(va_dir + '/samples.csv'), transform=val_transformer)

    train_loader = DataLoader(trs, batch_size=bs, drop_last=False, shuffle=True)
    val_loader = DataLoader(vas, batch_size=bs, drop_last=False, shuffle=False)

    # Initialize learning rate decay and learning rate
    init_lr = ilr
    # model
    model = Cuda(UNet())
    # initialize weight
    init_weights(model)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    trlosslist = []
    valosslist = []
    tr_metric_list = []
    va_metric_list = []
    tr_F_list = []
    va_F_list = []

    for epoch in range(ep):
        train_loss = 0
        validation_loss = 0
        tr_metric_e = 0
        va_metric_e = 0
        tr_F_e = 0
        va_F_e = 0
        for batch_index, batch_samples in enumerate(train_loader):
            if mode == 'nuke':
                data, target = batch_samples['img'].to('cuda'), batch_samples['label'].to('cuda')
                loss_fn = torch.nn.BCEWithLogitsLoss()
                # Prediction
                pred_mask = model(data)
                # BCE
                loss = loss_fn(pred_mask, target)
                train_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # ppv metric
                tr_metric = metric(F.sigmoid(pred_mask), target)
                tr_metric_list.append(tr_metric)
                tr_F = Fscore(F.sigmoid(pred_mask), target)
                tr_F_list.append(tr_F)
            elif mode == 'gap':
                data, target = batch_samples['img'].to('cuda'), batch_samples['gap'].to('cuda')
                loss_fn = torch.nn.BCEWithLogitsLoss()
                # Prediction
                pred_mask = model(data)
                # BCE
                loss = loss_fn(pred_mask, target)
                train_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # ppv metric
                tr_metric = metric(F.sigmoid(pred_mask), target)
                tr_metric_e += tr_metric
                tr_F = Fscore(F.sigmoid(pred_mask), target)
                tr_F_e += tr_F
        print('\nEpoch: {} \nTrain set: Average loss: {:.4f}\n Average PPV: {:.4f}\n Average F: {:.4f}\n'.format(
            epoch, train_loss / len(train_loader.dataset), tr_metric_e / len(train_loader.dataset),
            tr_F_e / len(train_loader.dataset)), flush=True)
        trlosslist.append(train_loss / len(train_loader.dataset))
        tr_metric_list.append(tr_metric_e / len(train_loader.dataset))
        tr_F_list.append(tr_F_e / len(train_loader.dataset))

        for batch_index, batch_samples in enumerate(val_loader):
            if mode == 'nuke':
                data, target = batch_samples['img'].to('cuda'), batch_samples['label'].to('cuda')
                loss_fn = torch.nn.BCEWithLogitsLoss()
                # Prediction
                pred_mask = model(data)
                # BCE
                loss = loss_fn(pred_mask, target)
                validation_loss += loss
                # ppv metric
                va_metric = metric(F.sigmoid(pred_mask), target)
                va_metric_list.append(va_metric)
                va_F = Fscore(F.sigmoid(pred_mask), target)
                va_F_list.append(va_F)
            elif mode == 'gap':
                data, target = batch_samples['img'].to('cuda'), batch_samples['gap'].to('cuda')
                loss_fn = torch.nn.BCEWithLogitsLoss()
                # Prediction
                pred_mask = model(data)
                # BCE
                loss = loss_fn(pred_mask, target)
                train_loss += loss
                # ppv metric
                va_metric = metric(F.sigmoid(pred_mask), target)
                va_metric_e += va_metric
                va_F = Fscore(F.sigmoid(pred_mask), target)
                va_F_e += va_F
        print('\nEpoch: {} \nValidation set: Average loss: {:.4f}\n Average PPV: {:.4f}\n Average F: {:.4f}\n'.format(
            epoch, validation_loss / len(val_loader.dataset), va_metric_e / len(val_loader.dataset),
            va_F_e / len(val_loader.dataset)), flush=True)
        valosslist.append(validation_loss / len(val_loader.dataset))
        va_metric_list.append(va_metric_e / len(val_loader.dataset))
        va_F_list.append(va_F_e / len(val_loader.dataset))

        # Save models
        if validation_loss / len(val_loader.dataset) == np.min(valosslist):
            print('Min loss found:')
            print(validation_loss / len(val_loader.dataset))
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, '../' + output + '/' + mode + 'loss_unet')
        if va_F_e / len(val_loader.dataset) == np.max(va_F_list):
            print('Max F found:')
            print(va_F_e / len(val_loader.dataset))
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, '../' + output + '/' + mode + 'F_unet')

        if va_metric_e / len(val_loader.dataset) == np.max(va_metric_list):
            print('Max PPV found:')
            print(va_metric_e / len(val_loader.dataset))
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, '../' + output + '/' + mode + 'PPV_unet')

        # if no change or increase in loss for consecutive 15 epochs, save validation predictions and stop training
        if epoch > 15:
            if losscp(trlosslist[-10:]) or losscp(valosslist[-10:]) or epoch+1 == ep:
                break

    # Loss figures
    plt.plot(trlosslist)
    plt.plot(valosslist)
    plt.title('Train & Validation Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('../' + output + '/'+mode+'_loss.png')


def vatest(vasample):
    if not os.path.exists('../' + output + '/validation'):
        os.makedirs('../' + output + '/validation')
    for itr in range(len(vasample['ID'])):
        vaid = vasample['ID'][itr]
        a = io.imread('../' + output + '/nukevalidation/' + vaid + '.png')
        b = io.imread('../' + output + '/gapvalidation/' + vaid + '.png')
        out = np.clip(a - b, 0, None)
        out = mph.remove_small_objects(out, min_size=30, connectivity=1)
        out = mph.remove_small_holes(out, min_size=30, connectivity=2)
        io.imsave('../' + output + '/validation/' + vaid + '_pred.png',
               ((out / out.max()) * 255).astype(np.uint8))


def cbtest(tesample, group):
    test_ids = []
    rles = []
    if not os.path.exists('../' + output + '/final_' + group):
        os.makedirs('../' + output + '/final_' + group)
    for itr in range(len(tesample['ID'])):
        teid = tesample['ID'][itr]
        a = io.imread('../' + output + '/' + group + '/' + teid + '_nuke_pred.png')
        b = io.imread('../' + output + '/' + group + '/' + teid + '_gap_pred.png')
        out = np.clip(a - b, 0, None)
        out = mph.remove_small_objects(out, min_size=30, connectivity=1)
        out = mph.remove_small_holes(out, min_size=30, connectivity=2)
        out = ((out / out.max()) * 255).astype(np.uint8)
        io.imsave('../' + output + '/final_' + group + '/' + teid + '_pred.png',
               ((out / out.max()) * 255).astype(np.uint8))
        # vectorize mask
        rle = list(prob_to_rles(out))
        rles.extend(rle)
        test_ids.extend([teid] * len(rle))
    # save vectorize masks as CSV
    sub = pd.DataFrame()
    sub['ImageId'] = test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

    return sub


if __name__ == '__main__':
    # training
    train('../inputs/train3D', '../inputs/validation3D', 1, int(eps), float(LR), 'nuke')
    train('../inputs/train3D', '../inputs/validation3D', 1, int(eps), float(LR), 'gap')

