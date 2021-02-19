# Import all relevant libraries
# import torchio as tio
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.nn import init
import sys
import os
from skimage import io
import torch.nn.functional as F
matplotlib.use('Agg')

# Type in output folder, epoch number, and initial learning rate
output = sys.argv[1]
eps = sys.argv[2]
LR = sys.argv[3]

# Make directory if not exist
if not os.path.exists(output):
    os.makedirs(output)

# Set random seeds
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
seed = 1008
np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Define label converter
def label_converter(mask_data):
    new_masks = np.zeros(shape=mask_data.shape, dtype=np.uint8)
    for i in range(len(mask_data)):
        new_masks[i] = np.where(mask_data[i] > 0, 1, 0)

    return new_masks


# Data loader
def dataloader(handles, mode='train'):
    # If pickle exists, load it
    try:
        with open('../inputs/flpickles/' + mode + '.pickle', 'rb') as f:
            images = pickle.load(f)

    except:

        images = {}
        images['Image'] = []
        images['Label'] = []
        images['Gap'] = []
        images['ID'] = []

        # # Data augmentations
        # random_flip = tio.RandomFlip(axes=1)
        # random_flip2 = tio.RandomFlip(axes=2)
        # random_affine = tio.RandomAffine(seed=0, scales=(3, 3))
        # random_elastic = tio.RandomElasticDeformation(
        #     max_displacement=(0, 20, 40),
        #     num_control_points=20,
        #     seed=0,
        # )
        # rescale = tio.RescaleIntensity((-1, 1), percentiles=(1, 99))
        # standardize_foreground = tio.ZNormalization(masking_method=lambda x: x > x.mean())
        # blur = tio.RandomBlur(seed=0)
        # standardize = tio.ZNormalization()
        # add_noise = tio.RandomNoise(std=0.5, seed=42)
        # add_spike = tio.RandomSpike(seed=42)
        # add_ghosts = tio.RandomGhosting(intensity=1.5, seed=42)
        # add_motion = tio.RandomMotion(num_transforms=6, image_interpolation='nearest', seed=42)
        # swap = tio.RandomSwap(patch_size=7)

        # For each image
        for idx, row in handles.iterrows():
            im_aug = []
            lb_aug = []
            gap_aug = []
            imgs = np.zeros(shape=(1, 1, 7, 1024, 1024), dtype=np.float32)  # change patch shape if necessary
            lbs = np.zeros(shape=(1, 1, 7, 1024, 1024), dtype=np.float32)
            gaps = np.zeros(shape=(1, 1, 7, 1024, 1024), dtype=np.float32)
            im = io.imread(row['Image'])
            im = im / 255  # Normalization
            im = np.expand_dims(im, axis=0)
            imgs[0] = im
            im_aug.append(imgs)
            images['ID'].append(row['ID'])
            # if mode == 'train':
            #     im_flip1 = random_flip(im)
            #     imgs[0] = im_flip1
            #     im_aug.append(imgs)
            #     im_flip2 = random_flip2(im)
            #     imgs[0] = im_flip2
            #     im_aug.append(imgs)
            #     im_affine = random_affine(im)
            #     imgs[0] = im_affine
            #     im_aug.append(imgs)
            #     im_elastic = random_elastic(im)
            #     imgs[0] = im_elastic
            #     im_aug.append(imgs)
            #     im_rescale = rescale(im)
            #     imgs[0] = im_rescale
            #     im_aug.append(imgs)
            #     im_standard = standardize_foreground(im)
            #     imgs[0] = im_standard
            #     im_aug.append(imgs)
            #     im_blur = blur(im)
            #     imgs[0] = im_blur
            #     im_aug.append(imgs)
            #     im_noisy = add_noise(standardize(im))
            #     imgs[0] = im_noisy
            #     im_aug.append(imgs)
            #     im_spike = add_spike(im)
            #     imgs[0] = im_spike
            #     im_aug.append(imgs)
            #     im_ghost = add_ghosts(im)
            #     imgs[0] = im_ghost
            #     im_aug.append(imgs)
            #     im_motion = add_motion(im)
            #     imgs[0] = im_motion
            #     im_aug.append(imgs)
            #     im_swap = swap(im)
            #     imgs[0] = im_swap
            #     im_aug.append(imgs)
            images['Image'].append(np.array(im_aug))

            if mode != 'test':
                lb = io.imread(row['Label'])
                lb = label_converter(lb)
                lb = np.expand_dims(lb, axis=0)
                lbs[0] = lb
                lb_aug.append(lbs)
                gap = io.imread(row['Gap'])
                gap = np.expand_dims(gap, axis=0)
                gaps[0] = gap
                gap_aug.append(gaps)
                # if mode == 'train':
                    # lb_flip1 = random_flip(lb)
                    # lbs[0] = lb_flip1
                    # lb_aug.append(lbs)
                    # lb_flip2 = random_flip2(lb)
                    # lbs[0] = lb_flip2
                    # lb_aug.append(lbs)
                    # lb_affine = random_affine(lb)
                    # lbs[0] = lb_affine
                    # lb_aug.append(lbs)
                    # lb_elastic = random_elastic(lb)
                    # lbs[0] = lb_elastic
                    # lb_aug.append(lbs)
                    # lbs[0] = lb
                    # lb_aug.append(lbs)
                    # lbs[0] = lb
                    # lb_aug.append(lbs)
                    # lbs[0] = lb
                    # lb_aug.append(lbs)
                    # lbs[0] = lb
                    # lb_aug.append(lbs)
                    # lbs[0] = lb
                    # lb_aug.append(lbs)
                    # lbs[0] = lb
                    # lb_aug.append(lbs)
                    # lbs[0] = lb
                    # lb_aug.append(lbs)
                    # lbs[0] = lb
                    # lb_aug.append(lbs)
                    #
                    # gap_flip1 = random_flip(gap)
                    # gaps[0] = gap_flip1
                    # gap_aug.append(gaps)
                    # gap_flip2 = random_flip2(gap)
                    # gaps[0] = gap_flip2
                    # gap_aug.append(gaps)
                    # gap_affine = random_affine(gap)
                    # gaps[0] = gap_affine
                    # gap_aug.append(gaps)
                    # gap_elastic = random_elastic(gap)
                    # gaps[0] = gap_elastic
                    # gap_aug.append(gaps)
                    # gaps[0] = gap
                    # gap_aug.append(gaps)
                    # gaps[0] = gap
                    # gap_aug.append(gaps)
                    # gaps[0] = gap
                    # gap_aug.append(gaps)
                    # gaps[0] = gap
                    # gap_aug.append(gaps)
                    # gaps[0] = gap
                    # gap_aug.append(gaps)
                    # gaps[0] = gap
                    # gap_aug.append(gaps)
                    # gaps[0] = gap
                    # gap_aug.append(gaps)
                    # gaps[0] = gap
                    # gap_aug.append(gaps)
                images['Label'].append(np.array(lb_aug))
                images['Gap'].append(np.array(gap_aug))
        # Save images
        with open("../inputs/" + mode + '.pickle', 'wb') as f:
            pickle.dump(images, f)
        with open('../inputs/' + mode + '.pickle', 'rb') as f:
            images = pickle.load(f)

    return images


# UNet model
class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv3d(input_channel, output_channel, 3, padding=(2, 1, 1))
        self.bn1 = torch.nn.BatchNorm3d(output_channel)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=(1, 1, 1))
        self.bn2 = torch.nn.BatchNorm3d(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=(2, 1, 1))
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
        self.up_sampling = torch.nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.conv1 = torch.nn.Conv3d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(output_channel)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm3d(output_channel)
        # self.relu = torch.nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_block1 = UNet_down_block(1, 16, True)
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
        self.up_sampling = torch.nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
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
        self.x7 = F.relu(self.bn1(self.mid_conv1(self.x7)))
        self.x7 = F.relu(self.bn2(self.mid_conv2(self.x7)))
        self.x7 = F.relu(self.bn3(self.mid_conv3(self.x7)))
        x = self.up_block1(self.x6, self.x7)
        x = self.up_block2(self.x5, x)
        x = self.up_block3(self.x4, x)
        x = self.up_block4(self.x3, x)
        x = self.up_block5(self.x2, x)
        x = self.up_block6(self.x1, x)

        x = F.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        x = self.up_sampling(x)
        return x


# Initiail weights
def init_weights(module):
    for name, param in module.named_parameters():
        if name.find('weight') != -1:
            if len(param.size()) == 1:
                init.uniform_(param.data, 1).type(torch.DoubleTensor).to(device)
            else:
                init.xavier_uniform_(param.data).type(torch.DoubleTensor).to(device)
        elif name.find('bias') != -1:
            init.constant_(param.data, 0).type(torch.DoubleTensor).to(device)


# Early stop function
def losscp(list):
    newlist = np.sort(list)
    if np.array_equal(np.array(list), np.array(newlist)):
        return 1
    else:
        return 0


# Dice loss function
def dice_loss(input, target):
    smooth = 1.
    iflat = input.view(-1).cpu()
    tflat = target.view(-1).cpu()
    intersection = (iflat * tflat).sum()
    return 1.0 - (((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)))


# F1 score
def Fscore(y_pred, target):
    pred = (y_pred.view(-1) > 0.5).type(torch.FloatTensor).to(device)
    target_vec = target.view(-1).type(torch.FloatTensor).to(device)
    label = target_vec.sum().cpu().data.numpy()
    tp = (pred * target_vec).sum().cpu().data.numpy()
    predicted = pred.sum().cpu().data.numpy()
    recall = tp / predicted
    precision = tp / label
    F = 2 * precision * recall / (precision + recall)
    return F


# PPV metric function
def metric(y_pred, target):
    pred = (y_pred.view(-1) > 0.5).type(torch.FloatTensor).to(device)
    target_vec = target.view(-1).type(torch.FloatTensor).to(device)
    label = target_vec.sum().cpu().data.numpy()
    tp = (pred * target_vec).sum().cpu().data.numpy()
    predicted = pred.sum().cpu().data.numpy()
    ppv = (tp) / (predicted + label - tp)
    return ppv


# Training and validation method
def train(bs, sample, vasample, ep, ilr, mode):
    # Initialize learning rate decay and learning rate
    lr_dec = 1
    init_lr = ilr
    torch.cuda.empty_cache()
    # Model
    model = UNet()
    # initialize weight
    init_weights(model)
    model.to(device)
    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=init_lr)
    opt.zero_grad()
    # train and validation samples
    rows_trn = len(sample['Label'])
    print('rows_trn', rows_trn, flush=True)
    rows_val = len(vasample['Label'])
    print('rows_val', rows_val, flush=True)
    # Batch per epoch
    batches_per_epoch = rows_trn // bs
    losslists = []
    vlosslists = []
    Fscorelist = []
    PPVlist = []

    for epoch in range(ep):
        print(epoch, flush=True)
        # Learning rate
        lr = init_lr * lr_dec
        torch.cuda.empty_cache()
        order = np.arange(rows_trn)
        losslist = []
        tr_metric_list = []
        va_metric_list = []
        tr_F_list = []
        va_F_list = []
        for itr in range(batches_per_epoch):
            rows = order[itr * bs: (itr + 1) * bs]
            if itr + 1 == batches_per_epoch:
                rows = order[itr * bs:]
            # read in a batch
            trim = sample['Image'][rows[0]]
            trla = sample['Label'][rows[0]]
            trga = sample['Gap'][rows[0]]
            for iit in range(1):
                trimm = trim[iit]
                trlaa = trla[iit]
                trgaa = trga[iit]
                if mode == 'nuke':
                    label_ratio = (trlaa > 0).sum() / (
                            trlaa.shape[1] * trlaa.shape[2] * trlaa.shape[3] * trlaa.shape[4] - (trlaa > 0).sum())
                    # If smaller than 1, add weight to positive prediction
                    if label_ratio < 1:
                        add_weight = (trlaa[0, 0, :, :, :] + 1 / (1 / label_ratio - 1))
                        add_weight = np.clip(add_weight / add_weight.max() * 255, 40, None)
                        loss_fn = torch.nn.BCEWithLogitsLoss(
                            weight=torch.from_numpy(add_weight).type(torch.FloatTensor).to(device))
                    # If smaller than 1, add weight to negative prediction
                    elif label_ratio > 1:
                        add_weight = (trlaa[0, 0, :, :, :] + 1 / (label_ratio - 1))
                        add_weight = np.clip(add_weight / add_weight.max() * 255, 40, None)
                        loss_fn = torch.nn.BCEWithLogitsLoss(
                            weight=torch.from_numpy(add_weight).type(torch.FloatTensor).to(device))
                    # If equal to 1, no weight added
                    elif label_ratio == 1:
                        add_weight = (np.ones([1, 1, trlaa.shape[2], trlaa.shape[3], trlaa.shape[4]])) / 2 * 255
                        loss_fn = torch.nn.BCEWithLogitsLoss(
                            weight=torch.from_numpy(add_weight).type(torch.FloatTensor).to(device))
                    # Cuda and tensor inputs and label
                    x = Variable(torch.from_numpy(trimm).type(torch.FloatTensor)).to(device)
                    y = Variable(torch.from_numpy(trlaa).type(torch.FloatTensor)).to(device)
                    torch.cuda.empty_cache()
                    # Prediction
                    pred_mask = model(x)
                    # BCE and dice loss
                    loss = loss_fn(pred_mask, y).cpu() + dice_loss(torch.sigmoid(pred_mask), y)
                    losslist.append(loss.data.numpy())
                    loss.backward()
                    # ppv metric
                    tr_metric = metric(torch.sigmoid(pred_mask), y)
                    tr_metric_list.append(tr_metric)
                    tr_F = Fscore(torch.sigmoid(pred_mask), y)
                    tr_F_list.append(tr_F)
                    torch.cuda.empty_cache()
                elif mode == 'gap':
                    label_ratio = (trgaa > 0).sum() / (
                                trgaa.shape[1] * trgaa.shape[2] * trgaa.shape[3] * trgaa.shape[4] - (trgaa > 0).sum())
                    # If smaller than 1, add weight to positive prediction
                    if label_ratio < 1:
                        add_weight = (trgaa[0, 0, :, :, :] / 255 + 1 / (1 / label_ratio - 1))
                        add_weight = np.clip(add_weight / add_weight.max() * 255, 40, None)
                        loss_fn = torch.nn.BCEWithLogitsLoss(
                            weight=torch.from_numpy(add_weight).type(torch.FloatTensor).to(device))
                    # If smaller than 1, add weight to negative prediction
                    elif label_ratio > 1:
                        add_weight = (trgaa[0, 0, :, :, :] / 255 + 1 / (label_ratio - 1))
                        add_weight = np.clip(add_weight / add_weight.max() * 255, 40, None)
                        loss_fn = torch.nn.BCEWithLogitsLoss(
                            weight=torch.from_numpy(add_weight).type(torch.FloatTensor).to(device))
                    # If equal to 1, no weight added
                    elif label_ratio == 1:
                        add_weight = (np.ones([1, 1, trgaa.shape[2], trgaa.shape[3], trgaa.shape[4]])) / 2 * 255
                        loss_fn = torch.nn.BCEWithLogitsLoss(
                            weight=torch.from_numpy(add_weight).type(torch.FloatTensor).to(device))
                    # Cuda and tensor inputs and label
                    x = Variable(torch.from_numpy(trimm).type(torch.FloatTensor)).to(device)
                    y = Variable(torch.from_numpy(trgaa / 255).type(torch.FloatTensor)).to(device)
                    torch.cuda.empty_cache()
                    # Prediction
                    pred_mask = model(x)
                    # BCE and dice loss
                    loss = loss_fn(pred_mask, y).cpu() + dice_loss(torch.sigmoid(pred_mask), y)
                    losslist.append(loss.data.numpy())
                    loss.backward()

                    # ppv metric
                    tr_metric = metric(torch.sigmoid(pred_mask), y)
                    tr_metric_list.append(tr_metric)
                    tr_F = Fscore(torch.sigmoid(pred_mask), y)
                    tr_F_list.append(tr_F)
                    torch.cuda.empty_cache()
            opt.step()
            opt.zero_grad()
            torch.cuda.empty_cache()

        vlosslist = []
        # For validation set
        for itr in range(rows_val):
            torch.cuda.empty_cache()
            vaim = vasample['Image'][itr][0]
            vala = vasample['Label'][itr][0]
            vaga = vasample['Gap'][itr][0]
            if mode == 'nuke':
                label_ratio = (vala > 0).sum() / (
                            vala.shape[1] * vala.shape[2] * vala.shape[3] * vala.shape[4] - (vala > 0).sum())
                # If smaller than 1, add weight to positive prediction
                if label_ratio < 1:
                    add_weight = (vala[0, 0, :, :, :] + 1 / (1 / label_ratio - 1))
                    add_weight = np.clip(add_weight / add_weight.max() * 255, 40, None)
                    loss_fn = torch.nn.BCEWithLogitsLoss(
                        weight=torch.from_numpy(add_weight).type(torch.FloatTensor).to(device))
                # If smaller than 1, add weight to negative prediction
                elif label_ratio > 1:
                    add_weight = (vala[0, 0, :, :, :] + 1 / (label_ratio - 1))
                    add_weight = np.clip(add_weight / add_weight.max() * 255, 40, None)
                    loss_fn = torch.nn.BCEWithLogitsLoss(
                        weight=torch.from_numpy(add_weight).type(torch.FloatTensor).to(device))
                # If equal to 1, no weight added
                elif label_ratio == 1:
                    add_weight = (np.ones([1, 1, vala.shape[2], vala.shape[3], vala.shape[4]])) / 2 * 255
                    loss_fn = torch.nn.BCEWithLogitsLoss(
                        weight=torch.from_numpy(add_weight).type(torch.FloatTensor).to(device))
                    # cuda and tensor sample
                torch.cuda.empty_cache()
                xv = Variable(torch.from_numpy(vaim).type(torch.FloatTensor)).to(device)
                yv = Variable(torch.from_numpy(vala).type(torch.FloatTensor)).to(device)
                # prediction
                torch.cuda.empty_cache()
                pred_maskv = model(xv)
                pred_np = (torch.sigmoid(pred_maskv)).cpu().data.numpy()
                torch.cuda.empty_cache()
                pred_np = pred_np.round().astype(np.uint8)
                pred_np = pred_np[0, 0, :, :, :]
                pred_t = torch.from_numpy(((pred_np / pred_np.max()) * 255).astype(np.uint8))
                if not os.path.exists(output + '/' + mode + 'results_trainning/'):
                    os.makedirs(output + '/' + mode + 'results_trainning/')
                # [imageio.imwrite(
                #     output + '/' + mode + 'results_trainning/' + vasample['ID'][itr] + '_' + str(epoch) + '_%d.png' % i,
                #     pred_t[i, :, :]) for i in range(7)]
                # dice and BCE loss
                vloss = loss_fn(pred_maskv, yv).cpu() + dice_loss(torch.sigmoid(pred_maskv), yv)
                vlosslist.append(vloss.data.numpy())
                # ppv metric
                va_metric = metric(torch.sigmoid(pred_maskv), yv)
                va_metric_list.append(va_metric)
                va_F = Fscore(torch.sigmoid(pred_maskv), yv)
                va_F_list.append(va_F)
            elif mode == 'gap':
                label_ratio = (vaga > 0).sum() / (
                            vaga.shape[1] * vaga.shape[2] * vaga.shape[3] * vaga.shape[4] - (vaga > 0).sum())
                # If smaller than 1, add weight to positive prediction
                if label_ratio < 1:
                    add_weight = (vaga[0, 0, :, :, :] / 255 + 1 / (1 / label_ratio - 1))
                    add_weight = np.clip(add_weight / add_weight.max() * 255, 40, None)
                    loss_fn = torch.nn.BCEWithLogitsLoss(
                        weight=torch.from_numpy(add_weight).type(torch.FloatTensor).to(device))
                # If smaller than 1, add weight to negative prediction
                elif label_ratio > 1:
                    add_weight = (vaga[0, 0, :, :, :] / 255 + 1 / (label_ratio - 1))
                    add_weight = np.clip(add_weight / add_weight.max() * 255, 40, None)
                    loss_fn = torch.nn.BCEWithLogitsLoss(
                        weight=torch.from_numpy(add_weight).type(torch.FloatTensor).to(device))
                # If equal to 1, no weight added
                elif label_ratio == 1:
                    add_weight = (np.ones([1, 1, vaga.shape[2], vaga.shape[3], vaga.shape[4]])) / 2 * 255
                    loss_fn = torch.nn.BCEWithLogitsLoss(
                        weight=torch.from_numpy(add_weight).type(torch.FloatTensor).to(device))
                # cuda and tensor sample
                xv = Variable(torch.from_numpy(vaim).type(torch.FloatTensor)).to(device)
                yv = Variable(torch.from_numpy(vaga / 255).type(torch.FloatTensor)).to(device)
                # prediction
                pred_maskv = model(xv)

                # dice and BCE loss
                vloss = loss_fn(pred_maskv, yv).cpu() + dice_loss(torch.sigmoid(pred_maskv), yv)
                vlosslist.append(vloss.data.numpy())
                # ppv metric
                va_metric = metric(torch.sigmoid(pred_maskv), yv)
                va_metric_list.append(va_metric)
                va_F = Fscore(torch.sigmoid(pred_maskv), yv)
                va_F_list.append(va_F)
        lossa = np.mean(losslist)
        vlossa = np.mean(vlosslist)
        tr_score = np.mean(tr_metric_list)
        va_score = np.mean(va_metric_list)
        tr_F_list = np.nan_to_num(tr_F_list)
        va_F_list = np.nan_to_num(va_F_list)
        tr_Fscore = np.mean(tr_F_list)
        va_Fscore = np.mean(va_F_list)
        # Print epoch summary
        print(
            'Epoch {:>3} |lr {:>1.5f} | Loss {:>1.5f} | VLoss {:>1.5f} | Train F1 {:>1.5f} | Val F1 {:>1.5f} | Train PPV {:>1.5f} | Val PPV {:>1.5f}'.format(
                epoch + 1, lr, lossa, vlossa, tr_Fscore, va_Fscore, tr_score, va_score))
        losslists.append(lossa)
        vlosslists.append(vlossa)
        Fscorelist.append(va_Fscore)
        PPVlist.append(va_score)

        for param_group in opt.param_groups:
            param_group['lr'] = lr
        # Save models
        if vlossa == np.min(vlosslists):
            print('Min loss found:')
            print(vlossa)
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
            }
            torch.save(checkpoint, output + '/' + mode + 'loss_unet')
        if va_Fscore == np.max(Fscorelist):
            print('Max F found:')
            print(va_Fscore)
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
            }
            torch.save(checkpoint, output + '/' + mode + 'F_unet')

        if va_score == np.max(PPVlist):
            print('Max PPV found:')
            print(va_score)
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
            }
            torch.save(checkpoint, output + '/' + mode + 'PPV_unet')

        # if no change or increase in loss for consecutive 6 epochs, decrease learning rate by 10 folds
        if epoch > 6:
            if losscp(losslists[-5:]) or losscp(vlosslists[-5:]):
                lr_dec = lr_dec / 10

        # if no change or increase in loss for consecutive 15 epochs, save validation predictions and stop training
        if epoch > 15:
            if losscp(losslists[-15:]) or losscp(vlosslists[-15:]) or epoch + 1 == ep:
                for itr in range(rows_val):
                    vaim = vasample['Image'][itr][0]
                    xv = Variable(torch.from_numpy(vaim).type(torch.FloatTensor)).to(device)
                    pred_maskv = model(xv)
                    pred_np = (torch.sigmoid(pred_maskv)).cpu().data.numpy()
                    print(pred_np.shape, flush=True)
                    ppp = pred_np[0, 0, :, :, :]
                    pred_np = pred_np.round().astype(np.uint8)
                    pred_np = pred_np[0, 0, :, :, :]
                    pww = pred_np
                    if not os.path.exists(output + '/' + mode + 'validation/'):
                        os.makedirs(output + '/' + mode + 'validation/')

                    if np.max(pred_np) == np.min(pred_np):
                        print('1st_BOOM!')
                        print(vasample['ID'][itr])
                        if np.max(pww) == np.min(pww):
                            print('2nd_BOOM!')
                            if ppp.max() == 0 or ppp.min() == 1:
                                print('3rd_BOOM!')
                                ppp_t = torch.from_numpy(ppp.astype(np.uint8))
                                # [imageio.imwrite(
                                #     output + '/' + mode + 'validation/' + vasample['ID'][itr] + '_%d.png' % i,
                                #     ppp_t[i, :, :]) for i in range(7)]

                            else:
                                ppp = (ppp / ppp.max()) * 1
                                ppp = (ppp > 0.95).astype(np.uint8)
                                ppp_m = torch.from_numpy(((ppp / ppp.max()) * 255).astype(np.uint8))
                                # [imageio.imwrite(
                                #     output + '/' + mode + 'validation/' + vasample['ID'][itr] + '_%d.png' % i,
                                #     ppp_m[i, :, :]) for i in range(7)]

                        else:
                            pww_t = torch.from_numpy(((pww / pww.max()) * 255).astype(np.uint8))
                            # [imageio.imwrite(output + '/' + mode + 'validation/' + vasample['ID'][itr] + '_%d.png' % i,
                            #                  pww_t[i, :, :]) for i in range(7)]

                    else:
                        pred_t = torch.from_numpy(((pred_np / pred_np.max()) * 255).astype(np.uint8))
                        # [imageio.imwrite(output + '/' + mode + 'validation/' + vasample['ID'][itr] + '_%d.png' % i,
                        #                  pred_t[i, :, :]) for i in range(7)]

                break

    # Loss figures
    plt.plot(losslists)
    plt.plot(vlosslists)
    plt.title('Train & Validation Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(output + '/' + mode + '_loss.png')


if __name__ == '__main__':
    # Read in files containing paths to training, validation, and testing images
    data = pd.read_csv('../train3D/samples.csv', header=0,
                       usecols=['Type', 'Image', 'Label', 'Gap', 'Width', 'Height', 'Depth', 'ID'])
    # Split into tranning and validation
    n_rows = len(data)
    n_rows_perm = np.random.permutation(range(n_rows))
    n_rows_perm_80 = n_rows_perm[:19]
    n_rows_perm_20 = n_rows_perm[19:]
    tr = data.iloc[n_rows_perm_80]
    va = data.iloc[n_rows_perm_20]

    # Load in images
    trsample = dataloader(tr, 'train')
    vasample = dataloader(va, 'val')

    # training
    train(1, trsample, vasample, int(eps), float(LR), 'nuke')