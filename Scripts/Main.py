import matplotlib
matplotlib.use('Agg')
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
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

# Type in output folder, epoch number, and initial learning rate
output = sys.argv[1]
eps = sys.argv[2]
LR = sys.argv[3]

# Make directory if not exist
if not os.path.exists('../' + output):
    os.makedirs('../' + output)

# Use cuda or not
USE_CUDA = 1

# Data loader
def dataloader(handles, mode = 'train'):
    # If pickle exists, load it
    try:
        with open('../inputs/mirror_roll/' + mode + '.pickle', 'rb') as f:
            images = pickle.load(f)
    except:
        images = {}
        images['Image'] = []
        images['Label'] = []
        images['ID'] = []
        images['Dim'] = []
        # For each image
        for idx, row in handles.iterrows():
            im = imread(row['Image'])
            # Normalization
            im = im / im.max() * 255
            im_aug = []
            shape = im.shape[0]
            # Training set will augment
            if mode == 'train':
                ima = np.rot90(im)
                imb = np.rot90(ima)
                imc = np.rot90(imb)
                imd = np.fliplr(im)
                ime = np.flipud(im)
            # rearrange channels
            image = np.empty((3, shape, shape), dtype='float32')
            for i in range(3):
                image[i,:,:] = im[:,:,i]
            im = image
            im_aug.append(im)
            if mode == 'train':
                for i in range(3):
                    image[i, :, :] = ima[:, :, i]
                ima = image
                for i in range(3):
                    image[i, :, :] = imb[:, :, i]
                imb = image
                for i in range(3):
                    image[i, :, :] = imc[:, :, i]
                imc = image
                for i in range(3):
                    image[i, :, :] = imd[:, :, i]
                imd = image
                for i in range(3):
                    image[i, :, :] = ime[:, :, i]
                ime = image
                im_aug.append(ima)
                im_aug.append(imb)
                im_aug.append(imc)
                im_aug.append(imd)
                im_aug.append(ime)
            images['Image'].append(np.array(im_aug))

            if mode != 'test':
                la_aug = []
                la = imread(row['Label'])
                # Augment for label
                if mode == 'train':
                    laa = np.rot90(la)
                    lab = np.rot90(laa)
                    lac = np.rot90(lab)
                    lad = np.fliplr(la)
                    lae = np.flipud(la)
                la = np.reshape(la, [1, la.shape[0], la.shape[1]])
                la_aug.append(la)
                if mode == 'train':
                    laa = np.reshape(laa, [1, laa.shape[0], laa.shape[1]])
                    lab = np.reshape(lab, [1, lab.shape[0], lab.shape[1]])
                    lac = np.reshape(lac, [1, lac.shape[0], lac.shape[1]])
                    lad = np.reshape(lad, [1, lad.shape[0], lad.shape[1]])
                    lae = np.reshape(lae, [1, lae.shape[0], lae.shape[1]])
                    la_aug.append(laa)
                    la_aug.append(lab)
                    la_aug.append(lac)
                    la_aug.append(lad)
                    la_aug.append(lae)
                images['Label'].append(np.array(la_aug))
            # For test set, save dimension for post processing
            elif mode == 'test':
                images['Dim'].append([(row['Width'], row['Height'])])
            images['ID'].append(row['ID'])

        with open("../inputs/mirror_roll/" + mode + '.pickle', 'wb') as f:
            pickle.dump(images, f)
        with open('../inputs/mirror_roll/' + mode + '.pickle', 'rb') as f:
            images = pickle.load(f)
    return images

# UNet model
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
def train(bs, sample, vasample, ep, ilr):
    # Initialize learning rate decay and learning rate
    lr_dec = 1
    init_lr = ilr
    # model
    model = Cuda(UNet())
    # initialize weight
    init_weights(model)
    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=init_lr)
    opt.zero_grad()
    # train and validation samples
    rows_trn = len(sample['Label'])
    rows_val = len(vasample['Label'])
    # Batch per epoch
    batches_per_epoch = rows_trn // bs
    losslists = []
    vlosslists = []

    for epoch in range(ep):
        # Learning rate
        lr = init_lr * lr_dec
        order = np.arange(rows_trn)
        losslist = []
        tr_metric_list = []
        va_metric_list = []
        for itr in range(batches_per_epoch):
            rows = order[itr * bs: (itr + 1) * bs]
            if itr + 1 == batches_per_epoch:
                rows = order[itr * bs:]
            # read in a batch
            trim = sample['Image'][rows[0]]
            trla = sample['Label'][rows[0]]
            # read in augmented images
            for iit in range(6):
                trimm = trim[iit:iit + 1, :, :, :]
                trlaa = trla[iit:iit + 1, :, :, :]
                # Calculate label positive and negative ratio
                label_ratio = (trlaa>0).sum() / (trlaa.shape[1]*trlaa.shape[2]*trlaa.shape[3] - (trlaa>0).sum())
                # If smaller than 1, add weight to positive prediction
                if label_ratio < 1:
                    add_weight = (trlaa[0,0,:,:] / 255 + 1 / (1 / label_ratio - 1))
                    add_weight = np.clip(add_weight / add_weight.max() * 255, 40, None)
                    loss_fn = torch.nn.BCEWithLogitsLoss(weight=Cuda(torch.from_numpy(add_weight).type(torch.FloatTensor)))
                # If smaller than 1, add weight to negative prediction
                elif label_ratio > 1:
                    add_weight = (trlaa[0,0,:,:] / 255 + 1 / (label_ratio - 1))
                    add_weight = np.clip(add_weight / add_weight.max() * 255, 40, None)
                    loss_fn = torch.nn.BCEWithLogitsLoss(weight=Cuda(torch.from_numpy(add_weight).type(torch.FloatTensor)))
                # If equal to 1, no weight added
                elif label_ratio == 1:
                    add_weight = np.ones([1,1,trlaa.shape[2], trlaa.shape[3]]) * 255
                    loss_fn = torch.nn.BCEWithLogitsLoss(weight=Cuda(torch.from_numpy(add_weight).type(torch.FloatTensor)))
                # Cuda and tensor inputs and label
                x = Cuda(Variable(torch.from_numpy(trimm).type(torch.FloatTensor)))
                y = Cuda(Variable(torch.from_numpy(trlaa / 255).type(torch.FloatTensor)))
                # Prediction
                pred_mask = model(x)
                # BCE and dice loss
                loss = loss_fn(pred_mask, y).cpu() + dice_loss(F.sigmoid(pred_mask), y)
                losslist.append(loss.data.numpy()[0])
                loss.backward()
                # ppv metric
                tr_metric = metric(F.sigmoid(pred_mask), y)
                tr_metric_list.append(tr_metric)
            opt.step()
            opt.zero_grad()

        vlosslist = []
        # For validation set
        for itr in range(rows_val):
            vaim = vasample['Image'][itr]
            vala = vasample['Label'][itr]
            for iit in range(1):
                # Load one batch
                vaimm = vaim[iit:iit + 1, :, :, :]
                valaa = vala[iit:iit + 1, :, :, :]
                # Calculate label positive and negative ratio
                label_ratio = (valaa>0).sum() / (valaa.shape[1]*valaa.shape[2] * valaa.shape[3] - (valaa>0).sum())
                # If smaller than 1, add weight to positive prediction
                if label_ratio < 1:
                    add_weight = (valaa[0,0,:,:] / 255 + 1 / (1 / label_ratio - 1))
                    add_weight = np.clip(add_weight / add_weight.max() * 255, 40, None)
                    loss_fn = torch.nn.BCEWithLogitsLoss(weight=Cuda(torch.from_numpy(add_weight).type(torch.FloatTensor)))
                # If smaller than 1, add weight to negative prediction
                elif label_ratio > 1:
                    add_weight = (valaa[0,0,:,:] / 255 + 1 / (label_ratio - 1))
                    add_weight = np.clip(add_weight / add_weight.max() * 255, 40, None)
                    loss_fn = torch.nn.BCEWithLogitsLoss(weight=Cuda(torch.from_numpy(add_weight).type(torch.FloatTensor)))
                # If equal to 1, no weight added
                elif label_ratio == 1:
                    add_weight = np.ones([1,1,valaa.shape[2], valaa.shape[3]]) * 255
                    loss_fn = torch.nn.BCEWithLogitsLoss(weight=Cuda(torch.from_numpy(add_weight).type(torch.FloatTensor)))
                # cuda and tensor sample
                xv = Cuda(Variable(torch.from_numpy(vaimm).type(torch.FloatTensor)))
                yv = Cuda(Variable(torch.from_numpy(valaa / 255).type(torch.FloatTensor)))
                # prediction
                pred_maskv = model(xv)
                # dice and BCE loss
                vloss = loss_fn(pred_maskv, yv).cpu() + dice_loss(F.sigmoid(pred_maskv), yv)
                vlosslist.append(vloss.data.numpy()[0])
                # ppv metric
                va_metric = metric(F.sigmoid(pred_maskv), yv)
                va_metric_list.append(va_metric)

        lossa = np.mean(losslist)
        vlossa = np.mean(vlosslist)
        tr_score = np.mean(tr_metric_list)
        va_score = np.mean(va_metric_list)
        # Print epoch summary
        print(
            'Epoch {:>3} |lr {:>1.5f} | Loss {:>1.5f} | VLoss {:>1.5f} | Train Score {:>1.5f} | Val Score {:>1.5f} '.format(
                epoch + 1, lr, lossa, vlossa, tr_score, va_score))
        losslists.append(lossa)
        vlosslists.append(vlossa)

        for param_group in opt.param_groups:
            param_group['lr'] = lr
        # Save model every 10 epoch
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
            }
            torch.save(checkpoint, '../' + output + '/unet-{}'.format(epoch + 1))
        # if no change or increase in loss for consecutive 6 epochs, decrease learning rate by 10 folds
        if epoch > 6:
            if losscp(losslists[-5:]) or losscp(vlosslists[-5:]):
                lr_dec = lr_dec / 10
        # if no change or increase in loss for consecutive 15 epochs, save validation predictions and stop training
        if epoch > 15:
            if losscp(losslists[-15:]) or losscp(vlosslists[-15:]):
                for itr in range(rows_val):
                    vaim = vasample['Image'][itr]
                    for iit in range(1):
                        vaimm = vaim[iit:iit + 1, :, :, :]
                        xv = Cuda(Variable(torch.from_numpy(vaimm).type(torch.FloatTensor)))
                        pred_maskv = model(xv)
                        pred_np = (F.sigmoid(pred_maskv) > 0.5).cpu().data.numpy().astype(np.uint8) * 255
                        if not os.path.exists('../' + output + '/validation/'):
                            os.makedirs('../' + output + '/validation/')
                        imsave('../' + output + '/validation/'+ vasample['ID'][itr] + '.png', pred_np[0,0,:,:])
                break

    # Loss figures
    plt.plot(losslists)
    plt.plot(vlosslists)
    plt.title('Train & Validation Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('../' + output + '/loss.png')
    return model

# method for test
def test(tesample, model, group):
    test_ids = []
    rles = []
    if not os.path.exists('../' + output + '/' + group):
        os.makedirs('../' + output + '/' + group)
    for itr in range(len(tesample['ID'])):
        teim = tesample['Image'][itr]
        teid = tesample['ID'][itr]
        tedim = tesample['Dim'][itr]
        # cuda and tensor input
        xt = Cuda(Variable(torch.from_numpy(teim).type(torch.FloatTensor)))
        # prediciton
        pred_mask = model(xt)
        # binarize output mask
        pred_np = (F.sigmoid(pred_mask) > 0.5).cpu().data.numpy().astype(np.uint8)
        # cut back to original image size
        pred_np = back_scale(pred_np, tedim)
        # save predicted mask
        imsave('../' + output + '/' + group + '/' + teid + '_pred.png', ((pred_np/pred_np.max())*255).astype(np.uint8))
        # vectorize mask
        rle = list(prob_to_rles(pred_np))
        rles.extend(rle)
        test_ids.extend([teid] * len(rle))
    # save vectorize masks as CSV
    sub = pd.DataFrame()
    sub['ImageId'] = test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

    return sub

# Read in files containing paths to training, validation, and testing images
tr = pd.read_csv('../inputs/stage_1_train/samples.csv', header=0,
                       usecols=['Image', 'Label', 'Width', 'Height', 'ID'])
va = pd.read_csv('../inputs/stage_1_test/vsamples.csv', header=0,
                       usecols=['Image', 'Label', 'Width', 'Height', 'ID'])
te = pd.read_csv('../inputs/stage_2_test/samples.csv', header=0, usecols=['Image', 'ID', 'Width', 'Height'])
# Load in images
trsample = dataloader(tr, 'train')
vasample = dataloader(va, 'val')
tebsample = dataloader(te, 'test')

# training
model = train(1, trsample, vasample, int(eps), float(LR))
# test set prediction
tebsub = test(tebsample, model, 'stage_2_test')
# save vectorize masks as CSV
tebsub.to_csv('../' + output + '/stage_2_test_sub.csv', index=False)
