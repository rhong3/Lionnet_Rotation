# Cycle-GAN 3D models

import random
import time
import datetime
import sys
from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReplicationPad3d(1),
                        nn.Conv3d(in_features, in_features, 3),
                        nn.InstanceNorm3d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReplicationPad3d(1),
                        nn.Conv3d(in_features, in_features, 3),
                        nn.InstanceNorm3d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [   nn.ReplicationPad3d(3),
                    nn.Conv3d(input_nc, 64, 7),
                    nn.InstanceNorm3d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv3d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm3d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose3d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm3d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReplicationPad3d(3),
                    nn.Conv3d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv3d(input_nc, 64, 3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv3d(64, 128, 3, stride=2, padding=1),
                    nn.InstanceNorm3d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv3d(128, 256, 3, stride=2, padding=1),
                    nn.InstanceNorm3d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv3d(256, 512, 3, padding=1),
                    nn.InstanceNorm3d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv3d(512, 1, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)


def tensor2image(tensor):
    tensor = torch.squeeze(tensor)
    tensor = torch.unsqueeze(tensor, 1)
    image = np.clip(255 * (tensor.cpu().float().numpy()), 0, 255)
    if image.shape[1] == 1:
        image = np.tile(image, (1, 3, 1, 1))
    return image.astype(np.uint8)


def tensor2numpy(tensor):
    tensor = torch.squeeze(tensor)
    image = np.clip(255 * (tensor.cpu().float().numpy()), 0, 255)
    return image.astype(np.uint8)


class Logger():
    def __init__(self, n_epochs, batches_epoch, outputfile, server_name):
        self.viz = Visdom(server=server_name, log_to_filename=outputfile)
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i + 1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s \n' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        try:
            # Draw images
            tlist = []
            for image_name, tensor in images.items():
                tlist.append(tensor)
            tensortemp = torch.cat([tlist[0], tlist[1]], dim=2)
            for ii in range(2, len(tlist)):
                tensortemp = torch.cat([tensortemp, tlist[ii]], dim=2)
            if 'images' not in self.image_windows:
                self.image_windows['images'] = self.viz.images(tensor2image(tensortemp.data), nrow=7,
                                                                             opts={'title': 'images'})
            else:
                self.viz.images(tensor2image(tensortemp.data), win=self.image_windows['images'], nrow=7,
                               opts={'title': 'images'})

            # End of epoch
            if (self.batch % self.batches_epoch) == 0:
                # Plot losses
                for loss_name, loss in self.losses.items():
                    if loss_name not in self.loss_windows:
                        self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]),
                                                                     Y=np.array([loss / self.batch]),
                                                                     opts={'xlabel': 'epochs', 'ylabel': loss_name,
                                                                           'title': loss_name})
                    else:
                        self.viz.line(X=np.array([self.epoch]), Y=np.array([loss / self.batch]),
                                      win=self.loss_windows[loss_name], update='append')
                    # Reset losses for next epoch
                    self.losses[loss_name] = 0.0
                self.epoch += 1
                self.batch = 1
                sys.stdout.write('\n')
            elif self.epoch * self.batch == 1:
                # Plot losses
                for loss_name, loss in self.losses.items():
                    if loss_name not in self.loss_windows:
                        self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch-1]),
                                                                     Y=np.array([loss]),
                                                                     opts={'xlabel': 'epochs', 'ylabel': loss_name,
                                                                           'title': loss_name})
                    else:
                        self.viz.line(X=np.array([self.epoch-1]), Y=np.array([loss]),
                                      win=self.loss_windows[loss_name], update='append')
                self.batch += 1
                sys.stdout.write('\n')
            else:
                self.batch += 1
        except:
            pass


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


