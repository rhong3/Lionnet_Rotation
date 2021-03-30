import numpy as np  # linear algebra
import torch
from torch.autograd import Variable
from skimage import io
from torch.nn import functional as F
import skimage.morphology as mph
import sys
import os
import pickle

output = sys.argv[1]
md = sys.argv[2]

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
    def __init__(self, prev_channel, input_channel, output_channel, depth):
        super(UNet_up_block, self).__init__()
        self.up_sampling = torch.nn.Upsample(size=[depth, int(8192/output_channel), int(8192/output_channel)], mode='trilinear')
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

        self.up_block1 = UNet_up_block(512, 1024, 512, 8)
        self.up_block2 = UNet_up_block(256, 512, 256, 8)
        self.up_block3 = UNet_up_block(128, 256, 128, 9)
        self.up_block4 = UNet_up_block(64, 128, 64, 10)
        self.up_block5 = UNet_up_block(32, 64, 32, 12)
        self.up_block6 = UNet_up_block(16, 32, 16, 16)
        self.up_sampling = torch.nn.Upsample(size=[24, 1024, 1024], mode='trilinear')
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


def image_ids_in(root_dir, ignore=['.DS_Store', 'trainset_summary.csv', 'stage2_train_labels.csv', 'samples.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if '.tif' in id:
            ids.append(id)
        else:
            print('Skipping ID:', id)
    return ids


def dataloader(mode='test'):
    try:
        with open(mode + '_0319.pickle', 'rb') as f:
            images = pickle.load(f)
    except:
        # imgs = np.zeros(shape=(1, 1, 7, 1024, 1024), dtype=np.float32)  # change patch shape if necessary
        images = {}
        images['Image'] = []
        images['ID'] = []
        handles = image_ids_in('../Nuclei_20210319')
        # images['Dim'] = []
        for i in handles:
            im = io.imread('../Nuclei_20210319/'+i)
            im = im / im.max() * 255
            im = im / 255  # Normalization
            im = np.expand_dims(im, axis=0)
            im = np.expand_dims(im, axis=0)
            images['Image'].append(im)
            images['ID'].append(i)

        with open('../inputs/' + mode + '_0319.pickle', 'wb') as f:
            pickle.dump(images, f)
        with open('../inputs/' + mode + '_0319.pickle', 'rb') as f:
            images = pickle.load(f)
    return images


def mem_efficient_test(model, mode):
    if not os.path.exists(output):
        os.makedirs(output)
    handles = image_ids_in('../Nuclei_20210319')
    for i in handles:
        torch.cuda.empty_cache()
        im = io.imread('../Nuclei_20210319/' + i)
        im = im / im.max() * 255
        im = im / 255  # Normalization
        im = np.expand_dims(im, axis=0)
        im = np.expand_dims(im, axis=0)
        teim = im
        teid = i
        xt = Variable(torch.from_numpy(teim).type(torch.FloatTensor)).to(device)
        pred_mask = model(xt)
        pred_np = (F.sigmoid(pred_mask) > 0.625).cpu().data.numpy().astype(np.uint8)
        pred_np = mph.remove_small_objects(pred_np.astype(bool), min_size=500, connectivity=2).astype(np.uint8)
        if mode == 'nuke':
            pred_np = mph.remove_small_holes(pred_np, area_threshold=500, connectivity=2)
        io.imsave(output + '/pred_' + teid, ((pred_np / pred_np.max()) * 255).astype(np.uint8))


def test(tesample, model, mode):
    if not os.path.exists(output):
        os.makedirs(output)
    for itr in range(len(tesample['ID'])):
        torch.cuda.empty_cache()
        teim = tesample['Image'][itr]
        # print(np.shape(teim))
        teid = tesample['ID'][itr]
        xt = Variable(torch.from_numpy(teim).type(torch.FloatTensor)).to(device)
        pred_mask = model(xt)
        pred_np = (F.sigmoid(pred_mask) > 0.625).cpu().data.numpy().astype(np.uint8)
        pred_np = mph.remove_small_objects(pred_np.astype(bool), min_size=500, connectivity=2).astype(np.uint8)
        if mode == 'nuke':
            pred_np = mph.remove_small_holes(pred_np, area_threshold=500, connectivity=2)
        io.imsave(output + '/pred_' + teid, ((pred_np/pred_np.max())*255).astype(np.uint8))


if __name__ == '__main__':
    # sample = dataloader('test')

    model = UNet().to(device)
    a = torch.load(md)
    model.load_state_dict(a['state_dict'])

    # test(sample, model, 'nuke')
    mem_efficient_test(model, 'nuke')

