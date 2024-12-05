import cv2
import random
from PIL import Image
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
import os
import numpy as np
import torch
import torchvision.transforms
from torchvision.transforms import transforms

# PSCC train dataset path
a_path = '///'

# seed = 1
# random.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def data_aug(img, data_aug_ind):
    # img = Image.fromarray(img)
    if data_aug_ind == 0:
        return img
    elif data_aug_ind == 1:
        return img.rotate(90, expand=True)
    elif data_aug_ind == 2:
        return img.rotate(180, expand=True)
    elif data_aug_ind == 3:
        return img.rotate(270, expand=True)
    elif data_aug_ind == 4:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    elif data_aug_ind == 5:
        return img.rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM)
    elif data_aug_ind == 6:
        return img.rotate(180, expand=True).transpose(Image.FLIP_TOP_BOTTOM)
    elif data_aug_ind == 7:
        return img.rotate(270, expand=True).transpose(Image.FLIP_TOP_BOTTOM)
    else:
        raise Exception('Data augmentation-1:1:1:10-1:1:5:20 index is not applicable.')


def area_label(mask, size):
    pos_index = (mask == 1)
    neg_index = (mask == 0)

    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num

    label = pos_num / sum_num

    return label


def generate_offset(mask):
    offset = np.zeros((2, 224, 224), np.float)

    mask = (mask == 1)
    points = np.where(mask)

    center_x = int(np.mean(points[1]))
    center_y = int(np.mean(points[0]))

    for i in range(len(points[0])):
        # label[points[0][i]][points[1][i]] = 1
        offset[0][points[0][i]][points[1][i]] = (points[0][i] - center_y) / 224
        offset[1][points[0][i]][points[1][i]] = (points[1][i] - center_x) / 224
    return offset


def split_data(val_rate: float=0.2):
    image_path_train = []
    image_path_val = []

    with open('PSCC-Netdataset/copymove/fake.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            image_path_train.append(os.path.join('copymove', line))
    with open('PSCC-Netdataset/removal/fake.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            image_path_train.append(os.path.join('removal', line))
    with open('PSCC-Netdataset/splice/fake.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            image_path_train.append(os.path.join('splice', line))
    with open('PSCC-Netdataset/splice_randmask/fake.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            image_path_train.append(os.path.join('splice_randmask', line))

    with open('PSCC-Netdataset/copymove/fake_val.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            image_path_val.append(os.path.join('copymove', line))
    with open('PSCC-Netdataset/removal/fake_val.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            image_path_val.append(os.path.join('removal', line))
    with open('PSCC-Netdataset/splice/fake_val.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            image_path_val.append(os.path.join('splice', line))
    with open('PSCC-Netdataset/splice_randmask/fake_val.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            image_path_val.append(os.path.join('splice_randmask', line))

    random.shuffle(image_path_train)
    random.shuffle(image_path_val)

    return image_path_train, image_path_val


class MyDataset_train(Dataset):
    def __init__(self, path) -> None:
        self.image_path = path

        self.data_transform = torchvision.transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = torchvision.transforms.Compose([

        ])
    def __getitem__(self, item):
        image_path = self.image_path[item]
        i_path = os.path.join(a_path, image_path)

        aug_index = random.randrange(0, 8)
        image = Image.open(i_path).convert('RGB')
        image = data_aug(image, aug_index)
        image = self.data_transform(image)

        mask = Image.open(i_path.replace('fake', 'mask', 1).replace('.jpg', '.png').replace('.tif', '.png'))
        mask = data_aug(mask, aug_index)
        # mask = self.mask_transform(mask)
        mask = np.asarray(mask).astype(np.float32) / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        offset = torch.from_numpy(generate_offset(mask))
        mask = torch.from_numpy(mask)

        edge = cv2.imread(i_path.replace('fake', 'mask', 1).replace('.jpg', '.png').replace('.tif', '.png'),
                          cv2.IMREAD_GRAYSCALE)
        edge = cv2.Canny(edge, 100, 200)
        kernel = np.ones((4, 4), np.uint8)
        edge = (cv2.dilate(edge, kernel, iterations=1) > 1) * 1.0
        edge = Image.fromarray(edge)
        edge = data_aug(edge, aug_index)
        # edge = self.mask_transform(edge)
        img_convert_to_numpy = np.array(edge)
        img_convert_to_numpy = np.where(img_convert_to_numpy > 0, 1, 0)  # (32, 32, 3)
        boundry = torch.tensor(img_convert_to_numpy)

        return image, mask, offset, boundry

    def __len__(self):
        return len(self.image_path)


class MyDataset_val(Dataset):
    def __init__(self, path) -> None:
        self.image_path = path

        self.data_transform = torchvision.transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = torchvision.transforms.Compose([

        ])

    def __getitem__(self, item):
        image_path = self.image_path[item]
        i_path = os.path.join(a_path, image_path)

        image = Image.open(i_path).convert('RGB')
        image = self.data_transform(image)

        mask = Image.open(i_path.replace('fake', 'mask', 1).replace('.jpg', '.png').replace('.tif', '.png'))
        # mask = self.mask_transform(mask)
        mask = np.asarray(mask).astype(np.float32) / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        offset = torch.from_numpy(generate_offset(mask))
        mask = torch.from_numpy(mask)

        edge = cv2.imread(i_path.replace('fake', 'mask', 1).replace('.jpg', '.png').replace('.tif', '.png'),
                          cv2.IMREAD_GRAYSCALE)
        edge = cv2.Canny(edge, 100, 200)
        kernel = np.ones((4, 4), np.uint8)
        edge = (cv2.dilate(edge, kernel, iterations=1) > 1) * 1.0
        edge = Image.fromarray(edge)
        # edge = self.mask_transform(edge)
        img_convert_to_numpy = np.array(edge)
        img_convert_to_numpy = np.where(img_convert_to_numpy > 0, 1, 0)  # (32, 32, 3)
        boundry = torch.tensor(img_convert_to_numpy)

        return image, mask, offset, boundry

    def __len__(self):
        return len(self.image_path)


train_set, val_set = split_data()

train_dataset = MyDataset_train(train_set)
val_dataset = MyDataset_val(val_set)

def getDataLoader(batchSize):

    trainloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True,pin_memory=True,num_workers=4, drop_last=True)
    testLoader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True,num_workers=4, drop_last=True)
    return trainloader, testLoader

