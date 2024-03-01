import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
    

def transunet_preprocess(img_array):
    # 将图像剪切到范围[-125, 275]
    img_array = np.clip(img_array, -125, 275)

    # 归一化到[0, 1]
    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))

    return img_array

class NiiDataset(Dataset):
    def __init__(self, transform, images_dir: str, mask_dir: str, split_list, split, zero_mask=True, read_disc = False):
        self.transform = transform
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.zero_mask = zero_mask
        
        self.ids = []
        for data_file in split_list:
            self.ids += open(data_file).readlines()


        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        print(f'Creating {split} NiiDataset with {len(self.ids)} examples')

        if read_disc:
            self.slices = torch.load('./slices.pt')
            self.masks = torch.load('./masks.pt')

        else: 
            self.slices = []
            self.masks = []

            for name in tqdm(self.ids):
                name = name.strip('\n')
                img_path = os.path.join(images_dir, name)

                name = name[:-7]
                name = name.replace('_image', '_mask')
                mask_path = next(self.mask_dir.glob(f'{name}*'))

                # 执行预处理
                img, mask = self.preprocess(img_path, mask_path, self.zero_mask)
                self.slices.extend(img)
                self.masks.extend(mask)
            
            #torch.save(self.slices, './slices.pt')
            #torch.save(self.masks, './masks.pt')



    def __len__(self):
        return len(self.slices)

    @staticmethod
    def preprocess(img_path, mask_path, zero_mask):
        # 读取原始图像和掩码图像
        img = sitk.ReadImage(str(img_path))
        mask = sitk.ReadImage(str(mask_path))

        # 将图像和掩码转换为NumPy数组
        img_array = sitk.GetArrayFromImage(img)

        # 【重要更改】增加归一化
        # img_array = img_array / 255.0
        # 【重要更改】增加TransUNET的预处理
        img_array = transunet_preprocess(img_array)

        mask_array = sitk.GetArrayFromImage(mask)

        # 确保图像和掩码的形状和通道数正确
        assert img_array.shape == mask_array.shape, f'Image and mask shape mismatch: {img_array.shape} vs {mask_array.shape}'

        img_tensors = []
        mask_tensors = []
        # 将NumPy数组转换为列表
        for i in range(img_array.shape[0]):
            # 【重要更改】将全0的mask去掉
            if not zero_mask and np.all(mask_array[i]==0):
                continue
            img_tensors.append(img_array[i])
            mask_tensors.append(mask_array[i])

        return img_tensors, mask_tensors

    def __getitem__(self, idx):
        img_array = self.slices[idx].astype(np.float32)
        mask_array = self.masks[idx].astype(np.float32)
        
        sample = {'image': img_array, 'label': mask_array}
        if self.transform:
            sample = self.transform(sample)
        return sample

class TetsNiiDataset(Dataset):
    def __init__(self, transform, images_dir: str, mask_dir: str, split_list, split, read_disc = False):
        self.transform = transform
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        
        self.ids = []
        for data_file in split_list:
            self.ids += open(data_file).readlines()


        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        print(f'Creating {split} NiiDataset with {len(self.ids)} examples')



    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(img_path, mask_path):
        # 读取原始图像和掩码图像
        img = sitk.ReadImage(str(img_path))
        mask = sitk.ReadImage(str(mask_path))

        # 将图像和掩码转换为NumPy数组
        # img_array = sitk.GetArrayFromImage(img).astype(np.float32)
        # mask_array = sitk.GetArrayFromImage(mask).astype(np.float32)
        img_array = sitk.GetArrayFromImage(img)

        # 【重要更改】增加归一化
        # img_array = img_array / 255.0
        # 【重要更改】增加TransUNET的预处理
        img_array = transunet_preprocess(img_array)

        mask_array = sitk.GetArrayFromImage(mask)

        # 确保图像和掩码的形状和通道数正确
        assert img_array.shape == mask_array.shape, f'Image and mask shape mismatch: {img_array.shape} vs {mask_array.shape}'

        return img_array, mask_array

    def __getitem__(self, idx):
        name = self.ids[idx]

        name = name.strip('\n')
        img_path = os.path.join(self.images_dir, name)

        name = name[:-7]
        name = name.replace('_image', '_mask')
        mask_path = next(Path(self.mask_dir).glob(f'{name}*'))

        # 执行预处理
        img, mask = self.preprocess(img_path, mask_path)

        img_array = img.astype(np.float32)
        mask_array = mask.astype(np.float32)
        
        sample = {'image': img_array, 'label': mask_array}
        if self.transform:
            sample = self.transform(sample)
        else:
            image = torch.from_numpy(img_array)
            label = torch.from_numpy(mask_array)
            sample = {'image': image, 'label': label}
        return sample
