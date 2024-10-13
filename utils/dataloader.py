# Copyright by HQ-SAM team
# All rights reserved.

## data loader
from __future__ import print_function, division
from typing import Any

import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict
from PIL import Image

def custom_collate_fn(data):
    from torch.utils.data import default_collate
    custom_collate_data = defaultdict(list)
    keys = set(data[0].keys())
    for data_this in data :
        assert set(data_this.keys()) == keys, '{} / {}'.format(set(data_this.keys())- keys, keys-set(data_this.keys()))
        for k in keys:
            if 'label' in k or k in ['neg_class_names']:
                v = data_this.get(k)
                custom_collate_data[k].append(v)
        for k in custom_collate_data.keys():
            data_this.pop(k)

    data = default_collate(data)
    data.update(**custom_collate_data)
    return data

#### --------------------- dataloader online ---------------------####

def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        tmp_im_list, tmp_gt_list = [], []
        tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))

        if(datasets[i]["gt_dir"]==""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
            print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))


        name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                "im_path":tmp_im_list,
                                "gt_path":tmp_gt_list,
                                "im_ext":datasets[i]["im_ext"],
                                "gt_ext":datasets[i]["gt_ext"]})

    return name_im_gt_list

def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False):
    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8

    if training:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms))
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        sampler = DistributedSampler(gos_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True)
        dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

        gos_dataloaders = dataloader
        gos_datasets = gos_dataset

    else:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
            sampler = DistributedSampler(gos_dataset, shuffle=False)
            dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets

class DualAug():
    def __init__(self, aug_list, aug_list2=None):
        self.aug_list1 = transforms.Compose(aug_list)
        self.aug_list2 = transforms.Compose(aug_list2 if aug_list2 is not None else aug_list)

    
    def __call__(self, sample) -> Any:
        sample_copy = deepcopy(sample)
        sample1 = self.aug_list1(sample)
        if 'image_dual' in sample:
            imidx, image_dual, label_dual, shape = sample['imidx'], sample['image_dual'], sample['label_dual'], sample['shape']
            sample2 = {'imidx':imidx,'image':image_dual, 'label':label_dual, 'shape':shape}
            sample2 = self.aug_list2(sample2)
        else :
            sample2 = self.aug_list2(sample_copy)

        valid_keys = {'imidx','image','label','shape', 'ori_size'}
        sample1.update({'{}_dual'.format(k):v for k,v in sample2.items() if k in valid_keys})
        return sample1

class RandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image,dims=[2])
            label = torch.flip(label,dims=[2])

        result = {'imidx':imidx,'image':image, 'label':label, 'shape':shape}
        # result.update(class_name=sample.get('class_name', ''))
        sample.update(result)
        return sample

class ResizeVOS(object):
    def __init__(self,size=[320,320]):
        assert size[0] == size[1]
        self.size = size

    def __call__(self, sample):
        images, inst_masks = sample['images'], sample['inst_masks']
        size = max(self.size)
        scale_factor = size / max(images.shape[-2:])
        images = F.interpolate(images, scale_factor=scale_factor, mode='bilinear')
        inst_masks = F.interpolate(inst_masks[None].float(), scale_factor=scale_factor, mode='bilinear')[0]
        padding_h = max(size - images.size(-2), 0)
        padding_w = max(size - images.size(-1), 0)
        ori_size_prepad = images.shape[-2:]
        images = F.pad(images, [0,padding_w, 0,padding_h],value=128)
        inst_masks = F.pad(inst_masks, [0,padding_w, 0,padding_h],value=0)
        sample['ori_size'] = torch.tensor(ori_size_prepad)
        sample.update(
            images=images,
            inst_masks=inst_masks
        )

        return sample


class Resize(object):
    def __init__(self,size=[320,320]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        size = max(self.size)
        scale_factor = size / max(image.shape[-2:])
        if image.shape[-2:] != label.shape[-2:] :
            print(imidx)
        image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0), scale_factor=scale_factor, mode='bilinear'),dim=0)
        label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0), scale_factor=scale_factor,mode='bilinear'),dim=0)
        padding_h = max(size - image.size(1), 0)
        padding_w = max(size - image.size(2), 0)
        ori_size_prepad = image.shape[-2:]
        image = F.pad(image, [0,padding_w, 0,padding_h],value=128)
        label = F.pad(label, [0,padding_w, 0,padding_h],value=0)
        # image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),self.size,mode='bilinear'),dim=0)
        # label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),self.size,mode='bilinear'),dim=0)

        aug_sample= {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}
        aug_sample.update(class_name=sample.get('class_name', ''))
        aug_sample['ori_size'] = torch.tensor(ori_size_prepad)
        if 'image_dual' in sample:
            # image_dual, label_dual = sample['image_dual'], sample['label_dual']
            # aug_sample['image_dual'] = torch.squeeze(F.interpolate(torch.unsqueeze(image_dual,0),self.size,mode='bilinear'),dim=0)
            # aug_sample['label_dual'] = torch.squeeze(F.interpolate(torch.unsqueeze(label_dual,0),self.size,mode='bilinear'),dim=0)
            # aug_sample['label_dual'] = torch.squeeze(F.interpolate(torch.unsqueeze(label_dual,0),self.size,mode='bilinear'),dim=0)
            # aug_sample['ori_size_dual'] = torch.tensor(sample['image_dual'].shape[-2:])
            image_dual, label_dual = sample['image_dual'], sample['label_dual']
            scale_factor = size / max(image_dual.shape[-2:])
            image_dual = torch.squeeze(F.interpolate(torch.unsqueeze(image_dual,0), scale_factor=scale_factor, mode='bilinear'),dim=0)
            label_dual = torch.squeeze(F.interpolate(torch.unsqueeze(label_dual,0), scale_factor=scale_factor,mode='bilinear'),dim=0)
            padding_h = max(size - image_dual.size(1), 0)
            padding_w = max(size - image_dual.size(2), 0)
            ori_size_prepad = image_dual.shape[-2:]
            image_dual = F.pad(image_dual, [0,padding_w, 0,padding_h],value=128)
            label_dual = F.pad(label_dual, [0,padding_w, 0,padding_h],value=0)

            aug_sample['image_dual'] = image_dual
            aug_sample['label_dual'] = label_dual
            aug_sample['ori_size_dual'] = torch.tensor(ori_size_prepad)
        
        sample.update(aug_sample)
        return sample

class RandomCrop(object):
    def __init__(self,size=[288,288]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:,top:top+new_h,left:left+new_w]
        label = label[:,top:top+new_h,left:left+new_w]

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}


class Normalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):

        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image,self.mean,self.std)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}

class DefaultBundle():
    def __call__(self, sample):
        if 'is_box_mask' not in sample:
            sample['is_box_mask'] = False
        if 'neg_class_names' not in sample:
            sample['neg_class_names'] = []

        return sample


class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
    """

    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, sample):
        imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        #resize keep ratio
        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()
        
        scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        
        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=128)
        label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)
        ori_size = torch.tensor(scaled_image.shape[-2:])

        result = {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(image.shape[-2:]), 'ori_size':ori_size}
        # result.update(class_name=sample.get('class_name', ''))
        sample.update(result)
        return sample


class EvalDataProcessor():
    def __init__(self, size=[320,320]) -> None:
        self.resize = Resize(size)

    def __call__(self, data_path, mask_path=None, batchify=True) -> Any:
        img = torch.tensor(np.array(Image.open(data_path).convert('RGB')), dtype=torch.float32).permute(2,0,1) #(3, h, w)
        if mask_path is not None:
            mask = torch.tensor(np.array(Image.open(mask_path).convert('L')), dtype=torch.float32)[None] #(1, h, w)
        else:
            mask = torch.zeros_like(img)[:1]

        shape = img.shape[-2:]
        imidx = os.path.split(data_path)[-1].split('.')[0]
        sample = dict(
            imidx=imidx,
            image=img,
            label=mask,
            shape=shape
        )

        sample['ori_label'] = mask
        sample = self.resize(sample)
        # FIXME: batchify the sample
        if batchify:
            sample = custom_collate_fn([sample])
        return sample

class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):

        self.transform = transform
        self.dataset = {}
        ## combine different datasets into one
        dataset_names = []
        dt_name_list = [] # dataset name per image
        im_name_list = [] # image name
        im_path_list = [] # im path
        gt_path_list = [] # gt path
        im_ext_list = [] # im ext
        gt_ext_list = [] # gt ext
        for i in range(0,len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])


        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

        self.eval_ori_resolution = eval_ori_resolution

    def __len__(self):
        return len(self.dataset["im_path"])
    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        im = io.imread(im_path)
        gt = io.imread(gt_path)

        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im,1,2),0,1)
        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32),0)

        sample = {
        "imidx": torch.from_numpy(np.array(idx)),
        "image": im,
        "label": gt,
        "shape": torch.tensor(im.shape[-2:]),
        }
        
        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
            sample['ori_im_path'] = self.dataset["im_path"][idx]
            sample['ori_gt_path'] = self.dataset["gt_path"][idx]

        return sample