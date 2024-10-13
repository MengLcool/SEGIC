import os
from os import path, replace

import torch
import json
import numpy as np
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

class YouTubeVOSTestDataset(Dataset):
    def __init__(self, data_root, split, transform):
        # self.image_dir = path.join(data_root, 'all_frames', split+'_all_frames', 'JPEGImages')
        self.image_dir = path.join(data_root, split, 'JPEGImages')
        self.mask_dir = path.join(data_root, split, 'Annotations')
        self.transform = transform

        self.vid_list = sorted(os.listdir(self.image_dir))
        self.req_frame_list = {}

        with open(path.join(data_root, split, 'meta.json')) as f:
            # read meta.json to know which frame is required for evaluation
            meta = json.load(f)['videos']

            for vid in self.vid_list:
                req_frames = []
                objects = meta[vid]['objects']
                for value in objects.values():
                    req_frames.extend(value['frames'])

                req_frames = list(set(req_frames))
                self.req_frame_list[vid] = req_frames

    def load_video(self, vid):
        image_dir_this = os.path.join(self.image_dir, vid)
        mask_dir_this = os.path.join(self.mask_dir, vid)
        frames = sorted(os.listdir(image_dir_this))
        first_gt_path = path.join(mask_dir_this, sorted(os.listdir(mask_dir_this))[0])
        
        
        frame_list = []
        for name in frames :
            frame_img = Image.open(os.path.join(image_dir_this, name)).convert('RGB')
            frame_list.append(frame_img)
        mask = Image.open(first_gt_path)
        mask = np.array(mask.convert('P'), dtype=np.uint8)
        
        return frame_list, mask, first_gt_path, frames
    
    def __getitem__(self, idx):
        # query_name, support_names, class_sample = self.sample_episode(idx)
        # query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()
        vid = self.vid_list[idx]
        frame_list, mask, mask_path, frames = self.load_video(vid)
        
        def to_tensor(x):
            return torch.tensor(np.array(x), dtype=torch.float32).permute(2,0,1)
        images = [to_tensor(x) for x in frame_list]
        images = torch.stack(images)
        mask = torch.tensor(mask)
        mask_ids = mask.unique()[1:]
        inst_masks = mask.unsqueeze(0).expand(len(mask_ids), -1, -1) == mask_ids.view(-1, 1, 1).int()

        sample = {'images': images,
                 'inst_masks': inst_masks * 255,
                 'inst_ids': mask_ids,
                 'vid': vid,
                 'frame_ids': frames,
                 'mask_path': mask_path
                 }

        if self.transform:
            sample = self.transform(sample)

        sample.update({
            'ori_inst_masks':inst_masks * 255,
        })

        return sample

    def __len__(self):
        return len(self.vid_list)
    
    
    
class DAVISTestDataset(Dataset):
    def __init__(self, data_root, imset, transform):
        if False:
            self.image_dir = path.join(data_root, 'JPEGImages', 'Full-Resolution')
            self.mask_dir = path.join(data_root, 'Annotations', 'Full-Resolution')
            if not path.exists(self.image_dir):
                print(f'{self.image_dir} not found. Look at other options.')
                self.image_dir = path.join(data_root, 'JPEGImages', '1080p')
                self.mask_dir = path.join(data_root, 'Annotations', '1080p')
            assert path.exists(self.image_dir), 'path not found'
        else :
            self.image_dir = path.join(data_root, 'JPEGImages', '480p')
            self.mask_dir = path.join(data_root, 'Annotations', '480p')

        self.transform = transform


        with open(path.join(data_root, 'ImageSets', imset)) as f:
            self.vid_list = sorted([line.strip() for line in f])
            # self.vid_list = sorted([line.strip().split('/')[-2] for line in f])
            # self.vid_list = sorted(list(set(self.vid_list)))

    def load_video(self, vid):
        image_dir_this = os.path.join(self.image_dir, vid)
        mask_dir_this = os.path.join(self.mask_dir, vid)
        frames = sorted(os.listdir(image_dir_this))
        first_gt_path = path.join(mask_dir_this, sorted(os.listdir(mask_dir_this))[0])
        
        
        frame_list = []
        for name in frames :
            frame_img = Image.open(os.path.join(image_dir_this, name)).convert('RGB')
            frame_list.append(frame_img)
        mask = Image.open(first_gt_path)
        mask = np.array(mask.convert('P'), dtype=np.uint8)
        
        return frame_list, mask, first_gt_path, frames
    
    def __getitem__(self, idx):
        # query_name, support_names, class_sample = self.sample_episode(idx)
        # query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()
        vid = self.vid_list[idx]
        frame_list, mask, mask_path, frames = self.load_video(vid)
        
        def to_tensor(x):
            return torch.tensor(np.array(x), dtype=torch.float32).permute(2,0,1)
        images = [to_tensor(x) for x in frame_list]
        images = torch.stack(images)
        mask = torch.tensor(mask)
        mask_ids = mask.unique()[1:]
        inst_masks = mask.unsqueeze(0).expand(len(mask_ids), -1, -1) == mask_ids.view(-1, 1, 1).int()

        sample = {'images': images,
                 'inst_masks': inst_masks * 255,
                 'inst_ids': mask_ids,
                 'vid': vid,
                 'frame_ids': frames,
                 'mask_path': mask_path
                 }

        if self.transform:
            sample = self.transform(sample)

        sample.update({
            'ori_inst_masks':inst_masks * 255,
        })

        return sample

    def __len__(self):
        return len(self.vid_list)