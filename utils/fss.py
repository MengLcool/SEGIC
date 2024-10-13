r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


r""" COCO-20i few-shot semantic segmentation dataset """
import os
import pickle
import random
import json

from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from pycocotools.coco import COCO
from collections import defaultdict
from tempfile import NamedTemporaryFile

class SemADE(Dataset):
    def __init__(self, base_image_dir, transform, is_train=True, dataset_name='ade20k', is_semseg=False, ext='png', is_meta=False):
        self.transform = transform
        self.max_inst = 1
        self.ext = ext
        self.split = split = 'training' if is_train else 'validation'
        self.is_semseg = is_semseg
        self.zero_start = False
        self.is_meta = is_meta
        with open("utils/dataset/{}_classes.json".format(dataset_name.replace('sd_', '')), "r") as f:
            classes_name_list = json.load(f)
        if dataset_name in ('ade20k'):
            self.ignore_idx = 255
            self.img_root = os.path.join(base_image_dir, "images", split)
            self.anno_root = os.path.join(base_image_dir, "annotations", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name in ('sd_ade20k'):
            self.ignore_idx = 255
            self.img_root = os.path.join(base_image_dir, "images_detectron2", split)
            self.anno_root = os.path.join(base_image_dir, "annotations_detectron2", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name == 'cocostuff':
            self.img_root = os.path.join(base_image_dir, "images", split)
            self.anno_root = os.path.join(base_image_dir, "annotations", split)
            classes_name_list = [x.split(':')[-1] for x in classes_name_list]
        elif dataset_name == 'ade847':
            self.ignore_idx = 65535
            self.img_root = os.path.join(base_image_dir, "images_detectron2", split)
            self.anno_root = os.path.join(base_image_dir, "annotations_detectron2", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name == 'sd_ade847':
            self.zero_start = False
            self.ignore_idx = 65535
            self.img_root = os.path.join(base_image_dir, "images_detectron2", split)
            self.anno_root = os.path.join(base_image_dir, "annotations_detectron2", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name == 'pc459':
            assert not is_train
            self.zero_start = True
            self.split = split = 'training' if is_train else 'validation'
            self.img_root = os.path.join(base_image_dir, 'JPEGImages')
            self.anno_root = os.path.join(base_image_dir, 'annotations_detectron2/pc459_val')
        else :
            raise NotImplementedError

        self.cid_to_cat = np.array(classes_name_list)
        self.all_cls = set(list(range(len(self.cid_to_cat)))) 
        if not self.zero_start:
            self.all_cls = self.all_cls- {0}
        self.class_ids = self.get_class_ids()
        file_names = sorted(
            os.listdir(self.img_root)
        )
        
        gt_names = os.listdir(self.anno_root)
        if len(file_names) != len(gt_names):
            print('warning, not equal')
            file_names = [x[:-len('.jpg')] for x in file_names]
            gt_names = [x[:-(len(self.ext)+1)] for x in gt_names]
            intersect = list(set(file_names) & set(gt_names))
            file_names = [x+'.jpg' for x in intersect]
            
        image_ids = []
        for x in file_names:
            if x.endswith(".jpg"):
                image_ids.append(x[:-4])
        self.image_ids = []

        meta_path = "utils/dataset/{}_{}_icl.pth".format('train' if is_train else 'val', dataset_name)
        if not os.path.exists(meta_path):
            meta_info = defaultdict(list)
            for img_id in tqdm(image_ids) :
                anno_path = os.path.join(self.anno_root, '{}.{}'.format(img_id, self.ext))
                label = Image.open(anno_path)
                if self.ext != 'tif':
                    label = label.convert('L')
                uni_cids = np.unique(np.asarray(label))
                if not self.zero_start:
                    if uni_cids[0] == 0:
                        uni_cids = uni_cids[1:]
                    if uni_cids[-1] == self.ignore_idx:
                        uni_cids = uni_cids[:-1]
                # if len(uni_cids) >= 1 and self.zero_start or len(uni_cids) > 1:
                if len(uni_cids) >= 1 :
                    self.image_ids.append(img_id)
                for cid in uni_cids:
                    if cid in self.class_ids:
                        meta_info[cid].append(img_id)
            self.meta_info = meta_info
            torch.save([self.meta_info, self.image_ids], meta_path)
        else :
            self.meta_info, self.image_ids = torch.load(meta_path)

    def get_meta(self, idx):
        cid = self.class_ids[idx]
        ref_img_id = self._get_ref_cid(cid, None)
        if ref_img_id is not None:
            ref_id = self.image_ids.index(ref_img_id)
        else :
            ref_id = 3
        return ref_id        

        # outputs = []
        # for cid in self.get_class_ids():
        #     ref_img_id = self._get_ref_cid(cid, None)
        #     if ref_img_id is not None:
        #         ref_id = self.image_ids.index(ref_img_id)
        #     else :
        #         ref_id = 3
        #         self.image_ids[0]
        #     outputs.append(self.__getitem__(ref_id, [cid]))
        # return outputs

    def get_class_ids(self,):
        return np.array(sorted(list(self.all_cls)))

    def get_class_names(self,):
        cls_ids = self.get_class_ids()
        return [self.cid_to_cat[x] for x in cls_ids]

    def _get_info(self, img_id, cats_list=None, ret_uni_cids=None, sample_max_inst=True):
        # print('img', img_id)
        image = Image.open(os.path.join(self.img_root, '{}.jpg'.format(img_id))).convert('RGB')
        masks = Image.open(os.path.join(self.anno_root, '{}.{}'.format(img_id, self.ext)))
        if self.ext != 'tif':
            masks = masks.convert('L')
        masks = np.array(masks)
        uni_cids = np.unique(masks)
        if not self.zero_start:
            if uni_cids[0] == 0:
                uni_cids = uni_cids[1:]
            if uni_cids[-1] == self.ignore_idx:
                uni_cids = uni_cids[:-1]

        if cats_list is None :
            if sample_max_inst and len(uni_cids) > self.max_inst :
                cats_list = np.random.choice(
                    uni_cids, size=self.max_inst, replace=False
                ).tolist()
            else :
                cats_list = uni_cids.tolist()
        else :
            uni_cids = np.array(cats_list)

        masks_list = []
        for cid in cats_list :
            masks_list.append(masks==cid)

        masks = np.stack(masks_list)
        def to_tensor(x):
            return torch.tensor(np.array(x), dtype=torch.float32).permute(2,0,1)
        image = to_tensor(image)
        masks = torch.tensor(masks).float()

        if ret_uni_cids :
            return image, masks, cats_list, uni_cids
        return image, masks, cats_list

    def __len__(self,):
        if self.is_meta:
            return len(self.class_ids)
        return len(self.image_ids)

    def _get_ref_cid(self, cid, index):
        idx_list = self.meta_info[cid]
        ref_index = index
        
        if not len(idx_list):
            return None

        if len(idx_list) > 1 or ref_index is None:
            while ref_index == index:
                ref_index = random.choice(idx_list)
        # else:
        #     return self._get_ref_cid(1, None)
        return ref_index

    def __getitem__(self, index, cat_id=None) :
        if self.is_meta :
            cat_id = [self.class_ids[index]]
            index = self.get_meta(index)

        img_id = self.image_ids[index]
        image, masks, cats_list, uni_cids = self._get_info(img_id, cat_id, ret_uni_cids=True)

        image_ref_list, masks_ref_list = [], []

        sample = {'image': image,
                'label': masks * 255,
                "imidx": torch.from_numpy(np.array(index)),
                "shape": torch.tensor(image.shape[-2:]),
                "class_name": self.cid_to_cat[cats_list[0]] if cats_list[0] < len(self.cid_to_cat) else self.cid_to_cat[0],
                'is_inst': False,
                }
        
        if self.is_meta:
            sample.update(class_id=cat_id[0])

        if cat_id is None :
            for cid in cats_list:
                ref_img_id = self._get_ref_cid(cid, img_id)
                image_ref, masks_ref, _ = self._get_info(ref_img_id, [cid])
                image_ref_list.append(image_ref)
                masks_ref_list.append(masks_ref)

            # FIXME: only support num_inst == 1
            image_ref, masks_ref = image_ref_list[0], masks_ref_list[0]
            sample.update({
                'image_dual': image_ref,
                'label_dual': masks_ref * 255,
            })

            # return super().__getitem__(index)
            sample['neg_class_names'] = [self.cid_to_cat[cid] for cid in (self.all_cls - set(uni_cids.tolist()))]

        if self.transform:
            sample = self.transform(sample)

        if self.is_semseg and not self.is_meta:
            _, masks, cat_ids = self._get_info(img_id, sample_max_inst=False)
            all_cats = np.array(self.get_class_ids())
            cat_ids = np.array(cat_ids) #(ninst, )
            # if not self.zero_start:
            #     cat_ids -= 1    
            # semmask = np.zeros((len(all_cats), *masks.shape[-2:]))
            # semmask[cat_ids] = masks
            all_cats_this = cat_ids[None] == all_cats[:, None]
 
            # ninst, h, w = masks.shape
            # semmask = np.zeros((len(all_cats), *masks.shape[-2:]))
            # semmask[all_cats_this.sum(-1)>0] = masks.numpy()
            semmask = masks
            sample['origin_semmask'] = semmask
            sample['valid_cids'] = all_cats_this.sum(-1)>0



        if False :
            import cv2
            cv2.imwrite('tmp.jpg', image.permute(1,2,0).int().numpy())
            cv2.imwrite('tmp.jpg', sample['image'].permute(1,2,0).int().numpy())
            cv2.imwrite('tmp.jpg', sample['image_dual'].permute(1,2,0).int().numpy())
            pass
        return sample


class SemCOCO(Dataset):
    def __init__(self, base_image_dir, transform, is_train=True, is_lvis=False, 
                    custom_json_path=None, is_semseg=False, is_meta=False, fold=None,
                    json_path=None):
        self.transform = transform
        self.is_lvis = is_lvis
        self.is_semseg = is_semseg
        self.is_meta = is_meta
        self.is_train = is_train
        self.fold = fold
        if json_path is not None:
            self.img_root = base_image_dir
        else:
            split = 'train2017' if is_train else 'val2017'
            json_path = 'annotations/instances_{}.json'.format(split)
            if is_lvis :
                split = 'train2017' if is_train else 'val2017'
                split_json = 'train' if is_train else 'val'
                json_path = 'lvis_v1_{}.json'.format(split_json)
                self.img_root = base_image_dir
            else :
                split = 'train2017' if is_train else 'val2017'
                json_path = 'annotations/instances_{}.json'.format(split)
                self.img_root = os.path.join(base_image_dir, split)
            if custom_json_path is not None :
                json_path = custom_json_path

        if fold is not None and is_lvis:
            assert is_train == False
            assert fold < 10

            with open(os.path.join(base_image_dir, json_path)) as f:
                lvis_json = json.load(f)
                cat_count = defaultdict(set)
                for info in lvis_json['annotations']:
                    cat_count[info['category_id']].add(info['image_id'])
            cats = [x for x,v in cat_count.items() if len(v) > 1]
            cats = sorted(cats)

            idx = range(fold, len(cats), 10)
            cats = [cats[x] for x in idx]
            cats_set = set(cats)
            new_annotations = [x for x in lvis_json['annotations'] if x['category_id'] in cats_set]
            new_image_ids = set([x['image_id'] for x in new_annotations])
            new_images = [x for x in lvis_json['images'] if x['id'] in new_image_ids]
            lvis_json['annotations'] = new_annotations
            lvis_json['images'] = new_images
            with NamedTemporaryFile('w+t') as f:
                json.dump(lvis_json, f)
                print('filename is:', f.name)
                f.flush()
                self.coco = COCO(
                    f.name
                )
        else :
            self.coco = COCO(os.path.join(base_image_dir, json_path))

        ids = list(sorted(self.coco.imgs.keys()))
        self.ids = []
        self.max_inst = 1

        self.cid_to_cat = {k:v['name'] for k,v in self.coco.cats.items()}
        self.cls_to_idx = defaultdict(list)
        self.class_ids = self.get_class_ids()
        for idx in tqdm(ids) :
            if len(self.coco.getAnnIds(idx)):
                self.ids.append(idx)
                annos = self.coco.loadAnns(self.coco.getAnnIds(idx))
                cat_ids = np.array([x['category_id'] for x in annos])
                uni_cats = np.unique(cat_ids)
                for cid in uni_cats:
                    self.cls_to_idx[cid].append(len(self.ids)-1) # append this idx

    def _get_ref_cid(self, cid, index):
        idx_list = self.cls_to_idx[cid]
        ref_index = index
        idx_list = list(set(idx_list) - {index})
        if len(idx_list) > 1 :
            ref_index = random.choice(idx_list)
        return ref_index

    def __len__(self,):
        if self.fold is not None and not self.is_train:
            if self.is_lvis:
                return 2300
            return max(len(self.ids) * 5, 100)
        return len(self.ids)
    
    def get_class_ids(self,):
        return np.array(sorted(self.coco.cats.keys()))

    def get_class_names(self,):
        cls_ids = sorted(self.coco.cats.keys())
        return [self.coco.cats[x]['name'] for x in cls_ids]
    
    def __getitem__(self, index):
        if self.fold is not None and not self.is_train:
            index = random.randint(0, len(self.ids) - 1)
            # self.ids[index]
        
        assert self.max_inst == 1
        def _get_info(index, cats_list=None):
            idx = self.ids[index]
            if self.is_lvis :
                coco_url = self.coco.loadImgs(idx)[0]["coco_url"]
                image_path = os.path.join(*coco_url.split('/')[-2:])
            else :
                image_path = self.coco.loadImgs(idx)[0]["file_name"]

            image = Image.open(os.path.join(self.img_root, image_path)).convert('RGB')
            annos = self.coco.loadAnns(self.coco.getAnnIds(idx))
            masks = np.stack([self.coco.annToMask(x) for x in annos])
            cat_ids = np.array([x['category_id'] for x in annos])
            uni_cats = np.unique(cat_ids)
            masks_list = []
            if cats_list is None :
                if len(uni_cats) > self.max_inst :
                    cats_list = np.random.choice(
                        uni_cats, size=self.max_inst, replace=False
                    ).tolist()
                else :
                    cats_list = uni_cats.tolist()

            for cat in cats_list :
                masks_list.append(masks[cat_ids==cat].max(0))

            masks = np.stack(masks_list)
            def to_tensor(x):
                return torch.tensor(np.array(x), dtype=torch.float32).permute(2,0,1)
            image = to_tensor(image)
            masks = torch.tensor(masks).float()

            return image, masks, cats_list

        if self.fold is not None and not self.is_train:
            cats_list = [random.choice(list(self.cls_to_idx.keys()))]
            index = self._get_ref_cid(cats_list[0], None)
            image, masks, _ = _get_info(index, cats_list)
        else:
            image, masks, cats_list = _get_info(index)
            

        image_ref_list, masks_ref_list = [], []
        for cid in cats_list:
            ref_index = self._get_ref_cid(cid, index)
            image_ref, masks_ref, _ = _get_info(ref_index, [cid])
            image_ref_list.append(image_ref)
            masks_ref_list.append(masks_ref)

        # FIXME: only support num_inst == 1
        image_ref, masks_ref = image_ref_list[0], masks_ref_list[0]
        masks_ori = masks

        sample = {'image': image,
                 'label': masks * 255,
                 'image_dual': image_ref,
                 'label_dual': masks_ref * 255,
                 "imidx": torch.from_numpy(np.array(index)),
                 "shape": torch.tensor(image.shape[-2:]),
                 "class_name": self.cid_to_cat[cats_list[0]],
                 'is_inst': False
                 }

        image_ref_list = torch.stack(image_ref_list)
        masks_ref_list = torch.stack(masks_ref_list)

        if self.is_meta:
            sample.update(class_id=cats_list[0])

        if self.transform:
            sample = self.transform(sample)

        if self.is_semseg:
            all_cats = np.array(sorted(self.coco.cats.keys()))
            img_id = self.ids[index]
            annos = self.coco.loadAnns(self.coco.getAnnIds(img_id))
            masks = np.stack([self.coco.annToMask(x) for x in annos]) #(ninst, h, w)
            cat_ids = np.array([x['category_id'] for x in annos]) #(ninst, )
            all_cats_this = cat_ids[None] == all_cats[:, None]
 
            all_cats_this = cat_ids[None] == all_cats[:, None]
            ninst, h, w = masks.shape
            semmask = (all_cats_this @ masks.reshape(ninst, -1)).reshape(-1, h, w).clip(max=1)
            sample['origin_semmask'] = semmask

        if not self.is_train:
            sample.update({
                'ori_label':masks_ori * 255,
                'class_id': torch.tensor(cats_list[0])
            })


        if False :
            import cv2
            cv2.imwrite('tmp.jpg', image.permute(1,2,0).int().numpy())
            cv2.imwrite('tmp.jpg', sample['image'].permute(1,2,0).int().numpy())
            cv2.imwrite('tmp.jpg', sample['image_dual'].permute(1,2,0).int().numpy())
            pass
        return sample


class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.base_path = os.path.join(datapath, 'COCO2014')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()
        cid2name = []
        with open("utils/coco80.txt") as f:
            for line in f.readlines():
                cid2name.append(line.strip())
        self.cid2name = np.array(cid2name)

        self.img2cls = defaultdict(set)
        self.all_cls = set(self.img_metadata_classwise.keys())
        for cid, img_list in self.img_metadata_classwise.items():
            for img in img_list :
                self.img2cls[img].add(cid)

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        # query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()

        def to_tensor(x):
            return torch.tensor(np.array(x), dtype=torch.float32).permute(2,0,1)
        assert len(support_imgs) == 1 and len(support_masks) == 1
        support_imgs, support_masks = support_imgs[0], support_masks[0]
        query_img, support_imgs = [to_tensor(x) for x in [query_img, support_imgs]]
        query_mask, support_masks = query_mask[None].float(), support_masks[None].float()

        sample = {'image': query_img,
                 'label': query_mask * 255,
                 'image_dual': support_imgs,
                 'label_dual': support_masks * 255,
                 "imidx": torch.from_numpy(np.array(idx)),
                 "shape": torch.tensor(query_img.shape[-2:]),
                 'class_name': self.cid2name[class_sample],
                 'is_inst': False
                 }

        if self.transform:
            sample = self.transform(sample)

        sample['neg_class_names'] = [self.cid2name[cid] for cid in (self.all_cls - self.img2cls[query_name])]

        if self.split in ['val', 'test']:
            sample.update({
                'ori_label':query_mask * 255,
                'class_id': torch.tensor(class_sample)
            })

        return sample


    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        if self.fold is not None :
            class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
            class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]    
            class_ids = class_ids_trn if self.split == 'trn' else class_ids_val
        else :
            class_ids = list(range(self.nclass))

        return class_ids

    def build_img_metadata_classwise(self):
        if self.fold is not None :
            with open('./data/splits/coco/%s/fold%d.pkl' % (self.split, self.fold), 'rb') as f:
                img_metadata_classwise = pickle.load(f)
        else :
            assert self.split == 'trn'
            with open('./data/splits/coco/%s/all.pkl' % (self.split), 'rb') as f:
                img_metadata_classwise = pickle.load(f)

        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize



class DatasetFSS(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = split
        self.benchmark = 'fss'
        self.shot = shot

        self.base_path = os.path.join(datapath, 'FSS-1000')

        # Given predefined test split, load randomly generated training/val splits:
        # (reference regarding trn/val/test splits: https://github.com/HKUSTCV/FSS-1000/issues/7))
        with open('./data/splits/fss/%s.txt' % split, 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = sorted(self.categories)

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()

        self.transform = transform

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        # if True:
        #     # import time
        #     # random.seed(int(time.time()))
        #     idx_x = random.choice(list(range(len(self))))
        #     _, support_names, _ = self.sample_episode(idx_x)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        def to_tensor(x):
            return torch.tensor(np.array(x), dtype=torch.float32).permute(2,0,1)
        assert len(support_imgs) == 1 and len(support_masks) == 1
        support_imgs, support_masks = support_imgs[0], support_masks[0]
        query_img, support_imgs = [to_tensor(x) for x in [query_img, support_imgs]]
        query_mask, support_masks = query_mask[None].float(), support_masks[None].float()
        if query_img.shape[-2:] != query_mask.shape[-2:] or support_imgs[0].shape[-2:] != support_masks[0].shape[-2:]:
            # bugs caused by mismatch of gt and img
            return self.__getitem__(idx+1)

        sample = {'image': query_img,
                 'label': query_mask * 255,
                 'image_dual': support_imgs,
                 'label_dual': support_masks * 255,
                 "imidx": torch.from_numpy(np.array(idx)),
                 "shape": torch.tensor(query_img.shape[-2:]),
                 'class_name': support_names[0].split('/')[-2],
                 'is_inst': False
                 }

        if self.transform:
            sample = self.transform(sample)

        if self.split in ('val', 'test'):
            sample.update({
                'ori_label':query_mask * 255,
                'class_id': torch.tensor(class_sample)
            })

        return sample

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_id = query_name.split('/')[-1].split('.')[0]
        query_name = os.path.join(os.path.dirname(query_name), query_id) + '.png'
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        support_names = [os.path.join(os.path.dirname(name), sid) + '.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        class_sample = self.categories.index(query_name.split('/')[-2])
        if self.split == 'val':
            class_sample += 520
        elif self.split == 'test':
            class_sample += 760

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(range(1, 11), 1, replace=False)[0]
            support_name = os.path.join(os.path.dirname(query_name), str(support_name)) + '.jpg'
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        if self.split == 'trn':
            class_ids = range(0, 520)
        elif self.split == 'val':
            class_ids = range(520, 760)
        elif self.split == 'test':
            class_ids = range(760, 1000)
        return class_ids

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata.append(img_path)
        return img_metadata