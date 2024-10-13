import os
import copy
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import time
import random
from PIL import Image
from typing import Dict, List, Tuple
from collections import OrderedDict, defaultdict
from typing import Sequence

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter, DualAug, ResizeVOS, DefaultBundle
from utils.dataset import CustomConcatDataset
from utils.lr_sched import adjust_learning_rate
from utils.loss_mask import loss_masks
from utils.dataloader import transforms, DistributedSampler, DataLoader, custom_collate_fn, EvalDataProcessor
import utils.misc as misc
from utils.logger import get_logger
from detectron2.layers import ROIAlign
from utils.fss import DatasetFSS, DatasetCOCO, SemCOCO, SemADE
from utils.fss_inst import InstCOCO, get_inst_aug

from model.segic import build_model


def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

    parser.add_argument("--output", type=str, required=True, 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--use_cos_sched', action='store_true')
    parser.add_argument('--warmup_epochs', default=0, type=float)
    parser.add_argument('--min_lr', default=1e-7, type=float)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=12, type=int)
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--input_keys', type=str, nargs='+', default=['box','point','noise_mask'])
    parser.add_argument('--eval_keys', type=str, nargs='+', default=['box'])
    parser.add_argument('--eval_datasets', type=str, default=None)
    parser.add_argument('--n_point', type=int, default=1)
    parser.add_argument('--noised_inst', action='store_true')
    parser.add_argument('--use_dual_aug', action='store_true')
    parser.add_argument('--use_simm_prompt', action='store_true')
    parser.add_argument('--use_ref_decoder', action='store_true')
    parser.add_argument('--use_ref_refine_img', action='store_true')
    parser.add_argument('--use_bbox_head', action='store_true')
    parser.add_argument('--use_ref_keypoint', action='store_true')
    parser.add_argument('--use_corr_prompt', action='store_true')
    parser.add_argument('--open_ft', action='store_true')

    parser.add_argument('--use_dift', action='store_true')
    parser.add_argument('--encoder_model', type=str, default='dift')
    parser.add_argument('--dinov2_model', type=str, default='l')
    parser.add_argument('--use_inst_proj', action='store_true')
    parser.add_argument('--diff_text_prompt_ratio', default=1., type=float)
    parser.add_argument('--use_keypoint', action='store_true')
    parser.add_argument('--num_keypoint', default=1, type=int)
    parser.add_argument('--no_text_eval', action='store_true')
    parser.add_argument('--no_text', action='store_true')
    parser.add_argument('--eval_vos', action='store_true')
    parser.add_argument('--vos_dataset', type=str, default='davis17')
    parser.add_argument('--vos_gt_root', type=str, default='data/DAVIS/2017/Annotations/480p/')

    parser.add_argument('--eval_semseg', action='store_true')
    parser.add_argument('--custom_eval', action='store_true')
    parser.add_argument('--custom_eval_with_instance', action='store_true')
    parser.add_argument('--no_sim_prompt', action='store_true')
    parser.add_argument('--reverse_context', action='store_true')

    parser.add_argument('--use_aug_inst', action='store_true')
    parser.add_argument('--use_neg_aug_inst', action='store_true')
    parser.add_argument('--add_neg_prompt', action='store_true')
    parser.add_argument('--que_len', default=128, type=int)
    parser.add_argument('--max_inst_used', default=10, type=int)

    parser.add_argument('--use_inst_train', action='store_true')
    parser.add_argument('--max_inst', type=int, default=5)
    parser.add_argument('--use_cross_inst_prompt', action='store_true')
    
    parser.add_argument('--inst_datasets', nargs='+', default=['coco','lvis'])
    parser.add_argument('--sem_datasets', nargs='+', default=['coco','ade20k'])
    parser.add_argument('--eval_sem_datasets', nargs='+', default=['ade'])
    parser.add_argument('--force_input_size', default=None, type=int)
    parser.add_argument('--use_task_indicator', action='store_true')
    parser.add_argument('--inst_for_simm', default=0, type=int)
    parser.add_argument('--tau_simm', default=1, type=float)
    parser.add_argument('--dataset_ratio', default=None, type=float, nargs='+')
    parser.add_argument('--samples_per_epoch', default=160000, type=int)

    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()


def main(args):

    misc.init_distributed_mode(args)
    assert args.input_size[0] == args.input_size[1]
    model = build_model(args)
    os.makedirs(args.output, exist_ok=True)

    logger = get_logger(name='mmdet', log_file=os.path.join(args.output, 'log.txt'), file_mode='a')
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    if torch.cuda.is_available():
        model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    custom_sam_without_ddp = model.module

    ### --- Step 1: Train or Valid dataset ---
    args.eval = args.eval or args.eval_vos or args.eval_semseg or args.custom_eval

    if not args.eval:
        logger.info("--- create training dataloader ---")
        aug_list = [RandomHFlip(), LargeScaleJitter(output_size=args.input_size[0])]
        if args.use_dual_aug:
            aug_list = [DualAug(aug_list+[DefaultBundle()])]

        train_dataset_sem = DatasetCOCO('data', None, transforms.Compose(aug_list), 'trn', 1, False)
        train_dataset = train_dataset_sem
        if args.use_inst_train :
            dataset_list = []
            for name in args.sem_datasets :
                print('sem name', name)
                if name == 'coco':
                    dataset_list.append(train_dataset_sem)
                elif name.startswith('coco'):
                    name, fold = name.split('_')
                    assert name == 'coco'
                    fold = int(fold)
                    dataset_list.append(DatasetCOCO('data', fold, transforms.Compose(aug_list), 'trn', 1, False))
                elif name == 'fss':
                    dataset_list.append(DatasetFSS('data', None, transforms.Compose(aug_list), 'trn', 1, False))
                elif name == 'ade20k':
                    dataset_list.append(SemADE('data/ade20k', transforms.Compose(aug_list)))
                elif name == 'empty':
                    continue
                else :
                    raise NotImplementedError

            aug_inst = get_inst_aug(args.input_size[0])
            aug_inst = DualAug([aug_inst])
            for name in args.inst_datasets :
                print('inst name', name)
                if name == 'o365':
                    dataset_list.append(InstCOCO('data/object365', aug_inst, dataset_name='o365'))
                elif name == 'coco':
                    dataset_list.append(InstCOCO('data/coco', aug_inst, dataset_name='coco'))
                elif name == 'lvis':
                    dataset_list.append(InstCOCO('data/lvis', aug_inst, dataset_name='lvis'))
                elif name == 'paco_lvis':
                    dataset_list.append(InstCOCO('data/paco', aug_inst, dataset_name='paco_lvis'))
                elif name == 'lip':
                    dataset_list.append(InstCOCO('data/lip', aug_inst, dataset_name='lip'))
                elif name == 'empty':
                    continue
                else :
                    raise NotImplementedError
            dataset_ratio = args.dataset_ratio
            if dataset_ratio is not None :
                assert len(dataset_ratio) == len(dataset_list)
            train_dataset  = CustomConcatDataset(dataset_list, dataset_ratio, samples_per_epoch=args.samples_per_epoch)
        sampler = DistributedSampler(train_dataset, shuffle=True)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, args.batch_size_train, drop_last=True)
        train_dataloaders = DataLoader(train_dataset, batch_sampler=batch_sampler_train, num_workers=4,
                                        collate_fn=custom_collate_fn)

        logger.info("{} train dataloaders created".format(len(train_dataloaders)))

    if args.use_dual_aug :
        aug_list_eval = [DualAug(
                            [Resize(args.input_size)], 
                            [RandomHFlip(), LargeScaleJitter(output_size=args.input_size[0])]
                            )]
    else :
        aug_list_eval = [Resize(args.input_size)]



    logger.info("--- create valid dataloader ---")
    if args.eval_vos :
        from utils.vos_dataset import YouTubeVOSTestDataset, DAVISTestDataset
        aug_list_eval = [ResizeVOS(args.input_size)]
        aug_eval = transforms.Compose(aug_list_eval)
        if args.vos_dataset == 'davis17':
            valid_datasets = [DAVISTestDataset('data/DAVIS/2017', '2017/val.txt', aug_eval)]
            args.vos_gt_root='data/DAVIS/2017/Annotations/480p/'
        elif args.vos_dataset == 'youtube' :
            valid_datasets = [YouTubeVOSTestDataset('data/ytbvos18', 'val', aug_eval)]
            args.vos_gt_root=None
        elif args.vos_dataset == 'showcase' :
            valid_datasets = [YouTubeVOSTestDataset('data/showcase', '', aug_eval)]
            args.vos_gt_root=None
        else :
            raise NotImplementedError
        valid_dataloaders = []
        for valid_dataset in valid_datasets :
            sampler = DistributedSampler(valid_dataset, shuffle=False)
            valid_dataloaders.append(DataLoader(valid_dataset, args.batch_size_valid, sampler=sampler, drop_last=False, num_workers=4))
    elif args.eval_semseg :
        valid_dataloaders = []
        aug_list_eval = [Resize(args.input_size)]
        aug_eval = transforms.Compose(aug_list_eval)
        
        # for data in args.parser.add_argument('--sem_datasets', nargs='+', default=['coco','ade20k'])
        valid_datasets = []
        valid_datasets_meta_loader = []
        for name in args.eval_sem_datasets:
            if name == 'ade20k':
                dataset = SemADE('data/ade20k', aug_eval, False, is_semseg=True)
                # dataset_meta = SemADE('data/ade20k', aug_eval, True, is_semseg=True, is_meta=True)
                dataset_meta = SemADE('sd_gen/ade20k_val', aug_eval, False, is_semseg=True, dataset_name='sd_ade20k', ext='tif', is_meta=True)
            elif name == 'coco':
                dataset = SemCOCO('data/coco', aug_eval, False, is_semseg=True)
                dataset_meta = SemCOCO('data/coco', aug_eval, is_train=True, custom_json_path='annotations/instances_train2017_fss.json', is_semseg=False, is_meta=True)
            elif name == 'ade847':
                dataset = SemADE('data/openseg_data/ADE20K_2021_17_01', aug_eval, False, is_semseg=True, dataset_name='ade847', ext='tif')
                dataset_meta = SemADE('data/openseg_data/ADE20K_2021_17_01', aug_eval, False, is_semseg=True, dataset_name='ade847', ext='tif', is_meta=True)
                dataset_meta_diff = SemADE('sd_gen/ade_val', aug_eval, False, is_semseg=True, dataset_name='sd_ade847', ext='tif', is_meta=True)
            elif name == 'pc459':
                dataset = SemADE('data/openseg_data/VOCdevkit/VOC2010', aug_eval, False, is_semseg=True, dataset_name='pc459', ext='tif')
                dataset_meta = SemADE('data/openseg_data/VOCdevkit/VOC2010', aug_eval, False, is_semseg=True, dataset_name='pc459', ext='tif', is_meta=True)
                dataset_meta_diff = SemADE('sd_gen/pc459_val', aug_eval, False, is_semseg=True, dataset_name='pc459', ext='tif', is_meta=True)
            else:
                raise NotImplementedError
            valid_datasets.append(dataset)
            sampler = DistributedSampler(dataset_meta, shuffle=False)
            valid_datasets_meta_loader.append(DataLoader(dataset_meta, 8, sampler=sampler, drop_last=False, num_workers=4,
                                                    collate_fn=custom_collate_fn))

        for valid_dataset in valid_datasets :
            sampler = DistributedSampler(valid_dataset, shuffle=False)
            valid_dataloaders.append(DataLoader(valid_dataset, args.batch_size_valid, sampler=sampler, drop_last=False, num_workers=4,
                                                collate_fn=custom_collate_fn))

    else :
        valid_dataloaders = []
        aug_list_eval = [Resize(args.input_size)]
        aug_eval = transforms.Compose(aug_list_eval)
        valid_datasets = []
        
        if args.eval_datasets == 'lvis':
            valid_datasets = [SemCOCO('data/lvis', aug_eval, False, is_semseg=True, is_lvis=True, fold=idx) for idx in range(10)]
        # TODO: custom datasets, coco format
        elif args.eval_datasets == 'coco':
            valid_datasets = [SemCOCO('data/coco', aug_eval, False, is_semseg=True, is_lvis=False)]
        elif args.eval_datasets == 'fss':
            valid_datasets = [(DatasetFSS('data', None, aug_eval, 'test', 1, False))]
        elif args.eval_datasets is None :
            valid_datasets = [DatasetCOCO('data', idx, aug_eval, 'test', 1, False) for idx in range(4)]
            valid_datasets.append(DatasetFSS('data', None, aug_eval, 'test', 1, False))
        for valid_dataset in valid_datasets :
            sampler = DistributedSampler(valid_dataset, shuffle=False)
            valid_dataloaders.append(DataLoader(valid_dataset, args.batch_size_valid, sampler=sampler, drop_last=False, num_workers=4,
                                                collate_fn=custom_collate_fn))

    logger.info("{} valid dataloaders created".format(len(valid_dataloaders)))
    
    if not args.restore_model and (args.eval_vos or args.eval_semseg or args.auto_resume):
        output_dir = args.output
        ckpt_list = [x for x in os.listdir(output_dir) if x.startswith('epoch_')]
        ckpt_list = sorted(ckpt_list, key=lambda x: int(x.split('.')[0].split('_')[-1]), reverse=True)
        if len(ckpt_list) :
            args.restore_model = os.path.join(output_dir, ckpt_list[0])
            if args.auto_resume:
                args.start_epoch = int(ckpt_list[0].split('.')[0].split('_')[-1]) + 1

    if args.restore_model:
        logger.info("restore model from: {}".format(args.restore_model))
        if torch.cuda.is_available():
            _info = model.module.load_state_dict(torch.load(args.restore_model), strict=False)
            print(_info)
        else:
            model.module.load_state_dict(torch.load(args.restore_model,map_location="cpu"))

    if not args.eval:
        logger.info("--- define optimizer ---")
        # optimizer = optim.Adam(custom_sam_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        optimizer = optim.AdamW(custom_sam_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        with torch.autograd.set_detect_anomaly(True):
            train(args, model, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:
        # sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)

        if args.eval_vos:
            evaluate_vos(args, model, valid_dataloaders, args.visualize)
        elif args.eval_semseg:
            evaluate_semseg(args, model, valid_dataloaders, valid_datasets_meta_loader)
        elif args.custom_eval:
            reference_list, target_list = None, None
            # refernece examples with gt masks
            reference_list = [('visual_prompt_examples/example_prompt1.jpg','visual_prompt_examples/example_mask1.png'),
                              ('visual_prompt_examples/example_prompt2.jpg', 'visual_prompt_examples/example_mask2.png')]
            # target images, (gt mask optional)
            target_list = [('visual_prompt_examples/test1_1.jpg', None), 
                              ('visual_prompt_examples/test1_2.jpg', None),
                              ('visual_prompt_examples/test2_1.jpg', None),
                              ('visual_prompt_examples/test2_2.jpg', None)]
            vis_root = 'vis_custom_eval'
            custom_eval(args, model, reference_list, target_list, vis_root=vis_root)
        else :
            evaluate(args, model, valid_dataloaders, args.visualize)

def train(args, model, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    logger = get_logger(name='mmdet', log_file=os.path.join(args.output, 'log.txt'), file_mode='a')

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    model.train()

    for epoch in range(epoch_start,epoch_num): 
        logger.info("epoch: {},  learning rate: {}".format(epoch, optimizer.param_groups[0]["lr"]))
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

        len_loader = len(train_dataloaders)
        for data in metric_logger.log_every(train_dataloaders,20,logger=logger):
            masks_hq, bbox_preds, loss, loss_dict = model(data)
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            # for k, v in model.named_parameters():
            #     if v.requires_grad and v.grad is None :
            #         print(k)
            # import pdb; pdb.set_trace()
            optimizer.step()
            if args.use_cos_sched :
                lr = adjust_learning_rate(optimizer, iter / len_loader + epoch, args)

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)


        logger.info("Finished epoch: {}".format(epoch))
        metric_logger.synchronize_between_processes()
        logger.info("Averaged stats: {}".format(metric_logger))
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            logger.info('come here save at {}'.format(args.output + model_name))
            # misc.save_on_master(net.module.state_dict(), args.output + model_name)
            misc.save_on_master(model.module.state_dict(), args.output + model_name)
        lr_scheduler.step()
        # test_stats = evaluate(args, net, sam, valid_dataloaders)
        test_stats = evaluate(args, model, valid_dataloaders, args.visualize)
        train_stats.update(test_stats)
        
        model.train()

    # Finish training
    logger.info("Training Reaches The Maximum Epoch Number")
    
def compute_iou(preds, target, return_inter_union=False):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds

    if return_inter_union :
        inter_all, union_all = 0, 0
        for i in range(0,len(preds)):
            inter, union = misc.mask_iter_union(postprocess_preds[i],target[i])
            inter_all += inter
            union_all += union
        return inter_all, union_all
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)

def evaluate_vos(args, model, valid_dataloaders, visualize=False):

    logger = get_logger(name='mmdet', log_file=os.path.join(args.output, 'log.txt'), file_mode='a')
    model.eval()
    logger.info("Validating...")
    # eval_root = 'test/yvos18'
    eval_root = os.path.join(args.output, 'test_{}'.format(args.vos_dataset))
    from vos_benchmark.benchmark import benchmark

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        for data_val in metric_logger.log_every(valid_dataloader,1, logger=logger):
            mask_preds = model(data_val, inference=True, forward_vos=True)
            vid = data_val['vid'][0]
            video_root = os.path.join(eval_root, vid)
            if not os.path.exists(video_root):
                os.makedirs(video_root)
            
            palette = Image.open(data_val['mask_path'][0]).getpalette()
            frame_ids = list(zip(*data_val['frame_ids']))[0]
            assert len(frame_ids) == len(mask_preds), '{} {}'.format(len(frame_ids), len(mask_preds))
            for i, (out_mask, fid) in enumerate(zip(mask_preds, frame_ids)) :
                out_img = Image.fromarray(out_mask.cpu().numpy().astype(np.uint8))
                if palette is not None :
                    out_img.putpalette(palette)
                out_img.save(os.path.join(video_root, fid[:-4]+'.png'))
                # data_val.keys()
    
    torch.cuda.synchronize()
    time.sleep(1) # force synch (PIL.Image)?
    if misc.is_main_process():
        if args.vos_gt_root is not None :
            global_jf, global_j, global_f, *_ = benchmark([args.vos_gt_root], [eval_root], False, 16, verbose=True, skip_first_and_last=True)
            global_jf, global_j, global_f = global_jf[0], global_j[0], global_f[0]
            logger.info(f'Global score: J&F: {global_jf:.1f} J: {global_j:.1f} F: {global_f:.1f}')

def evaluate(args, model, valid_dataloaders, visualize=False):

    logger = get_logger(name='mmdet', log_file=os.path.join(args.output, 'log.txt'), file_mode='a')
    model.eval()
    logger.info("Validating...")
    test_stats = {}

    from utils.meter import AverageMeter

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        logger.info('valid_dataloader len: {}'.format(len(valid_dataloader)))
        eval_meter = AverageMeter(valid_dataloader.dataset.class_ids, logger)

        # count = 0
        for data_val in metric_logger.log_every(valid_dataloader,50, logger=logger):
            masks_hq, bbox_preds, loss, loss_dict = model(data_val, inference=True)
            labels_ori = data_val['ori_label']
            if isinstance(labels_ori, (list, tuple)):
                labels_ori = torch.cat(labels_ori)[:, None]
            if torch.cuda.is_available():
                labels_ori = labels_ori.cuda()
            iou = compute_iou(masks_hq,labels_ori)
            boundary_iou = compute_boundary_iou(masks_hq,labels_ori)

            inter, union = compute_iou(masks_hq,labels_ori, True)
            eval_meter.update(inter.cuda(), union.cuda(), data_val['class_id'][0].cuda(), loss=None)

            if visualize:
                os.makedirs(args.output, exist_ok=True)
                masks_hq_vis = (F.interpolate(masks_hq.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
                imgs_aug = inputs_dual.permute(0, 2, 3, 1).cpu().numpy()
                labels_aug = (F.interpolate(labels_dual.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()

                labels_box_aug = misc.masks_to_boxes(labels_dual[:,0,:,:])
                # labels_box_aug = misc.masks_to_boxes(labels_aug[:,0,:,:])
                for ii in range(len(imgs)):
                    base = data_val['imidx'][ii].item()
                    logger.info('base:'.format(base))
                    save_base = os.path.join(args.output, str(k)+'_'+ str(base))
                    imgs_ii = imgs[ii].astype(dtype=np.uint8)
                    imgs_ii_aug = imgs_aug[ii].astype(dtype=np.uint8)
                    show_iou = torch.tensor([iou.item()])
                    show_boundary_iou = torch.tensor([boundary_iou.item()])
                    show_anns(masks_hq_vis[ii], None, bbox_preds_xyxy_show[ii].cpu(), None, save_base , imgs_ii, show_iou, show_boundary_iou)
                    show_anns(labels_aug[ii], None, labels_box_aug[ii].cpu(), None, save_base+'_aug' , imgs_ii_aug, show_iou, show_boundary_iou)
                    if 1 :
                        aa = cv2.imread('{}_0.png'.format(save_base))
                        bb = cv2.imread('{}_aug_0.png'.format(save_base))
                        cv2.imwrite('{}_0_combine.png'.format(save_base), np.concatenate([aa,bb], axis=1))
                       

            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)

        eval_meter.write_result()
        logger.info('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logger.info("Averaged stats: {}".format(metric_logger))
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)


    return test_stats

def evaluate_semseg(args, model, valid_dataloaders, valid_datasets_meta_loader):
    logger = get_logger(name='mmdet', log_file=os.path.join(args.output, 'log.txt'), file_mode='a')
    model.eval()
    logger.info("Validating...")
    test_stats = {}

    from utils.meter import AverageMeter

    with torch.no_grad():
        for k in range(len(valid_dataloaders)):
            metric_logger = misc.MetricLogger(delimiter="  ")
            valid_dataloader = valid_dataloaders[k]

            from tqdm import tqdm
            meta_data_dict = {}
            prompt_feats_list, inst_feats_list, cls_ids_list = [], [], []
            for data_val_meta in tqdm(valid_datasets_meta_loader[k]):
                prompt_feats, inst_feats = model.module.forward_cls_extraction(data_val_meta, 16)
                cls_ids = data_val_meta['class_id']
                for i, cls_id in enumerate(cls_ids.tolist()):
                    meta_data_dict[cls_id] = prompt_feats[i], inst_feats[i], data_val_meta['imidx'][i][None]
                prompt_feats_list.append(prompt_feats)
                inst_feats_list.append(inst_feats)
                cls_ids_list.append(cls_ids)
            
            prompt_feats_list = torch.cat(prompt_feats_list)
            inst_feats_list = torch.cat(inst_feats_list)
            cls_ids_list = torch.cat(cls_ids_list)
            
            # result = misc.all_gather(([prompt_feats_list, inst_feats_list, cls_ids_list], data_val_meta['class_id']))
            meta_data_dict_gather = misc.all_gather(meta_data_dict)
            meta_data = {k:v for dict_this in meta_data_dict_gather for k,v in dict_this.items()}
            meta_data = [(k,v) for k,v in meta_data.items()]
            meta_data = sorted(meta_data, key=lambda x:x[0])
            meta_data = [x[1] for x in meta_data]
            aa, bb, cc = list(zip(*meta_data))
            aa = torch.cat([x.cuda() for x in aa])
            bb = torch.cat([x.cuda() for x in bb])
            imidx_meta = torch.cat([x.cuda() for x in cc])
            data_meta = aa, bb

            logger.info('valid_dataloader len: {}'.format(len(valid_dataloader)))
            class_ids = torch.tensor(valid_dataloader.dataset.get_class_ids())
            eval_meter = AverageMeter(valid_dataloader.dataset.class_ids, logger)

            for data_val in tqdm(metric_logger.log_every(valid_dataloader, 50, logger=logger), total=len(valid_dataloader)):
                # data_val['class_name'] = class_names
                labels_ori = data_val['origin_semmask'].squeeze(0)[:, None] #(c, h, w)
                labels_ori = labels_ori * 255 # convert to 0-255 mask

                # valid_cids = labels_ori.flatten(1).sum(-1) > 0
                valid_cids = data_val['valid_cids'].squeeze(0)
                # valid_cids = labels_ori.flatten(1).sum(-1) >= 0 # use all cats
                if not valid_cids.sum():
                    continue
                data_meta_this = [x[valid_cids].clone() for x in data_meta]
                imidx_this = imidx_meta[valid_cids]
                if (imidx_this==data_val['imidx'].to(imidx_this)).sum():
                    print('skip', imidx_this==data_val['imidx'].to(imidx_this))
                for x in data_meta_this:
                    x[imidx_this==data_val['imidx'].to(imidx_this)] = 0

                # labels_ori = labels_ori[valid_cids]
                selected_class_ids = class_ids[valid_cids]

                masks_hq, masks_score, *_ = model(data_val, inference=True, semseg_meta=data_meta_this)

                masks_pred = F.interpolate(masks_hq, labels_ori.shape[-2:], mode='bilinear') > 0
                if isinstance(labels_ori, (list, tuple)):
                    labels_ori = torch.cat(labels_ori)[:, None]
                if torch.cuda.is_available():
                    labels_ori = labels_ori.cuda()

                iou = compute_iou(masks_hq,labels_ori)
                boundary_iou = compute_boundary_iou(masks_hq,labels_ori)

                for mask_this, labels_this, cls_this in zip(masks_hq,labels_ori, selected_class_ids):
                    inter, union = compute_iou(mask_this[None], labels_this[None], True)
                    # if labels_this[None].sum():
                    #     eval_meter.update(inter.cuda(), union.cuda(), cls_this.cuda(), loss=None)
                    eval_meter.update(inter.cuda(), union.cuda(), cls_this.cuda(), loss=None)

        eval_meter.write_result()
        logger.info('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logger.info("Averaged stats: {}".format(metric_logger))
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)

    torch.cuda.synchronize()

def custom_eval(args, model, reference_list, target_list, vis_root='vis_custom_eval'):
    
    logger = get_logger(name='mmdet', log_file=os.path.join(args.output, 'log.txt'), file_mode='a')
    model.eval()
    logger.info("Validating...")
    test_stats = {}

    
    data_processor = EvalDataProcessor(args.input_size)
    from tqdm import tqdm
    prompt_feats_list, inst_feats_list, cls_ids_list = [], [], []
    # extract prompts on reference examples
    for data_val_meta in tqdm(reference_list):
        data_val_meta = data_processor(*data_val_meta)
        if args.custom_eval_with_instance:
            data_val_meta['is_inst'] = [True] * len(data_val_meta['image'])
        prompt_feats, inst_feats = model.module.forward_cls_extraction(data_val_meta, 16)
        prompt_feats_list.append(prompt_feats)
        inst_feats_list.append(inst_feats)

    prompt_feats_list = torch.cat(prompt_feats_list)
    inst_feats_list = torch.cat(inst_feats_list)
    result = misc.all_gather([prompt_feats_list, inst_feats_list])
    prompt_feats_list, inst_feats_list = list(zip(*result))
    prompt_feats_list = torch.cat([x.cuda() for x in prompt_feats_list])
    inst_feats_list = torch.cat([x.cuda() for x in inst_feats_list])
    data_meta = prompt_feats_list, inst_feats_list

    metric_logger = misc.MetricLogger(delimiter="  ")
    # mask decoding on target images
    for data_val in tqdm(metric_logger.log_every(target_list, 50, logger=logger), total=len(target_list)):
        data_val = data_processor(*data_val)
        if args.custom_eval_with_instance:
            data_val['is_inst'] = [True] * len(data_val_meta['image'])
        masks_hq, *_ = model(data_val, inference=True, semseg_meta=data_meta)
        # rule-based mask score
        masks_score_map = masks_hq.sigmoid()
        mask_score = (masks_score_map>0.5)*masks_score_map + (masks_score_map<0.5)*(1-masks_score_map)
        mask_score = mask_score.flatten(1).mean(1)

        labels_ori = data_val['ori_label']
        if isinstance(labels_ori, (list, tuple)):
            labels_ori = torch.cat(labels_ori)[:, None]
        if torch.cuda.is_available():
            labels_ori = labels_ori.cuda()
        masks_pred = F.interpolate(masks_hq, labels_ori.shape[-2:], mode='bilinear') > 0

        # for i in range(len(masks_pred)):
        imidx = data_val['imidx']
        if isinstance(imidx, torch.Tensor):
            imidx = imidx.item()
        if isinstance(imidx, Sequence):
            imidx = imidx[0]
        for i, masks_pred_i in enumerate(masks_pred[:,0]):
            os.makedirs(vis_root, exist_ok=True)
            import cv2
            masks_pred_i = masks_pred_i.cpu().int().numpy() * 255
            cv2.imwrite(os.path.join(vis_root, '{}_{}.png'.format(imidx, i)), masks_pred_i)

    logger.info('============================')

    torch.cuda.synchronize()

if __name__ == "__main__":

    args = get_args_parser()

    main(args)
