import torch
import torch.nn as nn
from typing import List, Tuple, Type
from .segment_anything_training.modeling.common import LayerNorm2d
import torch.nn.functional as F


import math
import copy
import utils.misc as misc
from utils.loss_mask import loss_masks
import torch.nn.functional as F
from torchvision.transforms import PILToTensor
from detectron2.layers import ROIAlign
import random
from .segment_anything_training.modeling import TwoWayTransformer, MaskDecoder, ThreeWayTransformer
from collections import OrderedDict

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from collections import defaultdict
import queue
import cv2
import numpy as np
from tqdm import tqdm

from .backbone import MODEL_CONFIG, build_encoder

class CustomMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for outptu
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class SEGIC(nn.Module):
    def __init__(self, 
                 encoder, 
                 neck,
                 prompt_encoder,
                 mask_decoder, 
                 args=None
                ):
        super().__init__()
        self.args = args
        self.encoder_model = args.encoder_model
        self.neck = neck
        self.encoder = encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.reverse_context = args.reverse_context
        self.use_cross_inst_prompt = args.use_cross_inst_prompt
        if self.use_cross_inst_prompt:
            self.inst_type_embed = nn.Embedding(2, 256)

        self.no_sim_prompt = args.no_sim_prompt
        self.no_text = args.no_text

        self.use_inst_proj = args.use_inst_proj
        self.diff_text_prompt_ratio = args.diff_text_prompt_ratio
        self.use_keypoint = args.use_keypoint
        self.num_keypoint = args.num_keypoint
        self.up_ft_index = 1
        self.feat_pad_size = 64 * 2 ** (self.up_ft_index-1)

        self.add_neg_prompt = args.add_neg_prompt
        self.use_aug_inst= args.use_aug_inst
        self.use_neg_aug_inst = args.use_neg_aug_inst
        if args.use_neg_aug_inst :
            self.neg_cls_embed = nn.Embedding(1, 256)
        self.aug_inst_que = dict()
        self.text_dict = dict()
        self.que_len = args.que_len
        self.max_inst_used = args.max_inst_used
        self.use_task_indicator = args.use_task_indicator
        self.inst_for_simm = args.inst_for_simm
        self.tau_simm = args.tau_simm

        model_name = args.encoder_model
        if model_name == 'dinov2':
            model_name = '{}_{}'.format(model_name, args.dinov2_model)
        model_config = MODEL_CONFIG[model_name]
        vision_dim, text_dim, self.img_size = model_config['vision_dim'], model_config['text_dim'], model_config['img_size']

        self.prompt_proj = nn.Linear(text_dim, 256)
        if self.use_inst_proj:
            self.inst_proj = nn.Linear(vision_dim, 256)
        if self.use_keypoint :
            self.keypoint_proj = nn.Linear(vision_dim, 256)

        self.up_sample_size = self.img_size // 4
        self.up_sample = nn.Upsample(size=self.up_sample_size, mode='bilinear', align_corners=False)

        pixel_mean, pixel_std = model_config['pixel_mean'], model_config['pixel_std']
        pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

        self.register_buffer('pixel_mean', pixel_mean)
        self.register_buffer('pixel_std', pixel_std)


    def _enque(self, cls_name_list, inst_feat_list, is_inst_list):

        assert len(cls_name_list) == len(inst_feat_list), '{} {}'.format(len(cls_name_list), len(inst_feat_list))
        for cls_name, inst_feat, is_inst in zip(cls_name_list, inst_feat_list, is_inst_list):
            if is_inst :
                continue
            if cls_name in self.aug_inst_que :
                que = self.aug_inst_que[cls_name]
            else :
                que = queue.Queue(self.que_len)
                self.aug_inst_que[cls_name] = que
            if que.qsize() == self.que_len:
                que.get()
            # push current results into queue
            inst_feat = copy.deepcopy(inst_feat.detach().clone())
            que.put(inst_feat)

    def _sample_inst(self, cls_name, n_inst=None):
        if cls_name not in self.aug_inst_que:
            return None
        inst_feat_list = list(self.aug_inst_que[cls_name].queue)
        if n_inst is not None :
            num_inst_used = n_inst
        else :
            num_inst_used = random.randint(0, self.max_inst_used)
        if not num_inst_used :
            return None

        if len(inst_feat_list) > num_inst_used:
            selected_idx = np.random.choice(
                list(range(len(inst_feat_list))), size=num_inst_used, replace=False
            ).tolist()
            inst_feat_list = [inst_feat_list[idx] for idx in selected_idx]
        return torch.stack(inst_feat_list, dim=1)

    def train(self, mode=True):
        super().train(mode)
        if self.args.encoder_model in ['clip_b', 'clip', 'dinov2', 'dinov1', 'mae', 'convnext', 'deit']:
            self.encoder.eval()
    
    @torch.no_grad()
    def extract_dift_feature(self, images, ori_sizes, cls_names=None, inference=False, is_inst_list=None):
        ft_list = []
        prompt_list = []
        prompt_input_list = []
        for i in range(len(images)):
            cls_name = cls_names[i] if cls_names is not None else None
            if inference and self.args.no_text_eval or self.no_text:
                prompt = ''
            elif random.random() < self.diff_text_prompt_ratio or inference:
                prompt = f'a photo of a {cls_name}' if cls_name else ''
            else :
                prompt = ''
            if is_inst_list is not None and is_inst_list[i]:
                prompt = 'please segment the instances'
            prompt_input_list.append(prompt)

        if self.encoder_model == 'dift':
            images = (images / 255.0 - 0.5) * 2
            # DIFT style padding
            for i, (img, ori_size) in enumerate(zip(images, ori_sizes)):
                h, w = ori_size
                prompt = prompt_input_list[i]
                img = img.clone()
                img[:,h:, :] = 0
                img[:,:, w:] = 0
                ft, prompt = self.encoder.forward(img,
                                prompt=prompt,
                                t=261,
                                up_ft_index=self.up_ft_index,
                                ensemble_size=1, # FIXME: how to set ensemble_size
                                ret_prompt=True
                                )
                prompt_list.append(prompt.mean(0))
                h, w = ft.shape[-2:]
                # HACK: abs size, to improve
                ft_pad = F.pad(ft, (0, self.feat_pad_size-w,0,self.feat_pad_size-h), value=0)
                ft_list.append(ft_pad)
            ft_list, prompt_list = torch.cat(ft_list), torch.stack(prompt_list)
        elif self.encoder_model == 'sam':
            images = ((images / 255.0 ) - self.pixel_mean) / self.pixel_std
            for i, (img, ori_size) in enumerate(zip(images, ori_sizes)):
                h, w = ori_size
                img[:,h:, :] = 0
                img[:,:, w:] = 0
            image_embeddings, _ = self.encoder(images)
            ft_list = image_embeddings
            prompt_list = self.encoder.get_prompt_features(prompt_input_list)

        elif self.encoder_model in ['clip', 'clip_b', 'clip_conv', 'pyramid_clip', 'declip']:
            # CLIP style padding
            images = ((images / 255.0 ) - self.pixel_mean) / self.pixel_std
            for i, (img, ori_size) in enumerate(zip(images, ori_sizes)):
                h, w = ori_size
                img[:,h:, :] = 0
                img[:,:, w:] = 0
            if self.encoder_model in ('clip', 'clip_b'):
                image_embeddings = self.encoder.vision_model(images, output_hidden_states=True).hidden_states[-1][:,1:]
                bs, l, c = image_embeddings.shape
                ft_list = image_embeddings.reshape(bs, int(math.sqrt(l)), int(math.sqrt(l)), c).permute(0,3,1,2).contiguous()
            else :
                ft_list = self.encoder(images)
            prompt_list = self.encoder.get_prompt_features(prompt_input_list)
        elif self.encoder_model in ['dinov2', 'dinov1', 'mae', 'convnext', 'deit']:
            images = ((images / 255.0 ) - self.pixel_mean) / self.pixel_std
            for i, (img, ori_size) in enumerate(zip(images, ori_sizes)):
                h, w = ori_size
                img[:,h:, :] = 0
                img[:,:, w:] = 0
            image_embeddings = self.encoder(images)
            ft_list = image_embeddings
            prompt_list = self.encoder.get_prompt_features(prompt_input_list)


        for prompt in prompt_list:
            if is_inst_list is not None and not is_inst_list[i]:
                cls_name = cls_names[i] if cls_names is not None else None
                name = prompt_input_list[i] if prompt_input_list[i] == '' else cls_name
                if name not in self.text_dict:
                    self.text_dict[name] = prompt

        return ft_list, prompt_list

    def state_dict(self,):
        state_dict = super().state_dict()
        new_state_dict = OrderedDict()

        def recursive_getattr(obj, k):
            if len(k.split('.')) > 1 :
                name_list = k.split('.')
                prefix, next = name_list[0], '.'.join(name_list[1:])
                return recursive_getattr(getattr(obj,prefix), next)
            return getattr(obj, k)

        for k,v in state_dict.items():
            if 'prompt_encoder.pe_layer' in k or recursive_getattr(self, k).requires_grad :
                new_state_dict[k] = v
        return new_state_dict

    def extract_inst_feat(self, image_embed, inst_mask):
        is_list = isinstance(inst_mask, list)
        if is_list :
            inst_mask = None
        # HACK
        h, w = image_embed.shape[-2:]
        inst_labels_64 = F.interpolate(inst_mask.float(), (h, w), mode='bilinear', align_corners=False).squeeze(1)
        inst_labels_64 = (inst_labels_64 > 0.5).float()
        inst_embedding = torch.einsum('nchw,nhw->nc', image_embed, inst_labels_64) / inst_labels_64.sum((-1,-2)).clamp(min=1)[:, None]
        return inst_embedding

    def forward(self, data, inference=False, forward_vos=False, semseg_meta=None):
        if forward_vos:
            return self.forward_vos(data, inference)
        if semseg_meta is not None :
            return self.forward_semseg(data, semseg_meta)
        
        args = self.args
        inputs, ori_size = data['image'].cuda(), data['ori_size']
        assert 'image_dual' in data

        inputs_dual = data['image_dual'].cuda()
        ori_size_dual = data['ori_size_dual']

        labels, labels_dual = data['label'], data['label_dual']

        offset_list = [len(x) for x in labels]
        offset_list_dual = [len(x) for x in labels_dual]
        assert offset_list == offset_list_dual

        labels = torch.cat(labels)[:, None].cuda()
        labels_dual = torch.cat(labels_dual)[:, None].cuda()
        labels = ((labels / 255) > 0.5).float()
        labels_dual = ((labels_dual / 255) > 0.5).float()

        cls_names = data.get('class_name')
        if cls_names is None :
            cls_names = [''] * len(inputs)

        is_inst_list = data.get('is_inst', None)
        is_box_mask_list = data.get('is_box_mask', None)
        bs = len(inputs)

        inputs_total = torch.cat([inputs, inputs_dual])
        ori_size_total = torch.cat([ori_size, ori_size_dual])
        image_embeddings_total, input_prompt_total = self.extract_dift_feature(
                        inputs_total, ori_size_total, cls_names=cls_names+cls_names, inference=inference,
                        is_inst_list=torch.cat([is_inst_list,is_inst_list]))

        # process attn
        image_embeddings, image_embeddings_dual = image_embeddings_total.split([bs, bs])

        # text prompt
        input_prompt_total = self.prompt_proj(input_prompt_total)[:, None]
        input_prompt, input_prompt_dual = input_prompt_total.split([bs, bs])
        
        neg_class_names = data.get('neg_class_names', None)
        meta_info = dict(is_inst_list=is_inst_list, cls_names=cls_names,
                         is_box_mask_list=is_box_mask_list,
                         neg_class_names=neg_class_names)

        if not inference and self.reverse_context:
            masks_pred, bbox_preds, loss, loss_dict = self.maks_decoding(inference, ori_size,
                                    image_embeddings, image_embeddings_dual, input_prompt,
                                    labels, labels_dual,
                                    offset_list,
                                    **meta_info
                                    )
            labels_pred = F.interpolate(masks_pred.detach(), labels.shape[-2:], mode='bilinear', align_corners=False) > 0
            _, _, loss_reverse, loss_dict_reverse = self.maks_decoding(inference, ori_size,
                                    image_embeddings_dual, image_embeddings, input_prompt_dual,
                                    labels_dual, labels_pred,
                                    offset_list,
                                    **meta_info
                                    )
            loss = (loss + loss_reverse) / 2
            for k, v in loss_dict_reverse.items():
                loss_dict.update(**{'{}_reverse'.format(k):v})

            return masks_pred, bbox_preds, loss, loss_dict

        return self.maks_decoding(inference, ori_size,
                                  image_embeddings, image_embeddings_dual, input_prompt,
                                  labels, labels_dual,
                                  offset_list,
                                  **meta_info
                                )
    
    def maks_decoding(self, inference, ori_size,
                      image_embeddings, image_embeddings_dual, input_prompt,
                      labels, labels_dual,
                      offset_list,
                      semseg_meta=None,
                      cls_names=None,
                      is_inst_list=None,
                      is_box_mask_list=None,
                      neg_class_names=None,
                      **kwargs
                      ):
        offset_list_dual = offset_list

        img_size = self.img_size
        args = self.args

        # image_embeddings_dual_upsample = self.up_sample(image_embeddings_dual)
        image_embeddings_upsample = self.up_sample(image_embeddings)

        if is_box_mask_list is not None :
            is_box_mask_list = torch.cat([x[None].expand(num_inst) for x, num_inst in zip(is_box_mask_list, offset_list)])

        if semseg_meta is None :
            # extract instance feature
            # image_embeddings_dual_inst = torch.cat([x[None].expand(num_inst, -1, -1, -1) for x, num_inst in zip(image_embeddings_dual, offset_list_dual)])
            labels_dual_split = labels_dual.split(offset_list)
            inst_feat_duals = [self.extract_inst_feat(x[None], label) for x, label in zip(image_embeddings_dual,labels_dual_split)]
            inst_feat_duals = torch.cat(inst_feat_duals)
        else :
            input_prompt, inst_feat_duals = semseg_meta
            input_prompt = input_prompt.squeeze(1)
            inst_feat_duals = inst_feat_duals.squeeze(1)

        # extract keypoint feature
        keypoint_feat_list = []
        if self.use_keypoint:
            divisor = torch.tensor(labels_dual.shape[-2:], device=labels_dual.device, dtype=labels_dual.dtype)
            keypoint_xy_list = [((x>0.5).nonzero()[:, 1:]+0.5)/ divisor * 2 - 1 for x in labels_dual] # renorm to (-1, 1)
            for i, (kp, img_embed) in enumerate(zip(keypoint_xy_list, image_embeddings_dual_inst)):
                if len(kp):
                    if len(kp) > self.num_keypoint :
                        rand_idx = torch.randperm(len(kp))[:self.num_keypoint]
                    else :
                        rand_idx = torch.linspace(0, len(kp)-1, self.num_keypoint).long()
                    kp = kp[rand_idx]
                    sampled_feat_this = F.grid_sample(img_embed[None].float(), kp.flip(dims=(1,))[None,None], mode='bilinear')[0,:,0] 
                    sampled_feat_this = sampled_feat_this.transpose(0,1).contiguous() # (L, c)
                else :
                    sampled_feat_this = img_embed.new_zeros(self.num_keypoint,img_embed.shape[0])
                keypoint_feat_list.append(sampled_feat_this)
            keypoint_feat_list = torch.stack(keypoint_feat_list)
            keypoint_feat_list = self.keypoint_proj(keypoint_feat_list)
            keypoint_feat_list = keypoint_feat_list.split(offset_list_dual)
        else :
            keypoint_feat_list = [None] * len(image_embeddings)

        if not inference and labels is not None :
            labels_box = misc.masks_to_boxes(labels[:,0,:,:])
            labels_box = labels_box.split(offset_list)

        def get_max(sim, target_hw=1024):
            h, w = sim.shape[-2:]
            sim = sim.flatten(-2)
            _, idx = sim.topk(args.n_point, dim=-1)
            idx_h = idx // w * target_hw / h
            idx_w = idx % w * target_hw / w
            idx = torch.cat([idx_w, idx_h], dim=-1)
            return idx

        
        masks_pred =[]
        ious_pred = []
        ious_target = []
        not_neg_aug_list = []
        if self.use_inst_proj:
            inst_feat_prompt = self.inst_proj(inst_feat_duals)[:, None] # (bs, 1, c)
            inst_feat_prompt = inst_feat_prompt.split(offset_list_dual)
        inst_feat_duals = inst_feat_duals.split(offset_list)
        # image_embeddings_upsample_inst = image_embeddings_upsample_inst.split(offset_list)

        image_embedding_decoder = self.neck(image_embeddings)
        labels_list = []
        if not inference:
            labels_split = labels.split(offset_list)
        for i, (img_feat, inst_feat_dual) in enumerate(zip(image_embeddings, inst_feat_duals)):

            labels_this = labels_split[i] if not inference else None
            inst_feat_dual_this = inst_feat_dual[:, None] #(ninst, 1, c)
            labels_boxes_in = labels_box[i] if random.random() > 0.5 and not inference else None

            # semantic augment

            add_neg_prompt = False
            if not inference and self.add_neg_prompt and not is_inst_list[i] and random.random() > 0.75:
                neg_cls_names_this = neg_class_names[i]
                
                if len(neg_cls_names_this) > 40:
                    selected_idx = np.random.choice(
                        list(range(len(neg_cls_names_this))), size=40, replace=False
                    ).tolist()

                    neg_cls_names_this = [neg_cls_names_this[x] for x in selected_idx]
                
                neg_inst_list = []
                neg_text_list = []
                random.shuffle(neg_cls_names_this)
                for neg_cls in neg_cls_names_this:
                    neg_inst = self._sample_inst(neg_cls, 1)
                    if neg_inst is not None :
                        neg_inst_list.append(neg_inst)
                        neg_text_prompt = self.text_dict.get(neg_cls, None)
                        if neg_text_prompt is None:
                            # neg_text_prompt = torch.zeros_like(input_prompt[i])
                            neg_text_prompt = input_prompt[i].new_zeros(self.prompt_proj.weight.shape[-1])
                        neg_text_list.append(neg_text_prompt)
                            
                        # self.text_dict[neg_cls] if neg_cls in self.text_dict else self.text_dict[''])

                if len(neg_inst_list):
                    add_neg_prompt = True
                    neg_inst_list = torch.cat(neg_inst_list)
                    neg_text_list = self.prompt_proj(torch.stack(neg_text_list))

                    neg_labels_this = torch.zeros_like(labels_this[:1,]).expand(len(neg_inst_list), -1, -1, -1)
                    not_neg_aug_this = [True] * len(labels_this) + [False] * len(neg_labels_this)
                    labels_this = torch.cat([labels_this, neg_labels_this])
                    inst_feat_dual_this = torch.cat([inst_feat_dual_this, neg_inst_list])
            
            if not add_neg_prompt and not inference:
                not_neg_aug_this = [True] * len(labels_this)

            if not inference and self.use_aug_inst and cls_names is not None and not is_inst_list[i] and random.random() > 0.5:
                aug_inst_feat = self._sample_inst(cls_names[i])
                if aug_inst_feat is not None :
                    inst_feat_dual_this = torch.cat([inst_feat_dual_this, aug_inst_feat], dim=1)
                    aug_inst_prompt = self.inst_proj(aug_inst_feat)
                else :
                    aug_inst_prompt = None

                # neg aug
                if self.use_neg_aug_inst and neg_class_names is not None :
                    neg_cls_names_this = neg_class_names[i]
                    neg_aug_inst_feat = [self._sample_inst(x, 1) for x in neg_cls_names_this]
                    neg_aug_inst_feat = [x for x in neg_aug_inst_feat if x is not None]
                    if len(neg_aug_inst_feat):
                        n_neg_inst = len(neg_aug_inst_feat)
                        neg_aug_inst_feat = torch.cat(neg_aug_inst_feat, dim=1)
                        neg_aug_inst_prompt = self.inst_proj(neg_aug_inst_feat)
                        neg_type_embed = self.neg_cls_embed.weight.expand(n_neg_inst, -1).clone()
                        neg_aug_inst_prompt = neg_aug_inst_prompt + neg_type_embed
                    else :
                        neg_aug_inst_prompt = None

                    if aug_inst_prompt is None :
                        aug_inst_prompt = neg_aug_inst_prompt
                    elif neg_aug_inst_prompt is not None:
                        aug_inst_prompt = torch.cat([aug_inst_prompt, neg_aug_inst_prompt], dim=1)

            else :
                aug_inst_feat = None
                aug_inst_prompt = None

            # FIXME
            dift_sim_this = misc.cal_sim(image_embeddings_upsample[i][None], inst_feat_dual_this, eps=1e-8)
            dift_sim_this = F.interpolate(dift_sim_this, size=img_size, mode="bilinear", align_corners=False)
            if self.inst_for_simm > 0:
                if dift_sim_this.shape[1] > self.inst_for_simm :
                    labels_masks_in, _ = dift_sim_this.topk(self.inst_for_simm, 1)
                    labels_masks_in = labels_masks_in.mean(1, keepdim=True)
                else :
                    labels_masks_in = dift_sim_this.mean(1, keepdim=True)
            else :
                labels_masks_in = dift_sim_this[:, :1] #(ninst, 1, h, w)

            labels_masks_in = labels_masks_in / self.tau_simm

            # XXX: naive topk point
            # FIXME: mean cross topk
            # use simm map of maskpooling feat 
            labels_points_in = get_max(labels_masks_in, target_hw=self.img_size)
            labels_points_in_type = torch.ones_like(labels_points_in[:,:,0])

            if semseg_meta is None :
                if add_neg_prompt :
                    text_prompt_pos = input_prompt[i][None]
                    text_prompt_neg = neg_text_list[:, None]
                    text_prompt = torch.cat([text_prompt_pos, text_prompt_neg])
                else :
                    text_prompt = input_prompt[i][None].expand(len(labels_masks_in), -1, -1)
            else :
                text_prompt = input_prompt[:, None]
            # inst_feat_prompt_cinst = None
            aug_cross_inst = self.use_cross_inst_prompt and (inference or random.random() > 0.5)
            # if self.use_cross_inst_prompt and (inference or random.random() > 0.5):
            # FIXME
            # if aug_cross_inst :
            #     labels_points_in, labels_points_in_type = self.process_cross_inst_prompt(labels_points_in)

            if self.use_inst_proj :
                inst_feat_prompt_this = inst_feat_prompt[i]
                if add_neg_prompt:
                    inst_feat_prompt_this = self.inst_proj(inst_feat_dual_this)
                if aug_inst_prompt is not None :
                    assert text_prompt.shape[0] == 1
                    # aug_inst_prompt = self.inst_proj(aug_inst_feat)
                    inst_feat_prompt_this = torch.cat([inst_feat_prompt_this, aug_inst_prompt], dim=1)

                if aug_cross_inst:
                    inst_feat_prompt_cinst, inst_feat_prompt_cinst_type = self.process_cross_inst_prompt(inst_feat_prompt_this)
                    inst_feat_prompt_cinst = inst_feat_prompt_cinst + self.inst_type_embed(inst_feat_prompt_cinst_type)
                    inst_feat_prompt_this = inst_feat_prompt_cinst

                text_prompt = torch.cat([text_prompt, inst_feat_prompt_this], dim=1)

            if self.use_keypoint and keypoint_feat_list[i] is not None:
                text_prompt = torch.cat([text_prompt, keypoint_feat_list[i]], dim=1)

            point_in = (labels_points_in, labels_points_in_type)
            if self.no_sim_prompt:
                labels_masks_in = None
                point_in = None
            # elif random.random() < 0.15:
            #     labels_masks_in = None
            #     point_in = None
                
            if add_neg_prompt :
                labels_boxes_in = None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=point_in,
                boxes=labels_boxes_in,
                masks=labels_masks_in,
                texts=text_prompt,
            )

            if not inference and self.add_neg_prompt:
                num_inst = 10
                sparse_embeddings = sparse_embeddings[:num_inst]
                dense_embeddings = dense_embeddings[:num_inst]
                labels_this = labels_this[:num_inst]
                not_neg_aug_this = not_neg_aug_this[:num_inst]
                not_neg_aug_list.extend(not_neg_aug_this)

            if self.use_task_indicator:
                task_indicator = is_inst_list[i].expand(len(sparse_embeddings))
            else :
                task_indicator = None

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding_decoder[i].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                task_indicator=task_indicator
            )

            masks_pred.append(low_res_masks)
            ious_pred.append(iou_predictions)
            labels_list.append(labels_this)
            
        masks_pred = torch.cat(masks_pred)
        ious_pred = torch.cat(ious_pred)
        if not inference:
            labels = torch.cat(labels_list)
        if inference:
            scale_factor = self.img_size / masks_pred.shape[-1]
            masks_pred = F.interpolate(masks_pred, scale_factor=scale_factor, mode="bilinear", align_corners=False)
            # assert len(masks_pred) == 1
            h,w = ori_size[0]
            masks_pred = masks_pred[:, :, :h, :w]
            return masks_pred, ious_pred, None, None

        # loss_mask, loss_dice = loss_masks(masks_pred, labels, len(masks_pred), is_box_mask=is_box_mask_list)
        mask_weight = None
        num_masks = sum(offset_list)
        if self.add_neg_prompt :
            mask_weight = masks_pred.new_ones(masks_pred.shape[0], 1) * 0.1
            mask_weight[not_neg_aug_list] = 1
            num_masks = mask_weight.sum()

        loss_mask, loss_dice = loss_masks(masks_pred, labels, num_masks, is_box_mask=is_box_mask_list, 
                                        mask_weight=mask_weight)

        bbox_preds = None
        loss = loss_mask + loss_dice
        loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice}

        if self.add_neg_prompt :
            fg_labels = torch.tensor(not_neg_aug_list, dtype=ious_pred.dtype, device=ious_pred.device)
            loss_score = F.binary_cross_entropy_with_logits(ious_pred, fg_labels[:, None])
            loss = loss + loss_score
            loss_dict.update(loss_score=loss_score)

        if (self.use_aug_inst or self.add_neg_prompt) and self.training and cls_names is not None :
            self._enque(cls_names, inst_feat_duals, is_inst_list)


        if not inference and not all(not_neg_aug_list): # filter the negative pred by neg aug
            masks_pred = masks_pred[not_neg_aug_list]
        return masks_pred, bbox_preds, loss, loss_dict

    def forward_semseg(self, data, semseg_meta):
        images, ori_sizes = data['image'].cuda(), data['ori_size'].cuda()
        image_feat, *_ = self.extract_dift_feature(images, ori_sizes, None, inference=True,
                                                     is_inst_list=[False]*len(images))

        input_prompts, inst_feats = semseg_meta
        offset_list = [len(input_prompts)]
        result_list = []
        # semseg_meta = input_prompts[:20], inst_feats[:20]
        # offset_list = [20]
        # cls_names = cls_names[:20]
        is_inst_list = data.get('is_inst', None)
        return self.maks_decoding(True, ori_sizes, image_feat, None,
                                  None, None, None, offset_list,
                                  semseg_meta, None, is_inst_list=is_inst_list)


    @torch.no_grad()
    def forward_vos(self, data, inference=False):
        assert inference == True
        args = self.args
        images, inst_masks, inst_ids = data['images'][0], data['inst_masks'][0], data['inst_ids'][0]
        inst_masks = inst_masks.unsqueeze(1)
        inst_masks = ((inst_masks / 255) > 0.5).float()
        ori_sizes = data['ori_size'].expand(len(images), -1)
        n_inst = len(inst_masks)
        

        # FIXME
        def get_max(sim, target_hw=1024):
            h, w = sim.shape[-2:]
            sim = sim.flatten(-2)
            _, idx = sim.topk(args.n_point, dim=-1)
            idx_h = idx // w * target_hw / h
            idx_w = idx % w * target_hw / w
            idx = torch.cat([idx_w, idx_h], dim=-1)
            return idx

        split_len = 2
        images_split = [images[i:i+split_len] for i in range(0, len(images), split_len)]
        ori_sizes_split = [ori_sizes[i:i+split_len] for i in range(0, len(images), split_len)]
        
        image_feats, input_prompts = [], []
        with torch.no_grad():
            for x, size in zip(images_split, ori_sizes_split):
                image_feat, input_prompt = self.extract_dift_feature(x, size, is_inst_list=[True]*len(x))
                image_feats.append(image_feat)
                input_prompts.append(input_prompt)
                torch.cuda.empty_cache()
        image_feats = torch.cat(image_feats)
        image_feats_decoder = self.neck(image_feats)
        input_prompts = torch.cat(input_prompts)

        input_prompts = self.prompt_proj(input_prompts)[:, None].expand(-1, n_inst, -1)

        inst_masks_pred = [inst_masks]
        inst_masks_pred_logits = [inst_masks]
        inst_masks_pred_score = [inst_masks.new_ones(inst_masks.shape[:2])]
        inst_feat_list = []
        # images_feats_upsample = self.up_sample(image_feats)
        images_feats_upsample = image_feats
        inst_feat_list.append(self.extract_inst_feat(images_feats_upsample[0][None].expand(n_inst,-1,-1,-1), inst_masks_pred[-1]))

        if self.use_keypoint:
            keypoint_feat_list = []
            divisor = torch.tensor(inst_masks.shape[-2:], device=inst_masks.device, dtype=inst_masks.dtype)
            keypoint_xy_list = [((x>0.5).nonzero()[:, 1:]+0.5)/ divisor * 2 - 1 for x in inst_masks] # renorm to (-1, 1)
            for i, (kp, img_embed) in enumerate(zip(keypoint_xy_list, images_feats_upsample[0][None].expand(n_inst,-1,-1,-1))):
                if len(kp):
                    if len(kp) > self.num_keypoint :
                        rand_idx = torch.randperm(len(kp))[:self.num_keypoint]
                    else :
                        rand_idx = torch.linspace(0, len(kp)-1, self.num_keypoint).long()
                    kp = kp[rand_idx]
                    sampled_feat_this = F.grid_sample(img_embed[None].float(), kp.flip(dims=(1,))[None,None], mode='bilinear')[0,:,0] 
                    sampled_feat_this = sampled_feat_this.transpose(0,1).contiguous() # (L, c)
                else :
                    sampled_feat_this = img_embed.new_zeros(self.num_keypoint,img_embed.shape[0])
                keypoint_feat_list.append(sampled_feat_this)
            keypoint_feat_list = torch.stack(keypoint_feat_list)
            keypoint_feat_list = self.keypoint_proj(keypoint_feat_list)

        sparse_embeddings_pre, dense_embeddings_pre = None, None
        patch_weight = 1.
        for i, (img_feat, input_prompt) in enumerate(zip(image_feats, input_prompts)):

            img_feat_upsample = self.up_sample(img_feat[None])
            # HACK: improve the assemble machnism

            inst_feat = torch.stack(inst_feat_list[:1]).mean(0)[:, None] #(1, x, c)
            if len(inst_feat_list) > 2:
                idx = torch.linspace(0, len(inst_feat_list)-1, 2).int().tolist()
                inst_feat_list_this = [inst_feat_list[x] for x in idx]
            else :
                inst_feat_list_this = inst_feat_list

            if self.use_inst_proj:
                inst_feat_prompt = self.inst_proj(torch.stack(inst_feat_list_this, dim=1))
            else :
                inst_feat_prompt = None

            # dift_sim = misc.cal_sim(img_feat_upsample, inst_feat) # (x, 1, h, w)
            dift_sim = misc.cal_sim(img_feat_upsample, torch.stack(inst_feat_list_this, dim=1)) # (x, 1, h, w)

            dift_sim = F.interpolate(dift_sim, size=images.shape[-2], mode="bilinear", align_corners=False)

            if self.inst_for_simm > 0:
                if dift_sim.shape[1] > self.inst_for_simm :
                    labels_masks_in, _ = dift_sim.topk(self.inst_for_simm, 1)
                    labels_masks_in = labels_masks_in.mean(1, keepdim=True)
                else :
                    labels_masks_in = dift_sim.mean(1, keepdim=True)
            else :
                labels_masks_in = dift_sim[:, :1] #(ninst, 1, h, w)

            labels_masks_in = labels_masks_in / self.tau_simm
            if self.use_aug_inst:
                labels_points_in = get_max(dift_sim, target_hw=self.img_size)
            else :
                labels_points_in = get_max(labels_masks_in, target_hw=self.img_size)
            labels_points_in_type = torch.ones_like(labels_points_in[:,:,0])

            inst_feat_prompt_cinst = None
            if self.use_cross_inst_prompt :
                # aa, bb = self.process_cross_inst_prompt(labels_points_in)
                labels_points_in, labels_points_in_type = self.process_cross_inst_prompt(labels_points_in)
                if self.use_inst_proj:
                    inst_feat_prompt_cinst, inst_feat_prompt_cinst_type = self.process_cross_inst_prompt(inst_feat_prompt)
                    inst_feat_prompt = inst_feat_prompt_cinst + self.inst_type_embed(inst_feat_prompt_cinst_type)

            text_prompt = input_prompt[:, None]
            if self.use_inst_proj :
                text_prompt = torch.cat([text_prompt, inst_feat_prompt], dim=1)
            if self.use_keypoint :
                text_prompt = torch.cat([text_prompt, keypoint_feat_list], dim=1)

            if self.no_sim_prompt:
                labels_masks_in = None
                point_in = None

            sparse_embeddings_this, dense_embeddings_this = self.prompt_encoder(
                points=(labels_points_in, labels_points_in_type),
                boxes=None,
                masks=labels_masks_in,
                texts=text_prompt,
            )
            
            # frame1
            if not i :
                continue
            if sparse_embeddings_pre is None :
                sparse_embeddings, dense_embeddings = sparse_embeddings_this, dense_embeddings_this
            else :
                sparse_embeddings = sparse_embeddings_this * patch_weight + sparse_embeddings_pre * (1- patch_weight)
                dense_embeddings = dense_embeddings_this * patch_weight + dense_embeddings_pre * (1- patch_weight)
                sparse_embeddings_pre, dense_embeddings_pre = sparse_embeddings, dense_embeddings

            mask_pred = []
            iou_pred = []
            for ii in range(n_inst):
                if self.use_task_indicator:
                    task_indicator = torch.tensor([True]).expand(len(sparse_embeddings[ii:ii+1]))
                else :
                    task_indicator = None
                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=image_feats_decoder[i].unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings[ii:ii+1],
                    dense_prompt_embeddings=dense_embeddings[ii:ii+1],
                    multimask_output=False,
                    task_indicator=task_indicator
                )
                mask_pred.append(low_res_masks)
                iou_pred.append(iou_predictions)
            mask_pred = torch.cat(mask_pred)
            iou_pred = torch.cat(iou_pred)
            scale_factor = self.img_size / mask_pred.shape[-1]
            mask_pred = F.interpolate(mask_pred, scale_factor=scale_factor, mode="bilinear", align_corners=False)
            
            inst_masks_pred.append((mask_pred>0).float())
            inst_masks_pred_logits.append(mask_pred)
            inst_masks_pred_score.append(iou_pred)
            inst_feat_list.append(self.extract_inst_feat(images_feats_upsample[i][None].expand(n_inst,-1,-1,-1), inst_masks_pred[-1]))
            torch.cuda.empty_cache()
            
        inst_masks_pred = torch.stack(inst_masks_pred).squeeze(2) #(nframe, ninst, h, w)
        inst_masks_pred_logits = torch.stack(inst_masks_pred_logits).squeeze(2)
        inst_masks_pred_score = torch.stack(inst_masks_pred_score)

        h, w = ori_sizes[0]
        inst_masks_pred = inst_masks_pred[:, :, :h, :w]
        inst_masks_pred_logits = inst_masks_pred_logits[:, :, :h, :w]

        pre_resize_size = data['ori_inst_masks'].shape[-2:]
        inst_masks_pred_logits = F.interpolate(inst_masks_pred_logits, pre_resize_size, mode='bilinear', align_corners=False)
        inst_masks_pred = inst_masks_pred_logits > 0

        seg_result = inst_masks_pred

        seg_result = seg_result * inst_masks_pred_logits.sigmoid()

        seg_result = (seg_result.max(1)[1] + 1) * inst_masks_pred.max(1)[0]

        return seg_result.int()

    @torch.no_grad()
    def forward_cls_extraction(self, data, split_len=8):
        images, ori_sizes = data['image'].cuda(), data['ori_size'].cuda()
        if isinstance(data['label'], list):
            labels = torch.cat(data['label']).cuda() / 255
        else :
            labels= data['label'].cuda() / 255
        labels = labels[:, None]
        cls_names = data.get('class_name', None)
        if cls_names is None:
            cls_names = ['' for _ in range(len(images))]

        is_inst_list = data.get('is_inst', [False]*len(images))
        images_split = [images[i:i+split_len] for i in range(0, len(images), split_len)]
        ori_sizes_split = [ori_sizes[i:i+split_len] for i in range(0, len(images), split_len)]
        cls_names_split = [cls_names[i:i+split_len] for i in range(0, len(images), split_len)]
        labels_split = [labels[i:i+split_len] for i in range(0, len(images), split_len)]
        is_inst_list = [is_inst_list[i:i+split_len] for i in range(0, len(images), split_len)]

        image_feats, input_prompts = [], []
        inst_feats = []
        with torch.no_grad():
            for i, (x, size, cls_names_this, is_inst_this) in enumerate(zip(images_split, ori_sizes_split, cls_names_split, is_inst_list)):
                image_feat, input_prompt = self.extract_dift_feature(x, size, cls_names_this, inference=True,
                                                                        is_inst_list=is_inst_this)
                input_prompt = self.prompt_proj(input_prompt)
                image_feats.append(image_feat)
                input_prompts.append(input_prompt)
                torch.cuda.empty_cache()

                inst_feat_this = self.extract_inst_feat(image_feat, labels_split[i])
                inst_feats.append(inst_feat_this)
        input_prompts = torch.cat(input_prompts)[:, None]
        inst_feats = torch.cat(inst_feats)[:, None]
        return input_prompts, inst_feats


    def process_cross_inst_prompt(self, prompt):
        '''
        inst_prompt (n_inst, n_p, c)
        '''
        n_inst, n_p = prompt.shape[:2]
        prompt_cinst = prompt.flatten(0,1)[None].expand(n_inst, -1, -1)
        prompt_type_cinst = prompt_cinst.new_zeros(n_inst, n_inst, n_p, dtype=torch.long)
        prompt_type_cinst[range(n_inst), range(n_inst)] = 1
        prompt_type_cinst = prompt_type_cinst.flatten(1,2)

        return prompt_cinst, prompt_type_cinst


def build_model(args):

    encoder, prompt_encoder, neck = build_encoder(args)
    mask_decoder_transformer = TwoWayTransformer
    num_keypoint = args.num_keypoint

    prompt_embed_dim = 256
    model = SEGIC(
        encoder=encoder,
        neck = neck,
        prompt_encoder=prompt_encoder,
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=mask_decoder_transformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        args = args
    )

    return model