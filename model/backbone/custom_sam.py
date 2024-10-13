import math
import torch
import torch.nn as nn
import copy
import utils.misc as misc
from utils.loss_mask import loss_masks
import torch.nn.functional as F
from detectron2.layers import ROIAlign
import random
from segment_anything_training.modeling.transformer import TwoWayAttentionBlock

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes

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

class CorrespondenceEncoder(nn.Module):
    def __init__(self, 
                    depth,
                    num_heads: int,
                    mlp_dim: int,
                    keypoint_size=7,
                    embed_dim=256,
                    activation = nn.ReLU,
                    attention_downsample_rate: int = 2,
                 ):
        super().__init__()
        self.keypoint_size = keypoint_size
        self.keypoint_embedding = nn.Embedding(keypoint_size*keypoint_size, embed_dim)
        self.fgbg_embedding = nn.Embedding(2, embed_dim)
        self.layers = nn.ModuleList()
        
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.mlp_text_prompt = MLP(embed_dim, mlp_dim, embed_dim, 3)
        

    def pool_mask_ref(self, mask_ref):
        h = w = int(math.sqrt(mask_ref.shape[1]))
        assert h*w == mask_ref.shape[1]
        mask_ref = mask_ref.view(mask_ref.shape[0], 1, h, w)
        mask_ref_resize = F.adaptive_avg_pool2d(mask_ref, self.keypoint_size).flatten(1)
        return mask_ref_resize


    def forward(self, img_ref_feat, mask_ref):
        '''
        img_ref_feat: (bs, l, c)
        mask_ref: (bs, l)
        '''
        queries = self.keypoint_embedding.weight
        queries = queries[None].expand(img_ref_feat.shape[0], -1, -1)
        mask_ref = self.pool_mask_ref(mask_ref)
        mask_ref_embed = self.keypoint_embedding(mask_ref)
        queries = queries + mask_ref_embed
        keys = img_ref_feat
        for i, layer in enumerate(self.layers):
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=0,
                key_pe=0,
            )
        
        sparse_embedding = self.mlp_text_prompt(queries)
        return queries, sparse_embedding


class CustomSam(nn.Module):
    def __init__(self, sam, net, 
                 keypoint_size = 7,
                 args=None
                ):
        super().__init__()
        self.args = args
        self.sam = sam
        self.net = net

        # self.keypoint_embedding = nn.Embedding(keypoint_size*keypoint_size, 256)
    
    def forward(self, data, inference=False):
        args = self.args
        sam, net = self.sam, self.net
        inputs, labels = data['image'], data['label']
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            if 'image_dual' in data:
                inputs_dual = data['image_dual'].cuda()
                labels_dual = data['label_dual'].cuda()
                def custom_roi(input, bbox, resolution=1024):
                    bbox = bbox.int()
                    return F.interpolate(input[:, bbox[1]:bbox[3], bbox[0]:bbox[2]][:, None], resolution)[:, 0]
                if 1 :
                    pooler = ROIPooler((1024,1024), [1], 0, "ROIAlignV2")
                    inst_bbox  = misc.masks_to_boxes(labels_dual[:,0,:,:])
                    inst_bbox = [Boxes(x) for x in inst_bbox[:, None]]
                    inputs_dual = pooler([inputs_dual], inst_bbox)
                    labels_dual = pooler([labels_dual], inst_bbox)
                    # inputs_dual = torch.stack([custom_roi(input, bbox)] for input, bbox in zip(inputs_dual, inst_bbox))
                    # labels_dual = torch.stack([custom_roi(input, bbox)] for input, bbox in zip(labels_dual, inst_bbox))
                    # import pdb; pdb.set_trace()
            else:
                inputs_dual, labels_dual = None, None

        bs = len(inputs)
        
        if not inference :
            input_keys = copy.deepcopy(args.input_keys)
        else :
            input_keys = copy.deepcopy(args.eval_keys)
        labels_box = misc.masks_to_boxes(labels[:,0,:,:])
        try:
            labels_points = misc.masks_sample_points(labels[:,0,:,:])
        except:
            # less than 10 points
            # input_keys = ['box','noise_mask']
            if 'point' in input_keys:
                input_keys.remove('point')
        labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
        labels_noisemask = misc.masks_noise(labels_256)

        batched_input = [{'image':x} for x in inputs]
        if inputs_dual is not None :
            batched_input.extend([{'image':x} for x in inputs_dual])

        with torch.no_grad():
            image_embed_output = sam(batched_input, only_forward_img=True)
            inst_image_embeddings = image_embeddings = image_embed_output[0]

        if True :
            image_embeddings = image_embed_output[0]
            inst_image_embeddings = image_embeddings
            if inputs_dual is not None :
                image_embeddings = image_embeddings[:bs]
                inst_image_embeddings = image_embeddings[-bs:]
                def proccess_image_embed_output(image_embed_output):
                    a, b = image_embed_output
                    a = a[:bs]
                    b = [x[:bs] for x in b]
                    return a, b
                image_embed_output = proccess_image_embed_output(image_embed_output)

            inst_label = labels_dual if labels_dual is not None else labels
            if args.noised_inst :
                inst_label = misc.masks_noise(inst_label, apply_incoherent=True)
            inst_labels_64 = F.interpolate(inst_label, size=(64, 64), mode='bilinear') / 255
            inst_embedding = torch.einsum('nchw,nhw->nc', inst_image_embeddings, inst_labels_64.squeeze(1)) / inst_labels_64.sum((-1,2)).clamp(min=1)
            try:
                labels_points_inst = misc.masks_sample_points(inst_label[:,0,:,:])
            except :
                labels_points_inst = torch.zeros((bs, 0, 2), device=inputs.device)

            if args.use_ref_keypoint :
                pooler = ROIAlign(32, 1/16, 0)
                pooler_label = ROIAlign(32, 1, 0)
                inst_bbox  = misc.masks_to_boxes(inst_label[:,0,:,:])
                bid_bbox = torch.tensor(range(len(inst_bbox)), dtype=inst_bbox.dtype, device=inst_bbox.device)[:, None]
                inst_bbox_roi = torch.cat([bid_bbox, inst_bbox], dim=-1)
                inst_roi_features = pooler(inst_image_embeddings, inst_bbox_roi)
                inst_roi_mask = pooler_label(inst_label/255, inst_bbox_roi)
                inst_roi_masked_features = inst_roi_features * inst_roi_mask
            else :
                inst_roi_masked_features = None

            labels_pionts_labels_inst = torch.ones(labels_points_inst.shape[:2], device=labels_points_inst.device)
            point_embeddings_inst = sam.prompt_encoder._embed_points(labels_points_inst, labels_pionts_labels_inst, pad=True)

        if args.use_ref_keypoint :
            sim = misc.cal_sim(image_embeddings, inst_roi_masked_features)
        else :
            sim = misc.cal_sim(image_embeddings, inst_embedding).unsqueeze(1)

        sim = F.interpolate(sim, size=(256, 256), mode='bilinear')

        batched_input = [{'image':x, 'original_size':x.shape[-2:]} for x in inputs]
        for b_i in range(bs):
            dict_input = batched_input[b_i]
            input_type = random.choice(input_keys)
            if input_type == 'box':
                dict_input['boxes'] = labels_box[b_i:b_i+1]
            elif input_type == 'point':
                point_coords = labels_points[b_i:b_i+1]
                dict_input['point_coords'] = point_coords
                dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
            elif input_type == 'sem_corr':
                # dict_input['mask_inputs'] = sim[b_i:b_i+1]
                def get_max(sim, target_hw=1024):
                    h, w = sim.shape[-2:]
                    sim = sim.squeeze(1).flatten(1)
                    _, idx = sim.topk(args.n_point, dim=1)
                    idx_h = idx // w * target_hw / h
                    idx_w = idx % w * target_hw / w
                    idx = torch.stack([idx_w, idx_h], dim=-1)
                    return idx
                point_coords = get_max(sim[b_i:b_i+1])
                dict_input['point_coords'] = point_coords
                dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
            elif input_type == 'noise_mask':
                dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
            else:
                raise NotImplementedError

        batched_output, interm_embeddings = sam(batched_input, multimask_output=False,
                                                    image_embed_output=image_embed_output)
        
        batch_len = len(batched_output)
        encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
        image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
        sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
        dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

        masks_hq, bbox_preds = net(
            image_embeddings=encoder_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            hq_token_only=True,
            interm_embeddings=interm_embeddings,
            simm_input = sim,
            image_embedding_ref = inst_image_embeddings, 
            point_embedding_ref = point_embeddings_inst,
            inst_roi_masked_features=inst_roi_masked_features
        )

        if inference and bbox_preds is not None:
            point_coords = torch.cat([x['point_coords'] for x in batched_input])
            point_labels = torch.cat([x['point_labels'] for x in batched_input])
            bbox_preds_xyxy = misc.box_cxcywh_to_xyxy(bbox_preds) * 1024
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=bbox_preds_xyxy,
                masks=None
            )
            masks_hq, bbox_preds = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings.unsqueeze(1),
                dense_prompt_embeddings=dense_embeddings.unsqueeze(1),
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
                simm_input = sim,
                image_embedding_ref = inst_image_embeddings, 
                point_embedding_ref = point_embeddings_inst,

            )

        loss_mask, loss_dice = loss_masks(masks_hq, labels/255.0, len(masks_hq))
        loss = loss_mask + loss_dice
        loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice}
        if args.use_bbox_head :
            labels_box_xywh = misc.box_xyxy_to_cxcywh(labels_box) / 1024
            bbox_preds = bbox_preds
            num_boxes = bs

            loss_bbox = F.l1_loss(bbox_preds, labels_box_xywh, reduction='none')
            loss_giou = 1 - torch.diag(misc.generalized_box_iou(
                misc.box_cxcywh_to_xyxy(bbox_preds),
                misc.box_cxcywh_to_xyxy(labels_box_xywh)))
            
            loss_dict['loss_giou'] = loss_giou = loss_giou.sum() / num_boxes
            loss_dict['loss_bbox'] = loss_bbox = loss_bbox.sum() / num_boxes
            loss = loss + loss_bbox*5 + loss_giou*2

        return masks_hq, bbox_preds, loss, loss_dict