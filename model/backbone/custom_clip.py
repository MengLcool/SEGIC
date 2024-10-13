import torch
import torch.nn as nn
import math
import open_clip
from copy import deepcopy
from transformers import CLIPVisionModel
from transformers.models.clip.modeling_clip import (CLIPPreTrainedModel, CLIPVisionConfig, CLIPVisionTransformer, CLIPVisionEmbeddings, CLIPEncoder, 
                    CLIPConfig, CLIPTextConfig, CLIPTextModel)
from transformers import AutoTokenizer

class CustomCLIPVisionEmbeddings(CLIPVisionEmbeddings):
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        
        h, w = pixel_values.shape[-2:]
        pos_embeddings = self.position_embedding(self.position_ids)
        pos_embeddings = self.interpolate_pos_encoding(embeddings, w, h)
        # pos_embeddings = self.interpolate_pos_encoding(embeddings, pos_embeddings[0], w, h)

        embeddings = embeddings + pos_embeddings
        return embeddings

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.position_embedding.weight.shape[0] - 1
        if npatch == N and w == h:
            return self.position_embedding.weight
        pos_embed = self.position_embedding.weight.float()
        class_pos_embed = pos_embed[:1, :]
        patch_pos_embed = pos_embed[1:, :]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

class CustomCLIPVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config: CLIPVisionConfig):
        super(CLIPVisionTransformer, self).__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CustomCLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

class CustomCLIPVisionModel(CLIPVisionModel):
    def __init__(self, config: CLIPVisionConfig):
        super(CLIPVisionModel, self).__init__(config)
        self.vision_model = CustomCLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()


class CLIPModel(CLIPPreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextModel(text_config)
        self.vision_model = CustomCLIPVisionModel(vision_config)
        self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)

    def get_prompt_features(self, prompt):
        device = next(self.text_model.parameters()).device
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        for k, x in text_inputs.items():
            if hasattr(x, 'to'): text_inputs[k] = text_inputs[k].to(device)
        
        return self.text_model(**text_inputs).pooler_output
    
    

class CLIPModelConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.model, *_ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg')
        # self.tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg')
        self.model, *_ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg')
        del self.model.visual.trunk.head
        self.model.visual.trunk.head = nn.Identity()
        self.requires_grad_(False)
        if True :
            stages = deepcopy(self.model.visual.trunk.stages)
            del self.model.visual.trunk.stages
            self.model.visual.trunk.stages = stages[:3]

    def forward(self, x):
        return self.model.visual.trunk(x)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def get_prompt_features(self, prompt):
        text = self.tokenizer(prompt).to(self.device)
        return self.model.encode_text(text)


class PyramidCLIPModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from models.models import build_model
        from models.simple_tokenizer import tokenize

        self.tokenizer = tokenize
        model = build_model('RN50')
        ckpt_path = 'pretrained_checkpoint/PyramidCLIP-YFCC15MV2-RN50.pth' # specify path of checkpoint
        if ckpt_path:
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
        self.model = model

        self.requires_grad_(False)

    def forward(self, x):
        return self.model.encode_image(x, extract_dense_feature=True)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def get_prompt_features(self, prompt):
        text = self.tokenizer(prompt).to(self.device)
        return self.model.encode_text(text)


class DeCLIPModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from prototype.model import model_entry
        from prototype.utils.misc import parse_config
        from models.simple_tokenizer import tokenize

        config_file = 'DeCLIP/experiments/declip_experiments/declip88m/declip88m_r50_declip/config.yaml'
        config = parse_config(config_file)
        model = model_entry(config.model)

        import torch
        from collections import OrderedDict
        ckpt = torch.load('DeCLIP/r50.pth.tar', map_location='cpu')
        new_ckpt = OrderedDict()
        model_dict = model.state_dict()
        for k,v in ckpt['model'].items():
            k = k.replace('module.', '')
            if k in model_dict and model_dict[k].shape == v.shape:
                new_ckpt[k]= v
            else:
                print(k)

        model.load_state_dict(new_ckpt, strict=False)
        self.tokenizer = tokenize
        self.model = model

        self.requires_grad_(False)

    def forward(self, x):
        return self.model.encode_image(x, return_dense=True)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def get_prompt_features(self, prompt):
        return self.model.encode_text(prompt)

