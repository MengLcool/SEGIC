import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel, ConvNextModel
from transformers.models.vit_mae.modeling_vit_mae import  ViTMAEEmbeddings, ViTMAEEncoder, ViTMAEPatchEmbeddings, ViTMAEModel
from copy import deepcopy
from .src_dino.hubconf import dino_vitb16

class CustomDINOv2(nn.Module):
    def __init__(self, dinov2_model) -> None:
        super().__init__()
        assert dinov2_model in ('b', 'l', 'g')
        # import pdb; pdb.set_trace()
        if dinov2_model == 'b':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        elif dinov2_model == 'l':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        elif dinov2_model == 'g':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        elif dinov2_model == 'b_reg':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        elif dinov2_model == 'l_reg':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        elif dinov2_model == 'g_reg':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')

        else :
            raise NotImplementedError

        self.text_model = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
        self.tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')

    def forward(self, x):
        h, w = [xx// 14 for xx in x.shape[-2:]]
        output = self.dinov2.forward_features(x)
        output = output['x_norm_patchtokens'] #(bs,l,c)
        bs, l, c = output.shape
        assert h*w == l
        return output.view(bs, h, w, c).permute(0,3,1,2).contiguous() #(bs, h, w ,c)

    def get_prompt_features(self, prompt):
        device = next(self.parameters()).device
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
    

class CustomDINOv1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.dinov1 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.dinov1 = dino_vitb16()

        self.text_model = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
        self.tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')

    def forward_features(self, x, masks=None):

        x = self.dinov1.prepare_tokens(x)

        for blk in self.dinov1.blocks:
            x = blk(x)

        x_norm = self.dinov1.norm(x)
        return x_norm[:, 1:]

    def forward(self, x):
        output = self.forward_features(x) #(bs,l,c)
        bs, l, c = output.shape
        h = w = int(math.sqrt(l))
        assert h*w == l
        return output.view(bs, h, w, c).permute(0,3,1,2).contiguous() #(bs, h, w ,c)

    def get_prompt_features(self, prompt):
        device = next(self.parameters()).device
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
    
    

class CustomEncoder(nn.Module):
    def __init__(self, img_encoder) -> None:
        super().__init__()
        self.img_encoder = img_encoder
        self.text_model = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
        self.tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')

    def forward(self, x):
        return self.img_encoder(x)

    def get_prompt_features(self, prompt):
        device = next(self.parameters()).device
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

class CustomViTMAEPatchEmbeddings(ViTMAEPatchEmbeddings):
    def forward(self, pixel_values):
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x        

class CustomViTMAEEmbeddings(ViTMAEEmbeddings):
    def __init__(self, config):
        nn.Module.__init__(self)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = CustomViTMAEPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.config = config
        self.initialize_weights()

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.position_embeddings.shape[1] - 1
        if npatch == N and w == h:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = x.shape[-1]
        # patch_size = self.patch_embeddings.patch_size
        w0 = w // self.patch_embeddings.patch_size[0]
        h0 = h // self.patch_embeddings.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, pixel_values, noise=None):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        
        # add position embeddings w/o cls token
        # import pdb; pdb.set_trace()
        # aa = torch.rand(1, 1025, 768)
        B, nc, h, w = pixel_values.shape
        position_embeddings = self.interpolate_pos_encoding(embeddings, w, h)
        embeddings = embeddings + position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore

class CustomMAEEncoder(ViTMAEModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = CustomViTMAEEmbeddings(config)
        self.encoder = ViTMAEEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()


class CustomMAE(nn.Module):
    def __init__(self, config='facebook/vit-mae-base') -> None:
        super().__init__()
        self.img_encoder = CustomMAEEncoder.from_pretrained(config)
        self.text_model = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
        self.tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')

    def forward(self, x):
        image_embeddings = self.img_encoder(x).last_hidden_state[:,1:]
        bs, l, c = image_embeddings.shape
        ft_list = image_embeddings.reshape(bs, int(math.sqrt(l)), int(math.sqrt(l)), c).permute(0,3,1,2).contiguous()
        return ft_list

    def get_prompt_features(self, prompt):
        device = next(self.parameters()).device
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

class CustomConvNext(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.model, *_ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg')
        # self.tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg')
        self.model = ConvNextModel.from_pretrained("facebook/convnext-base-224")
        self.text_model = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
        self.tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')

        del self.model.layernorm
        self.model.layernorm = nn.Identity()

        if True :
            stages = deepcopy(self.model.encoder.stages)
            del self.model.encoder.stages
            self.model.encoder.stages = stages[:3]
        self.requires_grad_(False)

    def forward(self, x):
        return self.model(x).last_hidden_state
    
    @property
    def device(self):
        return next(self.parameters()).device

    def get_prompt_features(self, prompt):
        device = next(self.parameters()).device
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