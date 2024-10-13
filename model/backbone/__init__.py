import torch
import torch.nn as nn
from typing import List, Tuple, Type
from .src_dift.models.dift_sd import SDFeaturizer
from .custom_clip import CustomCLIPVisionModel, CLIPModel, CLIPModelConv, DeCLIPModel
from .custom_dino import CustomMAE, CustomConvNext, CustomDINOv1, CustomEncoder, CustomDINOv2
from .custom_deit import CustomDeiT
from .custom_clip import PyramidCLIPModel
from ..segment_anything_training import sam_model_registry
from ..segment_anything_training.modeling.common import LayerNorm2d
from ..segment_anything_training.modeling.prompt_encoder import PromptEncoder, PositionEmbeddingRandom

MODEL_CONFIG={}

MODEL_CONFIG['dift']=dict(
    vision_dim = 1280,
    text_dim = 1024,
    img_size = 1024,
    pixel_mean = (0.48145466, 0.4578275, 0.40821073),
    pixel_std= (0.26862954, 0.26130258, 0.27577711),
)
MODEL_CONFIG['clip']=dict(
    vision_dim = 1024,
    text_dim = 768,
    img_size = 896,
    pixel_mean = (0.48145466, 0.4578275, 0.40821073),
    pixel_std= (0.26862954, 0.26130258, 0.27577711)
)
MODEL_CONFIG['clip_b']=dict(
    vision_dim = 768,
    text_dim = 512,
    img_size = 32 * 32,
    pixel_mean = (0.48145466, 0.4578275, 0.40821073),
    pixel_std= (0.26862954, 0.26130258, 0.27577711)
)
MODEL_CONFIG['mae']=dict(
    vision_dim = 768,
    text_dim = 768,
    img_size = 32 * 14,
    pixel_mean = (0.485, 0.456, 0.406),
    pixel_std= (0.229, 0.224, 0.225)

)
MODEL_CONFIG['convnext']=dict(
    vision_dim = 512,
    text_dim = 768,
    img_size = 64 * 16,
    pixel_mean = (0.485, 0.456, 0.406),
    pixel_std= (0.229, 0.224, 0.225)

)
MODEL_CONFIG['deit']=dict(
    vision_dim = 768,
    text_dim = 768,
    img_size = 64 * 16,
    pixel_mean = (0.485, 0.456, 0.406),
    pixel_std= (0.229, 0.224, 0.225)

)
MODEL_CONFIG['clip_conv']=dict(
    vision_dim = 512,
    text_dim = 640,
    img_size = 1024,
    pixel_mean = (0.48145466, 0.4578275, 0.40821073),
    pixel_std= (0.26862954, 0.26130258, 0.27577711)
)
MODEL_CONFIG['pyramid_clip'] = MODEL_CONFIG['declip']=dict(
    vision_dim = 1024,
    text_dim = 1024,
    img_size = 1024,
    pixel_mean = (0.48145466, 0.4578275, 0.40821073),
    pixel_std= (0.26862954, 0.26130258, 0.27577711)
)
MODEL_CONFIG['dinov2_b']=dict(
    vision_dim = 768,
    text_dim = 768,
    img_size = 14*64,
    pixel_mean = (0.485, 0.456, 0.406),
    pixel_std= (0.229, 0.224, 0.225)
)
MODEL_CONFIG['dinov2_l']=dict(
    vision_dim = 1024,
    text_dim = 768,
    img_size = 14*64,
    pixel_mean = (0.485, 0.456, 0.406),
    pixel_std= (0.229, 0.224, 0.225)
)
MODEL_CONFIG['dinov2_g']=dict(
    vision_dim = 1536,
    text_dim = 768,
    img_size = 14*64,
    pixel_mean = (0.485, 0.456, 0.406),
    pixel_std= (0.229, 0.224, 0.225)
)
MODEL_CONFIG['dinov1']=dict(
    vision_dim = 768,
    text_dim = 768,
    img_size = 1024,
    pixel_mean = (0.485, 0.456, 0.406),
    pixel_std= (0.229, 0.224, 0.225)

)
MODEL_CONFIG['sam']=dict(
    vision_dim = 768,
    text_dim = 768,
    img_size = 1024,
    pixel_mean = (0.485, 0.456, 0.406),
    pixel_std= (0.229, 0.224, 0.225)

)


class CustomPromptEncoder(PromptEncoder):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        num_keypoint = 1,
        mask_down_stride=[2,2]
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super(PromptEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(num_keypoint, mask_in_chans // 4, kernel_size=mask_down_stride[0], stride=mask_down_stride[0]),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=mask_down_stride[1], stride=mask_down_stride[1]),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

def build_encoder(args):
    prompt_embed_dim = 256
    if args.encoder_model == 'dift':
        encoder = SDFeaturizer('stabilityai/stable-diffusion-2-1')
        embed_dim = 1280
        image_size = 1024
        downscale_stride = 16
        image_embedding_size = image_size // downscale_stride
        prompt_encoder = CustomPromptEncoder(
                    embed_dim=prompt_embed_dim,
                    image_embedding_size=(image_embedding_size, image_embedding_size),
                    input_image_size=(image_size, image_size),
                    mask_in_chans=16,
                    # num_keypoint= 1 if not args.use_keypoint else 2 * num_keypoint + 1,
                    mask_down_stride=[8,2]
                )
    elif args.encoder_model == 'clip' :
        embed_dim = 1024
        image_size = 14* 64
        downscale_stride = 14
        image_embedding_size = image_size // downscale_stride
        args.input_size = [image_size, image_size]
        prompt_encoder = CustomPromptEncoder(
                    embed_dim=prompt_embed_dim,
                    image_embedding_size=(image_embedding_size, image_embedding_size),
                    input_image_size=(image_size, image_size),
                    mask_in_chans=16,
                    # num_keypoint= 1 if not args.use_keypoint else 2 * num_keypoint + 1,
                    mask_down_stride=[7,2]
                )

        encoder = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
        for p in encoder.parameters():
            p.requires_grad = False
    elif args.encoder_model == 'clip_b' :
        embed_dim = 768
        image_size = 32*32
        downscale_stride = 32
        image_embedding_size = image_size // downscale_stride
        args.input_size = [image_size, image_size]
        prompt_encoder = CustomPromptEncoder(
                    embed_dim=prompt_embed_dim,
                    image_embedding_size=(image_embedding_size, image_embedding_size),
                    input_image_size=(image_size, image_size),
                    mask_in_chans=16,
                    # num_keypoint= 1 if not args.use_keypoint else 2 * num_keypoint + 1,
                    mask_down_stride=[8,4]
                )

        encoder = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        for p in encoder.parameters():
            p.requires_grad = False
    elif args.encoder_model == 'mae' :
        embed_dim = 768
        # image_size = 32*32
        image_size = 32 * 14
        downscale_stride = 32
        image_embedding_size = image_size // downscale_stride
        args.input_size = [image_size, image_size]
        prompt_encoder = CustomPromptEncoder(
                    embed_dim=prompt_embed_dim,
                    image_embedding_size=(image_embedding_size, image_embedding_size),
                    input_image_size=(image_size, image_size),
                    mask_in_chans=16,
                    # num_keypoint= 1 if not args.use_keypoint else 2 * num_keypoint + 1,
                    mask_down_stride=[8,4]
                )

        encoder = CustomMAE()
        for p in encoder.parameters():
            p.requires_grad = False
    elif args.encoder_model == 'convnext' :
        embed_dim = 512
        # image_size = 32*32
        image_size = 64 * 16
        downscale_stride = 16
        image_embedding_size = image_size // downscale_stride
        args.input_size = [image_size, image_size]
        prompt_encoder = CustomPromptEncoder(
                    embed_dim=prompt_embed_dim,
                    image_embedding_size=(image_embedding_size, image_embedding_size),
                    input_image_size=(image_size, image_size),
                    mask_in_chans=16,
                    # num_keypoint= 1 if not args.use_keypoint else 2 * num_keypoint + 1,
                    mask_down_stride=[8,2]
                )

        encoder = CustomConvNext()
        for p in encoder.parameters():
            p.requires_grad = False
    elif args.encoder_model == 'deit' :
        embed_dim = 768
        # image_size = 32*32
        image_size = 64 * 16
        downscale_stride = 16
        image_embedding_size = image_size // downscale_stride
        args.input_size = [image_size, image_size]
        prompt_encoder = CustomPromptEncoder(
                    embed_dim=prompt_embed_dim,
                    image_embedding_size=(image_embedding_size, image_embedding_size),
                    input_image_size=(image_size, image_size),
                    mask_in_chans=16,
                    mask_down_stride=[8,2]
                )

        encoder = CustomDeiT()
        for p in encoder.parameters():
            p.requires_grad = False
    elif args.encoder_model == 'clip_conv' :
        embed_dim = 512
        image_size = 64 * 16
        downscale_stride = 16
        
        image_embedding_size = image_size // downscale_stride
        args.input_size = [image_size, image_size]
        prompt_encoder = CustomPromptEncoder(
                    embed_dim=prompt_embed_dim,
                    image_embedding_size=(image_embedding_size, image_embedding_size),
                    input_image_size=(image_size, image_size),
                    mask_in_chans=16,
                    mask_down_stride=[8,2]
                )

        encoder = CLIPModelConv()
        for p in encoder.parameters():
            p.requires_grad = False

    elif args.encoder_model == 'pyramid_clip' :
        embed_dim = 1024
        image_size = 64 * 16
        downscale_stride = 16
        
        image_embedding_size = image_size // downscale_stride
        args.input_size = [image_size, image_size]
        prompt_encoder = CustomPromptEncoder(
                    embed_dim=prompt_embed_dim,
                    image_embedding_size=(image_embedding_size, image_embedding_size),
                    input_image_size=(image_size, image_size),
                    mask_in_chans=16,
                    mask_down_stride=[8,2]
                )

        encoder = PyramidCLIPModel()
        for p in encoder.parameters():
            p.requires_grad = False
    elif args.encoder_model == 'declip' :
        embed_dim = 1024
        image_size = 64 * 16
        downscale_stride = 16
        
        image_embedding_size = image_size // downscale_stride
        args.input_size = [image_size, image_size]
        prompt_encoder = CustomPromptEncoder(
                    embed_dim=prompt_embed_dim,
                    image_embedding_size=(image_embedding_size, image_embedding_size),
                    input_image_size=(image_size, image_size),
                    mask_in_chans=16,
                    mask_down_stride=[8,2]
                )

        encoder = DeCLIPModel()
        for p in encoder.parameters():
            p.requires_grad = False
    elif args.encoder_model =='dinov2':
        vision_dim_dict = {
            'b': 768,
            'l': 1024,
            'g': 1536,
            'b_reg': 768,
            'l_reg': 1024,
            'g_reg': 1536,
        }
        embed_dim = vision_dim_dict[args.dinov2_model]
        image_size = 14*64 if args.force_input_size is None else args.force_input_size
        downscale_stride = 14
        assert image_size % 14 == 0
        image_embedding_size = image_size // downscale_stride
        args.input_size = [image_size, image_size]
        prompt_encoder = CustomPromptEncoder(
                    embed_dim=prompt_embed_dim,
                    image_embedding_size=(image_embedding_size, image_embedding_size),
                    input_image_size=(image_size, image_size),
                    mask_in_chans=16,
                    # num_keypoint= 1 if not args.use_keypoint else 2 * num_keypoint + 1,
                    mask_down_stride=[7,2]
                )

        encoder = CustomDINOv2(args.dinov2_model)
        encoder.vision_dim = embed_dim
        encoder.img_size = image_size
        for p in encoder.parameters():
            p.requires_grad = False

    elif args.encoder_model =='sam':
        sam = sam_model_registry['vit_b'](checkpoint=args.checkpoint)
        encoder = sam.image_encoder
        del encoder.neck
        encoder.neck = nn.Identity()
        encoder = CustomEncoder(encoder)
        embed_dim = 768
        image_size = 1024
        downscale_stride = 16
        image_embedding_size = image_size // downscale_stride
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.vision_dim = embed_dim
        encoder.img_size = image_size

        prompt_encoder = CustomPromptEncoder(
                    embed_dim=prompt_embed_dim,
                    image_embedding_size=(image_embedding_size, image_embedding_size),
                    input_image_size=(image_size, image_size),
                    mask_in_chans=16,
                    # num_keypoint= 1 if not args.use_keypoint else 2 * num_keypoint + 1,
                    mask_down_stride=[8,2]
                )

    elif args.encoder_model == 'dinov1' :
        embed_dim = 768
        image_size = 16* 64
        downscale_stride = 16
        image_embedding_size = image_size // downscale_stride
        args.input_size = [image_size, image_size]
        prompt_encoder = CustomPromptEncoder(
                    embed_dim=prompt_embed_dim,
                    image_embedding_size=(image_embedding_size, image_embedding_size),
                    input_image_size=(image_size, image_size),
                    mask_in_chans=16,
                    # num_keypoint= 1 if not args.use_keypoint else 2 * num_keypoint + 1,
                    mask_down_stride=[8,2]
                )

        encoder = CustomDINOv1()
        for p in encoder.parameters():
            p.requires_grad = False
    else :
        raise NotImplementedError
    
    neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                prompt_embed_dim,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(prompt_embed_dim),
            nn.Conv2d(
                prompt_embed_dim,
                prompt_embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(prompt_embed_dim),
        )

    return encoder, prompt_encoder, neck
