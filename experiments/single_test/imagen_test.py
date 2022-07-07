#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: czh
@email: zhihao.chen@kuwo.cn
@date: 2022/6/17 14:53
"""
import torch
from imagen_pytorch import Unet, Imagen

device = torch.device('cpu')
# unet for imagen

unet1 = Unet(
    dim=32,
    cond_dim=512,
    dim_mults=(1, 2, 4, 8),
    num_resnet_blocks=3,
    layer_attns=(False, True, True, True),
    layer_cross_attns=(False, True, True, True)
)

unet2 = Unet(
    dim=32,
    cond_dim=512,
    dim_mults=(1, 2, 4, 8),
    num_resnet_blocks=(2, 4, 8, 8),
    layer_attns=(False, False, False, True),
    layer_cross_attns=(False, False, False, True)
)

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    unets=(unet1, unet2),
    image_sizes=(64, 256),
    timesteps=1000,
    cond_drop_prob=0.1
).to(device)

# mock images (get a lot of this) and text encodings from large T5

text_embeds = torch.randn(4, 256, 768).to(device)
text_masks = torch.ones(4, 256).bool().to(device)
images = torch.randn(4, 3, 256, 256).to(device)

# feed images into imagen, training each unet in the cascade

for i in (1, 2):
    loss = imagen(images, text_embeds=text_embeds, text_masks=text_masks, unet_number=i)
    loss.backward()

# do the above for many many many many steps
# now you can sample an image based on the text embeddings from the cascading ddpm

images = imagen.sample(texts=[
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles'
], cond_scale=3.)

print(images.shape)  # (3, 3, 256, 256)
