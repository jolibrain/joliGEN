################
 JoliGEN Models
################

.. _models-gan:

***********
 GAN Models
***********

+------------+----------------------------------+
| Name       | Paper                            |
+============+==================================+
| CycleGAN   | https://arxiv.org/abs/1703.10593 |
+------------+----------------------------------+
| CUT        | https://arxiv.org/abs/2007.15651 |
+------------+----------------------------------+
| RecycleGAN | https://arxiv.org/abs/1808.05174 |
+------------+----------------------------------+

.. _models-ddpm:

************
 DDPM Models
************

+------------+----------------------------------+
| Name       | Paper                            |
+============+==================================+
| Palette    | https://arxiv.org/abs/2111.05826 |
+------------+----------------------------------+

.. _models-architectures:

*****************************
 GANs Generator architectures
*****************************

+------------------------+------------------------+----------------------+
| --G_netG               | Architecture           | Number of parameters |
+========================+========================+======================+
| `resnet`               | ResNet                 | 11.378M  (9 blocks)  |
+------------------------+------------------------+----------------------+
| `mobile_resnet`        | Mobile ResNet          | 1.987M  (9 blocks)   |
+------------------------+------------------------+----------------------+
| `resnet_attn`          | ResNet with attention  | 11.823M (9 blocks)   |
+------------------------+------------------------+----------------------+
| `mobile_resnet_attn`   | Mobile ResNet attention| 2.432M  (9 blocks)   |
+------------------------+------------------------+----------------------+
| `unet_{128256}`        | UNet 128/256 (CycleGAN)| 41M / 58M            |
+------------------------+------------------------+----------------------+
| `segformer_conv`       | Segformer b0 to b5     |      4.158M          |
+------------------------+------------------------+----------------------+
| `segformer_attn_conv`  | Segformer attn b0 to b5| 4.60M to 83M         |
+------------------------+------------------------+----------------------+
| `{small}stylegan2`     | StyleGAN2              | 3.3M / 7.8M          |
+------------------------+------------------------+----------------------+
| `unet_mha`             | UNet with mha          | ~60M configurable    |
+------------------------+------------------------+----------------------+
| `ittr`                 | ITTR                   | ~30M configurable    |
+------------------------+------------------------+----------------------+

*********************************
 GANs Discriminator architectures
*********************************
+------------------------+------------------------------+----------------------+
| --D_netD               | Architecture                 | Number of parameters |
+========================+==============================+======================+
| `basic`                | PatchGAN                     | 2.7M                 |
+------------------------+------------------------------+----------------------+
| `n_layers`             | PatchGAN                     | 2.7M+  (3 layers+)   |
+------------------------+------------------------------+----------------------+
| `pixel`                | PixelGAN                     | 0.009M+  (3 layers)  |
+------------------------+------------------------------+----------------------+
| `stylegan2`            | StyleGAN2 disc               |                      |
+------------------------+------------------------------+----------------------+
| `projected_d`          | Projected: EfficientNet, ViT | 13.6M, 32.7M         |
+------------------------+------------------------------+----------------------+
| `vision_aided`         | VGG, Swin-T, ViT             | 0.132M               |
+------------------------+------------------------------+----------------------+
| `depth`                | BeIT, Swin, ViT (MiDaS)      |  2.7M + 300M (frozen)|
+------------------------+------------------------------+----------------------+
| `sam`                  | ViT (Segment Anything)       |  2.7M + 93M+ (frozen)|                    
+------------------------+------------------------------+----------------------+
| `temporal`             | Projected: EfficientNet, ViT | 13.6M, 32.7M         |
+------------------------+------------------------------+----------------------+
| `mask`                 | PatchGAN                     | 2.7M+                |
+------------------------+------------------------------+----------------------+

====================
Discriminator losses
====================

**basic, n_layers and stylegan2:**
They compute the standard GAN loss.
They produce good results at the pixel level (style, texture, overall look).

**projected_d:**
Use the projection of the input image and apply a contrastive loss on it.
This loss is good for style transfer tasks. It will encourage the generator to
produce images that are similar to the input image in the latent space of the discriminator.

**vision_aided:**
Use a pretrained model as a standard image classifier.
See the `paper <https://arxiv.org/abs/2112.09130>`_ for more information.

**depth:**
Use a trained model able to predict the depth of the input image.
This model will be used as a regularizer for image-to-image tasks.
The generator is encouraged to produce images that are similar to the conditional image
in terms of depth.

The model used to predict the depth-map is not trained and is used as is. Thus, we
do not need the ground truth depth-map labels.

**sam:**
Same regularization as above but with the Segment Anything model, a segmenter.

**temporal:**
The discriminator is applied to a successive number of frames. This loss is used
for video generation tasks, to add temporal consistency.

**mask:**
The discriminator should output the same masks than the input conditional image.
It is another form of regularization, along with the *sam* and *depth* regularizers.

===========================
Discriminator architectures
===========================

**PatchGAN and PixelGAN:**
This model discriminates only patches of the input image.
This makes it a good discriminator for local features of the image such as
the style and the texture.
Thus it is generally a good discriminator for style transfer tasks.
See the `Pix2Pix paper <https://arxiv.org/abs/1611.07004>`_ for more information about this model.

**PixelGAN:**
This model output a probability of being true or fake for each pixel in the input image.
This model is the same as the PatchGAN but with its receptive field being 1x1.
It will encourage a greater color diversity in the output image.
This model can be used as an additional discriminator, but it will not produce good results
if used as the only discriminator.
It is also very lightweight.

**StyleGAN2:**
This is the classic StyleGAN2 discriminator. It is a powerful discriminator
that can be used for a wide variety of tasks.
See the `paper <https://arxiv.org/abs/1912.04958>`_ for more.

**EfficientNet:**
This is the EfficientNet model from the `paper <https://arxiv.org/abs/1905.11946>`_.
This model is known for its computational efficiency and its good performance.

**ViT, Swin and BEiT:**
They are new models that are based on the transformer architecture.
See the `ViT paper <https://arxiv.org/abs/2010.11929>`_, the `Swin paper <https://arxiv.org/abs/2103.14030>`_
and the `BEiT paper <https://arxiv.org/abs/2106.08254>`_.

**VGG:**
The old good VGG model from the `paper <https://arxiv.org/abs/1409.1556>`_.


*****************************
 DDPM Generator architectures
*****************************
+------------------------+------------------------------+----------------------+
| --G_netG               | Architecture                 | Number of parameters |
+========================+==============================+======================+
| `unet_mha`             | (Efficient) UNet with mha    | 60M+                 |
+------------------------+------------------------------+----------------------+
| `uvit`                 | (Efficient) UNet with ViT    | 64M+                 |
+------------------------+------------------------------+----------------------+
| `resnet_attn`          | ResNet attention             | 1.6M+                |
+------------------------+------------------------------+----------------------+
| `mobile_resnet_attn`   | Mobile ResNet attention      | 2.4M+                |
+------------------------+------------------------------+----------------------+
