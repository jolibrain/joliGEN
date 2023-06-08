<div align="center">
<img src="imgs/joligen.svg" width="512">
</div>

<h1 align="center">Generative AI Toolset with GANs and Diffusion for Real-World Applications</h1>

**JoliGEN** provides easy-to-use generative AI for image to image transformations.

Main Features:

- JoliGEN support both GAN and Diffusion models for unpaired and paired image to image translation tasks, including domain and style adaptation with conservation of semantics such as image and object classes, masks, ...

- JoliGEN generative capabilities are targeted at real world applications such as Augmented Reality, Dataset Smart Augmentation and object insertion, Synthetic to real transforms.

- JoliGEN allows for fast and stable training with astonishing results. A server with REST API is provided that allows for simplified deployment and usage.

- JoliGEN has a large scope of options and parameters. To not get overwhelmed, follow the simple steps below. There are then links to more detailed documentation on models, dataset formats, and data augmentation.

## Use cases

- AR and metaverse: replace any image element with super-realistic objects
- Image manipulation: seamlessly insert or remove objects/elements in images
- Image to image translation while preserving semantic, e.g. existing source dataset annotations
- Simulation to reality translation while preserving elements, metrics, ...
- Image to image translation to cope with scarce data

This is achieved by combining powerful and customized generator architectures, bags of discriminators, and configurable neural networks and losses that ensure conservation of fundamental elements between source and target images.

## Example results

### Image translation while preserving the class

Mario to Sonic while preserving the action (running, jumping, ...)

![Clipboard - June 6, 2022 9 44 PM](https://user-images.githubusercontent.com/3530657/196461791-9ff55a47-1e74-4ee7-ad3b-0a915dee6ae6.png)
![Clipboard - June 5, 2022 12 02 PM](https://user-images.githubusercontent.com/3530657/196461802-21d3015b-f5e8-467b-9096-78fcabd1f57b.png)

### Object insertion

Car insertion (BDD100K) with Diffusion
![image](https://user-images.githubusercontent.com/3530657/196428508-3eae3415-8e15-4505-9e97-41c0ba99350e.png)
![image](https://user-images.githubusercontent.com/3530657/196428593-6ad8e229-368a-4714-a1cc-8aa8210beaad.png)

Glasses insertion (FFHQ) with Diffusion

<img src="https://github.com/jolibrain/joliGEN/assets/3530657/eba7920d-4430-4f46-b65c-6cf2267457b0" alt="drawing" width="512"/>
<img src="https://github.com/jolibrain/joliGEN/assets/3530657/ef908a7f-375f-4d0a-afec-72d1ee7eaafe" alt="drawing" width="512"/>

### Object removal

Glasses removal with GANs

![Clipboard - November 9, 2022 4_33 PM](https://user-images.githubusercontent.com/3530657/200873590-6d1abe9a-7d86-458a-a9a5-97a1bcf4b816.png)
![Clipboard - November 9, 2022 10_40 AM](https://user-images.githubusercontent.com/3530657/200873601-e8c2d165-af58-4b39-a0bf-ecab510981c5.png)

### Style transfer while preserving label boxes (e.g. cars, pedestrians, street signs, ...)

Day to night (BDD100K) with Transformers and GANs
![image](https://user-images.githubusercontent.com/3530657/196472056-b342c326-056f-4680-ad8d-4bf932b1404a.png)

Clear to snow (BDD100K) by applying a generator multiple times to add snow incrementally
![image](https://user-images.githubusercontent.com/3530657/196426503-bfaee698-b135-493f-81a6-644881cc1a5c.png)

Clear to overcast (BDD100K)
![image](https://user-images.githubusercontent.com/3530657/196426571-f7e6189b-fa3a-4f5e-b6d5-fd580d14f29c.png)

Clear to rainy (BDD100K)
![image](https://user-images.githubusercontent.com/3530657/196426461-e983c48f-ce19-4e83-a490-7a73b28c8181.png)
![image](https://user-images.githubusercontent.com/3530657/196426623-deb7c00d-77e7-448e-827f-2423fd76b0ef.png)

## Features

- SoTA image to image translation
- Semantic consistency: conservation of labels of many types: bounding boxes, masks, classes.
- SoTA discriminator models: [projected](https://arxiv.org/abs/2111.01007), [vision_aided](https://arxiv.org/abs/2112.09130), custom transformers.
- Advanced generators: [real-time](https://github.com/jolibrain/joliGEN/blob/chore_new_readme/models/modules/resnet_architecture/resnet_generator.py#L388), [transformers](https://arxiv.org/abs/2203.16015), [hybrid transformers-CNN](https://github.com/jolibrain/joliGEN/blob/chore_new_readme/models/modules/segformer/segformer_generator.py#L95), [Attention-based](https://arxiv.org/abs/1911.11897), [UNet with attention](https://github.com/jolibrain/joliGEN/blob/chore_new_readme/models/modules/unet_generator_attn/unet_generator_attn.py#L323), [StyleGAN2](https://github.com/jolibrain/joliGEN/blob/chore_new_readme/models/modules/stylegan_networks.py)
- Multiple models based on adversarial and diffusion generation: [CycleGAN](https://arxiv.org/abs/1703.10593), [CyCADA](https://arxiv.org/abs/1711.03213), [CUT](https://arxiv.org/abs/2007.15651), [Palette](https://arxiv.org/abs/2111.05826)
- GAN data augmentation mechanisms: [APA](https://arxiv.org/abs/2111.06849), discriminator noise injection, standard image augmentation, online augmentation through sampling around bounding boxes
- Output quality metrics: [FID](https://github.com/mseitzer/pytorch-fid)
- Server with [REST API](https://github.com/jolibrain/joliGEN/blob/master/server/api_specs.md)
- Support for both CPU and GPU
- [Dockerized server](https://github.com/jolibrain/joliGEN/blob/master/docker/Dockerfile.server)
- Production-grade deployment in C++ via [DeepDetect](https://github.com/jolibrain/deepdetect/)

---

## Quick Start

### Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Installation

Clone this repo:
```bash
git clone --recursive https://github.com/jolibrain/joliGEN.git
cd joliGEN
```

Install [PyTorch](http://pytorch.org) and other dependencies (torchvision, [visdom](https://github.com/facebookresearch/visdom) with:
```bash
pip install -r requirements.txt --upgrade
```

## Dataset formats

#### Image to image without semantics

Example: horse to zebra from two sets of images
Dataset: https://joligen.com/datasets/horse2zebra.zip

```
horse2zebra/
horse2zebra/trainA  # horse images
horse2zebra/trainB  # zebra images
horse2zebra/testA
horse2zebra/testB
```

#### Image to image with class semantics

Example: font number conversion
Dataset: https://joligen.com/datasets/mnist2USPS.zip

```
mnist2USPS/
mnist2USPS/trainA
mnist2USPS/trainA/0  # images of number 0
mnist2USPS/trainA/1  # images of number 1
mnist2USPS/trainA/2  # images of number 2
...
mnist2USPS/trainB
mnist2USPS/trainB/0  # images of target number 0
mnist2USPS/trainB/1  # images of target number 1
mnist2USPS/trainB/2  # images of target number 2
```

#### Image to image with mask semantics

Example: Add glasses to a face without modifying the rest of the face
Dataset: https://joligen.com/datasets/noglasses2glasses_ffhq_mini.zip
Full dataset: https://joligen.com/datasets/noglasses2glasses_ffhq.zip

```
noglasses2glasses_ffhq_mini
noglasses2glasses_ffhq_mini/trainA
noglasses2glasses_ffhq_mini/trainA/img
noglasses2glasses_ffhq_mini/trainA/img/0000.png # source image, e.g. face without glasses
...
noglasses2glasses_ffhq_mini/trainA/bbox
noglasses2glasses_ffhq_mini/trainA/bbox/0000.png # source mask, e.g. mask around eyes
...
noglasses2glasses_ffhq_mini/trainA/paths.txt # list of associated source / mask images
noglasses2glasses_ffhq_mini/trainB
noglasses2glasses_ffhq_mini/trainB/img
noglasses2glasses_ffhq_mini/trainB/img/0000.png # target image, e.g. face with glasses
...
noglasses2glasses_ffhq_mini/trainB/bbox
noglasses2glasses_ffhq_mini/trainB/bbox/0000.png # target mask, e.g. mask around glasses
...
noglasses2glasses_ffhq_mini/trainB/paths.txt # list of associated target / mask images
```

#### Image to image with bounding box semantics

Example: Super Mario to Sonic while preserving the position and action, e.g. crouch, jump, still, ...
Dataset: https://joligen.com/datasets/online_mario2sonic_lite.zip
Full dataset: https://joligen.com/datasets/online_mario2sonic_full.tar

```
online_mario2sonic_lite
online_mario2sonic_lite/mario
online_mario2sonic_lite/mario/bbox
online_mario2sonic_lite/mario/bbox/r_mario_frame_19538.jpg.txt # contains bboxes, see format below
online_mario2sonic_lite/mario/imgs
online_mario2sonic_lite/mario/imgs/mario_frame_19538.jpg
online_mario2sonic_lite/mario/all.txt # list of associated source image / bbox file, 
...
online_mario2sonic_lite/sonic
online_mario2sonic_lite/sonic/bbox
online_mario2sonic_lite/sonic/bbox/r_sonic_frame_81118.jpg.txt
online_mario2sonic_lite/sonic/imgs
online_mario2sonic_lite/sonic/imgs/sonic_frame_81118.jpg
online_mario2sonic_lite/sonic/all.txt # list of associated target image / bbox file
...
online_mario2sonic_lite/trainA
online_mario2sonic_lite/trainA/paths.txt # symlink to ../mario/all.txt
online_mario2sonic_lite/trainB
online_mario2sonic_lite/trainB/paths.txt # symlink to ../sonic/all.txt
```

List file format:
```
cat online_mario2sonic_lite/mario/all.txt
mario/imgs/mario_frame_19538.jpg mario/bbox/r_mario_frame_19538.jpg.txt
```

Bounding boxes format, e.g. `r_mario_frame_19538.jpg.txt`:
```
2 132 167 158 218
```
in this order:
```
cls xmin ymin xmax ymax
```
where `cls` is the class, in this dataset `2` means `running`.

#### Image to image with multiple semantics: bounding box and class

Example: Image seasonal modification while preserving objects with mask (cars, pedestrians, ...) and overall image weather (snow, rain, clear, ...) with class
Dataset: https://joligen.com/datasets/daytime2dawn_dusk_lite.zip

```
daytime2dawn_dusk_lite
daytime2dawn_dusk_lite/dawn_dusk
daytime2dawn_dusk_lite/dawn_dusk/img
daytime2dawn_dusk_lite/dawn_dusk/mask
daytime2dawn_dusk_lite/daytime
daytime2dawn_dusk_lite/daytime/img
daytime2dawn_dusk_lite/daytime/mask
daytime2dawn_dusk_lite/trainA
daytime2dawn_dusk_lite/trainA/paths.txt
daytime2dawn_dusk_lite/trainB
daytime2dawn_dusk_lite/trainB/paths.txt
```

`paths.txt` format:
```
cat trainA/paths.txt
daytime/img/00054602-3bf57337.jpg 2 daytime/mask/00054602-3bf57337.png
```
in this order: `source image path`, `image class`, `image mask`, where `image class` in this dataset represents the weather class.

#### Other semantics

Other semantics are possible, i.e. an algorithm that runs on both source and target

## JoliGEN training

Training requires the following:
- GPU
- a `checkpoints` directory to be specified in which model weights are stored
- a [Visdom](https://github.com/fossasia/visdom) server, by default the training script starts a Visdom server on http://0.0.0.0:8097 if none is running
- Go to http://localhost:8097 to follow training losses and image result samples

JoliGEN has (too) many options, for finer grained control, see the [full option list](docs/options.md).

### Training image to image without semantics
 
Modify as required and run with the following line command:
```
python3 train.py --dataroot /path/to/horse2zebra --checkpoints_dir /path/to/checkpoints --name horse2zebra \
--data_load_size 256 --data_crop_size 256 --train_n_epochs 200 \
--train_G_lr 0.0002 --train_D_lr 0.0001 \
--data_dataset_mode unaligned --train_n_epochs_decay 100 --model_type cut --G_netG mobile_resnet_attn \
--dataaug_no_rotate --train_batch_size 4 --train_iter_size 8
```

### Training with class semantics :
 
```
python3 train.py --dataroot /path/to/mnist2USPS --checkpoints_dir /path/to/checkpoints --name mnist2USPS \
--data_load_size 128 --data_crop_size 128 --train_n_epochs 10 \
--data_dataset_mode unaligned_labeled_cls --train_n_epochs_decay 1 --model_type cut --cls_semantic_nclasses 10 \
--train_G_lr 0.00002 --train_D_lr 0.00001 \
--train_sem_use_label_B --train_semantic_cls --dataaug_no_rotate --dataaug_no_flip --dataaug_D_noise 0.001 \
--G_netG mobile_resnet --G_nblocks 6 --D_netDs basic --train_batch_size 4 --train_iter_size 2
```

### Training with mask semantics :
 
```
python3 train.py --dataroot /path/to/noglasses2glasses_ffhq/ --checkpoints_dir /path/to/checkpoints/ \
--name noglasses2glasses --output_display_env noglasses2glasses --output_display_freq 200 --output_print_freq 200 \
--train_G_lr 0.0002 --train_D_lr 0.0001 --train_sem_lr_f_s 0.0002 --data_crop_size 256 --data_load_size 256 \
--data_dataset_mode unaligned_labeled_mask --model_type cut --train_semantic_mask --train_batch_size 2 \
--train_iter_size 1 --model_input_nc 3 --model_output_nc 3 --f_s_net unet --train_mask_f_s_B \
--train_mask_out_mask --f_s_semantic_nclasses 2 --G_netG mobile_resnet_attn --alg_cut_nce_idt \
--train_sem_use_label_B --D_netDs projected_d basic vision_aided --D_proj_interp 256 \
--D_proj_network_type efficientnet --train_G_ema --G_padding_type reflect --dataaug_no_rotate \
--data_relative_paths
```

### Training with bounding box semantics and online sampling around boxes as data augmentation:
 
```
python3 train.py --dataroot /path/to/online_mario2sonic/ --checkpoints_dir /path/to/checkpoints/ \
--name mario2sonic --output_display_env mario2sonic --output_display_freq 200 --output_print_freq 200 \
--train_G_lr 0.0002 --train_D_lr 0.0001 --train_sem_lr_f_s 0.0002 --data_crop_size 128 --data_load_size 180 \
--data_dataset_mode unaligned_labeled_mask_online --model_type cut --train_semantic_m --train_batch_size 2 \
--train_iter_size 1 --model_input_nc 3 --model_output_nc 3 --f_s_net unet --train_mask_f_s_B \
--train_mask_out_mask --data_online_creation_crop_size_A 128 --data_online_creation_crop_delta_A 50 \
--data_online_creation_mask_delta_A 50 --data_online_creation_crop_size_B 128 \
--data_online_creation_crop_delta_B 15 --data_online_creation_mask_delta_B 15 \
--f_s_semantic_nclasses 2 --G_netG segformer_attn_conv \
--G_config_segformer models/configs/segformer/segformer_config_b0.json --alg_cut_nce_idt --train_sem_use_label_B \
--D_netDs projected_d basic vision_aided --D_proj_interp 256 --D_proj_network_type vitsmall \
--train_G_ema --G_padding_type reflect --dataaug_no_rotate --data_relative_paths
```

### Training object insertion :

Trains a diffusion model to insert glasses onto faces.

```
python3 train.py --dataroot /path/to/noglasses2glasses_ffhq/ --checkpoints_dir /path/to/checkpoints/ \
--name noglasses2glasses --data_direction BtoA --output_display_env noglasses2glasses --gpu_ids 0,1 \
--model_type palette --train_batch_size 4 --train_iter_size 16 --model_input_nc 3 --model_output_nc 3 \
--data_relative_paths --train_G_ema --train_optim radam --data_dataset_mode self_supervised_labeled_mask \
--data_load_size 256 --data_crop_size 256 --G_netG unet_mha --data_online_creation_rand_mask_A \
--train_G_lr 0.00002 --train_n_epochs 400 --dataaug_no_rotate --output_display_freq 10000 \
--train_optim adamw --G_nblocks 2
```

## JoliGEN inference

JoliGEN reads the model configuration from a generated `train_config.json` file that is stored in the model directory. When loading a previously trained model, make sure the the `train_config.json` file is in the directory.

Python scripts are provided for inference, that can be used as a baseline for using a model in another codebase.

### Generate an image with a GAN generator model

```
cd scripts
python3 gen_single_image.py --model-in-file /path/to/model/latest_net_G_A.pth \
--img-in /path/to/source.jpg --img-out target.jpg
```

### Generate an image with a diffusion model

Using a pretrained glasses insertion model (see above):

```
python3 gen_single_image_diffusion.py --model-in-file /path/to/model/latest_net_G_A.pth --img-in /path/to/source.jpg\
--mask-in /path/to/mask.jpg --img-out target.jpg  --img-size 256
```

The mask image has 1 where to insert the object and 0 elsewhere.

## JoliGEN server

Ensure everything is installed
```bash
pip install fastapi uvicorn
```

Then run server:
```bash
server/run.sh --host localhost --port 8000
```

## Unit tests
To launch tests before new commits:
```
bash scripts/run_tests.sh /path/to/dir
```

## Models

| Name | Paper |
| -- | -- |
| CycleGAN | https://arxiv.org/abs/1703.10593 | 
| CyCADA | https://arxiv.org/abs/1711.03213 |
| CUT | https://arxiv.org/abs/2007.15651 |
| RecycleGAN | https://arxiv.org/abs/1808.05174 |
| StyleGAN2 | https://arxiv.org/abs/1912.04958 |

## Generator architectures

| Architecture  | Number of parameters |
| -- | -- |
|Resnet 9 blocks|11.378M|
|Mobile resnet 9 blocks|1.987M|
|Resnet attn|11.823M|
|Mobile resnet attn|2.432M|
|Segformer b0|4.158M|
|Segformer attn b0|4.60M|
|Segformer attn b1|14.724M|
|Segformer attn b5|83.016M|
|UNet with mha| ~60M configurable|
|ITTR| ~30M configurable|

## Docker build
To build a docker for joliGEN server:
```
docker build -t jolibrain/joligen_build -f docker/Dockerfile.build .
docker build -t jolibrain/joligen_server -f docker/Dockerfile.server .
```
To run the joliGEN docker:
```
nvidia-docker run jolibrain/myjoligen
```

## Code format
If you want to contribute please use [black](https://github.com/psf/black) code format.
Install:
```
pip install black 
```

Usage :
```
black .
```

If you want to format the code automatically before every commit :
```
pip install pre-commit
pre-commit install
```

## Authors

**JoliGEN** is created and maintained by [Jolibrain](https://www.jolibrain.com/).

Code is making use of [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [CUT](https://github.com/taesungp/contrastive-unpaired-translation), [AttentionGAN](https://github.com/Ha0Tang/AttentionGAN), [MoNCE](https://github.com/fnzhan/MoNCE) among others.

Some elements from JoliGEN are supported by the French National AI program ["Confiance.AI"](https://www.confiance.ai/en/)
