![Logo](imgs/joligan.svg)


JoliGAN is an implementation of an unpaired image to image translation. It uses cycle consistency such as CycleGAN but it allows the use of :
- more generator architectures such as styleGAN2 decoder / mobile resnet
- semanctic consistency 
- new losses : out mask loss, w loss (for sty2 decoder)
 
## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone --recursive https://github.com/jolibrain/joliGAN.git
cd joliGAN
```

- Install [PyTorch](http://pytorch.org) and other dependencies (torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate), [FID](https://github.com/jolibrain/pytorch-fid)).  
  - For pip users, please type the command `pip install -r requirements.txt`.

## JoliGAN train

- Options :

|Model|Network|Decoder|
|-|-|-|
|CycleGAN, CycleGAN_semantic, CycleGAN_semantic_mask|resnet, Unet, mobile_resnet|Vanilla, Sty2|

<br>
With a dataset located in directory `dataroot`:

- Train a [cycleGAN](docs/cyclegan.md) :
 
You can tune the hyperparameters in `./scripts/train_cyclegan.sh` and then use the following line command.
```
bash ./scripts/train_cyclegan.sh dataroot
```
<br>

- Train a [cycleGAN with labels](docs/cyclegan_semantic.md) :
 
You can tune the hyperparameters in `./scripts/train_cyclegan_semantic.sh` and then use the following line command.
```
bash ./scripts/train_cyclegan_semantic.sh dataroot
```
<br>

- Train a [cycleGAN with mask labels](docs/cyclegan_semantic_mask.md) :
 
You can tune the hyperparameters in `./scripts/train_cyclegan_semantic_mask.sh` and then use the following line command.
```
bash ./scripts/train_cyclegan_semantic_mask.sh dataroot
```
## [Datasets](docs/datasets.md)
- Unaligned : apple2orange, horse2zebra
- Unaligned with labels : svhn2mnist
- Unaligned with mask labels : glasses2noglasses,


## [Dataloader](docs/dataloader.md)

To choose a dataloader please use the flag `--dataset_mode dataloader_name`.
There are three dataloaders for different dataset architectures :
- Unaligned (`unaligned`) 
- Unaligned with labels (`unaligned_labeled`)
- Unaligned with mask labels (`unaligned_labeled_mask`)

## Acknowledgments
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
