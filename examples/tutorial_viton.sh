#!/bin/bash

# Install joliGEN
cd
git clone https://github.com/jolibrain/joliGEN.git
cd joliGEN
pip install -r requirements.txt --upgrade

# Prepare the Dataset
cd
wget --continue https://www.dropbox.com/s/10bfat0kg4si1bu/zalando-hd-resized.zip
python3 ~/joliGEN/scripts/preprocess_viton.py --zip-file zalando-hd-resized.zip --target-dir ~/datasets/VITON-HD/ --dilate 5

# Train your Diffusion Model
cd ~/joliGEN
python3 train.py --config_json examples/example_ddpm_viton_tutorial.json
cp examples/example_ddpm_viton_tutorial.json ~/checkpoints/VITON-HD/train_config.json

# Inference
mkdir -p ~/inferences
cd ~/joliGEN/scripts
python3 gen_single_image_diffusion.py \
     --model-in-file ~/checkpoints/VITON-HD/latest_net_G_A.pth \
     --img-in ~/datasets/VITON-HD/testA/imgs/00006_00.jpg \
     --mask-in ~/datasets/VITON-HD/testA/mask/00006_00.png \
     --dir-out ~/inferences \
     --nb_samples 4 \
     --img-width 256 \
     --img-height 256
