#!/bin/bash

current_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

gpu=0

########
# Data #
########

if [ $# -ne 1 ]
then
    echo "Usage: $0 [dataroot]"
    exit
fi

src=$1
echo "Data root = ${src}"
checkpoints_dir='./checkpoint_semantic'
echo "Checkpoint dir = ${checkpoints_dir}"
name='train_1'
dataset_mode='unaligned_labeled'

######################
# loss weight params #
######################
lr=0.0002
d_lr=0.0001
batch=2

########
# Model #
########
model='cut'
netG='mobile_resnet_attn'

################
# train params #
################
crop='256'
load='256'
input_nc='3'
output_nc='3'

################
# Visualization #
################
display_env='train_semantic'
display_freq='100'
print_freq='100'
nclasses='10'

# Run python script #

python3 "${current_dir}/../train.py"\
	--dataroot "${src}" --checkpoints_dir "${checkpoints_dir}" --name $name\
	--output_display_env $display_env  --output_display_freq ${display_freq} --output_print_freq ${print_freq}\
	--gpu ${gpu}\
	--G_lr ${lr} --D_lr ${d_lr} --train_batch_size ${batch}\
	--data_crop_size ${crop} --data_load_size ${load} --data_dataset_mode ${dataset_mode}\
	--model_input_nc ${input_nc} --model_output_nc ${output_nc}\
	--model_type ${model} --G_netG $netG\
	--f_s_semantic_nclasses $nclasses\
	--dataaug_no_flip --dataaug_no_rotate
