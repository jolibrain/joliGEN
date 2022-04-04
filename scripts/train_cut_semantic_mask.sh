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
checkpoints_dir='./checkpoint_semantic_mask'
echo "Checkpoint dir = ${checkpoints_dir}"
name='train_1'
dataset_mode='unaligned_labeled_mask'
max_dataset_size='10'


######################
# loss weight params #
######################
lr=0.0002
d_lr=0.0001
momentum=0.99
lambda_d=1
lambda_g=1
lambda_out_mask=50
lambda_w_context=100


################
# train params #
################
nb_attn='10'
nb_mask_input='0'
batch='1'
crop='256'
load='256'
decoder_size='256'
n_epochs='100'
d_reg_every=16
g_reg_every=4
rec_noise=1


output_nc='3'
input_nc='3'
nclasses='3'

model='cut_semantic_mask'
netG='mobile_resnet_attn'

#base_model="base_models/${model}-${src}-iter${baseiter}.pth"
#outdir="${resdir}/${model}/lr${lr}_crop${crop}_ld${lambda_d}_lg${lambda_g}_momentum${momentum}"
display_port=8097
display_freq=100
print_freq=100
save_latest_freq=10000
fid_every=1000

# Run python script #
#CUDA_VISIBLE_DEVICES=${gpu}
python3 "${current_dir}/../train.py" \
    --dataroot "${src}" --checkpoints_dir "${checkpoints_dir}" --name $name\
    --output_display_env $name  --output_display_freq ${display_freq} --output_print_freq ${print_freq}\
    --gpu ${gpu}\
    --G_lr ${lr} --D_lr ${d_lr}\
    --data_crop_size ${crop} --data_load_size ${load}\
    --data_dataset_mode ${dataset_mode}\
    --model_type ${model} --G_netG $netG\
    --train_batch_size ${batch}\
    --model_input_nc ${input_nc} --model_output_nc ${output_nc}\
