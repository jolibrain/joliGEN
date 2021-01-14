gpu='3'

######################
# loss weight params #
######################
lr=5e-6
momentum=0.99
lambda_d=1
lambda_g=0.1

################
# train params #
################
batch='1'
crop='256'
load='256'


########
# Data #
########
src='/data1/pnsuau/cartier/markers2ring'
#src='/data1/pnsuau/planes/cropped_centered'
name='segmentation_1'
dataset_mode='unaligned_labeled_mask'
#max_dataset_size='1'
checkpoints_dir='./checkpoints/cartier/segmentation'

output_nc='3'
input_nc='3'
nclasses='2'
# init with pre-trained cyclegta5 model
#model='cycle_gan_semantic_mask'
model='segmentation'

#base_model="base_models/${model}-${src}-iter${baseiter}.pth"
#outdir="${resdir}/${model}/lr${lr}_crop${crop}_ld${lambda_d}_lg${lambda_g}_momentum${momentum}"
display_port=8000
display_freq=1000
save_latest_freq=1000
# Run python script #
#CUDA_VISIBLE_DEVICES=${gpu}
python3 ../train.py \
    --dataroot ${src} --name $name \
    --gpu ${gpu} --save_latest_freq ${save_latest_freq}\
    --dataset_mode ${dataset_mode} --checkpoints_dir ${checkpoints_dir} \
    --model ${model} --semantic_nclasses ${nclasses} \
    --crop_size ${crop} --load_size ${load} --batch_size ${batch} \
    --save_epoch_freq 10 --display_port ${display_port}\
    --no_flip --input_nc ${input_nc} --output_nc ${output_nc} --display_freq ${display_freq} #--continue_train --epoch_count 1




