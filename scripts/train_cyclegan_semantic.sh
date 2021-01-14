gpu='0'
########
# Data #
########
src='/data1/cycada/datasets/svhn2mnist_jpg/'
checkpoints_dir='/data1/pnsuau/train/train_semantic/'
name='train_1'
dataset_mode='unaligned_labeled'

######################
# loss weight params #
######################
lambda_A=50
lambda_B=50
lambda_identity=0.5
lr=0.0002
d_lr=0.0002
batch=2

########
# Model #
########
model='cycle_gan_semantic'
netG='mobile_resnet_9blocks'

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

python3 /home/pnsuau/pytorch-CycleGAN-and-pix2pix/train.py\
	--dataroot ${src} --checkpoints_dir ${checkpoints_dir} --name $name\
	--display_env $display_env  --display_freq ${display_freq} --print_freq ${print_freq}\
	--gpu ${gpu}\
	--lambda_A ${lambda_A} --lambda_B ${lambda_B} --lambda_identity ${lambda_identity}\
	--lr ${lr} --D_lr ${d_lr} --batch_size ${batch}\
	--crop_size ${crop} --load_size ${load} --dataset_mode ${dataset_mode}\
	--input_nc ${input_nc} --output_nc ${output_nc}\
	--model ${model} --netG $netG\
	--semantic_nclasses $nclasses\
	--no_flip --no_rotate
