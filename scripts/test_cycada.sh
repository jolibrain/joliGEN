gpu='1'


########
# Data #
########
src='/data1/pnsuau/cityscapes2gta/'
name='mask_test_1'
dataset_mode='unaligned_labeled_mask'

nclasses='8'
model='cycle_gan_semantic_mask'
num_test=1000





python3 ../test_2.py --dataroot ${src} --name $name --model ${model}\
	--phase test --dataset_mode ${dataset_mode} --num_test ${num_test}\
	--gpu ${gpu} --semantic_nclasses ${nclasses} #--no_dropout

