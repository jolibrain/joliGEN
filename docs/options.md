
# JoliGAN Options

Here are all the available options to call with `train.py`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --dataroot | str |  | path to images (should have subfolders trainA, trainB, valA, valB, etc) |
| --name | str | experiment_name | name of the experiment. It decides where to store samples and models |
| --suffix | str |  | customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size} |
| --gpu_ids | str | 0 | gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU |
| --checkpoints_dir | str | ./checkpoints | models are saved here |
| --phase | str | train | train, val, test, etc |
| --ddp_port | str | 12355 |  |
| --model_type | str | cut | chooses which model to use. |
| --model_input_nc | int | 3 | # of input image channels: 3 for RGB and 1 for grayscale |
| --model_output_nc | int | 3 | # of output image channels: 3 for RGB and 1 for grayscale |
| --model_init_type | str | normal | network initialization |
| --model_init_gain | float | 0.02 | scaling factor for normal, xavier and orthogonal. |
| --G_ngf | int | 64 | # of gen filters in the last conv layer |
| --G_netG | str | mobile_resnet_attn | specify generator architecture |
| --G_dropout | flag |  | dropout for the generator |
| --G_spectral | flag |  | whether to use spectral norm in the generator |
| --G_padding_type | str | reflect | whether to use padding in the generator |
| --G_norm | str | instance | instance normalization or batch normalization for G |
| --G_stylegan2_num_downsampling | int | 1 | Number of downsampling layers used by StyleGAN2Generator |
| --G_config_segformer | str | models/configs/segformer/segformer_config_b0.py | path to segforme configuration file for G |
| --G_attn_nb_mask_attn | int | 10 |  |
| --G_attn_nb_mask_input | int | 1 |  |
| --D_ndf | int | 64 | # of discrim filters in the first conv layer |
| --D_netD | str | basic | specify discriminator architecture, D_n_layers allows you to specify the layers in the discriminator |
| --D_netD_global | str | none | specify discriminator architecture, any torchvision model can be used. By default no global discriminator will be used. |
| --D_n_layers | int | 3 | only used if netD==n_layers |
| --D_norm | str | instance | instance normalization or batch normalization for D |
| --D_dropout | flag |  | whether to use dropout in the discriminator |
| --D_spectral | flag |  | whether to use spectral norm in the discriminator |
| --D_proj_interp | int | -1 | whether to force projected discriminator interpolation to a value > 224, -1 means no interpolation |
| --D_proj_network_type | str | efficientnet |  |
| --D_no_antialias | flag |  | if specified, use stride=2 convs instead of antialiased-downsampling (sad) |
| --D_no_antialias_up | flag |  | if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv] |
| --D_proj_config_segformer | str | models/configs/segformer/segformer_config_b0.py | path to segformer configuration file |
| --D_proj_weight_segformer | str | models/configs/segformer/pretrain/segformer_mit-b0.pth | path to segformer weight |
| --f_s_net | str | vgg | specify f_s network [vgg\|unet\|segformer] |
| --f_s_dropout | flag |  | dropout for the semantic network |
| --f_s_semantic_nclasses | int | 2 | number of classes of the semantic loss classifier |
| --f_s_semantic_threshold | float | 1.0 | threshold of the semantic classifier loss below with semantic loss is applied |
| --f_s_all_classes_as_one | flag |  | if true, all classes will be considered as the same one (ie foreground vs background) |
| --f_s_nf | int | 64 | # of filters in the first conv layer of classifier |
| --f_s_config_segformer | str | models/configs/segformer/segformer_config_b0.py | path to segformer configuration file for f_s |
| --f_s_weight_segformer | str | models/configs/segformer/pretrain/segformer_mit-b0.pth | path to segformer weight for f_s |
| --data_dataset_mode | str | unaligned | chooses how datasets are loaded. |
| --data_direction | str | AtoB | AtoB or BtoA |
| --data_serial_batches | flag |  | if true, takes images in order to make batches, otherwise takes them randomly |
| --data_num_threads | int | 4 | # threads for loading data |
| --data_load_size | int | 286 | scale images to this size |
| --data_crop_size | int | 256 | then crop to this size |
| --data_max_dataset_size | int | 1000000000 | Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded. |
| --data_preprocess | str | resize_and_crop | scaling and cropping of images at load time |
| --data_online_creation_crop_size_A | int | 512 | crop to this size during online creation, it needs to be greater than bbox size for domain A |
| --data_online_creation_crop_delta_A | int | 50 | size of crops are random, values allowed are online_creation_crop_size more or less online_creation_crop_delta for domain A |
| --data_online_creation_mask_delta_A | int | 0 | mask offset to allow genaration of a bigger object in domain B (for semantic loss) for domain A |
| --data_online_creation_mask_square_A | flag |  | whether masks should be squared for domain A |
| --data_online_creation_crop_size_B | int | 512 | crop to this size during online creation, it needs to be greater than bbox size for domain B |
| --data_online_creation_crop_delta_B | int | 50 | size of crops are random, values allowed are online_creation_crop_size more or less online_creation_crop_delta for domain B |
| --data_online_creation_mask_delta_B | int | 0 | mask offset to allow genaration of a bigger object in domain B (for semantic loss) for domain B |
| --data_online_creation_mask_square_B | flag |  | whether masks should be squared for domain B |
| --data_sanitize_paths | flag |  | if true, wrong images or labels paths will be removed before training |
| --data_relative_paths | flag |  | whether paths to images are relative to dataroot |


## cut

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --alg_cut_lambda_GAN | float | 1.0 | weight for GAN loss：GAN(G(X)) |
| --alg_cut_lambda_NCE | float | 1.0 | weight for NCE loss: NCE(G(X), X) |
| --alg_cut_nce_idt | str2bool | True | use NCE loss for identity mapping: NCE(G(Y), Y)) |
| --alg_cut_nce_layers | str | 0,4,8,12,16 | compute NCE loss on which layers |
| --alg_cut_nce_includes_all_negatives_from_minibatch | str2bool | False | (used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details. |
| --alg_cut_netF | str | mlp_sample | how to downsample the feature map |
| --alg_cut_netF_nc | int | 256 |  |
| --alg_cut_netF_norm | str | instance | instance normalization or batch normalization for F |
| --alg_cut_netF_dropout | flag |  | whether to use dropout with F |
| --alg_cut_nce_T | float | 0.07 | temperature for NCE loss |
| --alg_cut_num_patches | int | 256 | number of patches per layer |
| --alg_cut_flip_equivariance | str2bool | False | Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT |


## cut_semantic

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --alg_cut_lambda_GAN | float | 1.0 | weight for GAN loss：GAN(G(X)) |
| --alg_cut_lambda_NCE | float | 1.0 | weight for NCE loss: NCE(G(X), X) |
| --alg_cut_nce_idt | str2bool | True | use NCE loss for identity mapping: NCE(G(Y), Y)) |
| --alg_cut_nce_layers | str | 0,4,8,12,16 | compute NCE loss on which layers |
| --alg_cut_nce_includes_all_negatives_from_minibatch | str2bool | False | (used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details. |
| --alg_cut_netF | str | mlp_sample | how to downsample the feature map |
| --alg_cut_netF_nc | int | 256 |  |
| --alg_cut_netF_norm | str | instance | instance normalization or batch normalization for F |
| --alg_cut_netF_dropout | flag |  | whether to use dropout with F |
| --alg_cut_nce_T | float | 0.07 | temperature for NCE loss |
| --alg_cut_num_patches | int | 256 | number of patches per layer |
| --alg_cut_flip_equivariance | str2bool | False | Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT |


## cut_semantic_mask

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --alg_cut_lambda_GAN | float | 1.0 | weight for GAN loss：GAN(G(X)) |
| --alg_cut_lambda_NCE | float | 1.0 | weight for NCE loss: NCE(G(X), X) |
| --alg_cut_nce_idt | str2bool | True | use NCE loss for identity mapping: NCE(G(Y), Y)) |
| --alg_cut_nce_layers | str | 0,4,8,12,16 | compute NCE loss on which layers |
| --alg_cut_nce_includes_all_negatives_from_minibatch | str2bool | False | (used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details. |
| --alg_cut_netF | str | mlp_sample | how to downsample the feature map |
| --alg_cut_netF_nc | int | 256 |  |
| --alg_cut_netF_norm | str | instance | instance normalization or batch normalization for F |
| --alg_cut_netF_dropout | flag |  | whether to use dropout with F |
| --alg_cut_nce_T | float | 0.07 | temperature for NCE loss |
| --alg_cut_num_patches | int | 256 | number of patches per layer |
| --alg_cut_flip_equivariance | str2bool | False | Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT |


## cycle_gan

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --alg_cyclegan_lambda_A | float | 10.0 | weight for cycle loss (A -> B -> A) |
| --alg_cyclegan_lambda_B | float | 10.0 | weight for cycle loss (B -> A -> B) |
| --alg_cyclegan_lambda_identity | float | 0.5 | use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1 |
| --alg_cyclegan_rec_noise | float | 0.0 | whether to add noise to reconstruction |


## cycle_gan_semantic

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --alg_cyclegan_lambda_A | float | 10.0 | weight for cycle loss (A -> B -> A) |
| --alg_cyclegan_lambda_B | float | 10.0 | weight for cycle loss (B -> A -> B) |
| --alg_cyclegan_lambda_identity | float | 0.5 | use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1 |
| --alg_cyclegan_rec_noise | float | 0.0 | whether to add noise to reconstruction |


## cycle_gan_semantic_mask

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --alg_cyclegan_lambda_A | float | 10.0 | weight for cycle loss (A -> B -> A) |
| --alg_cyclegan_lambda_B | float | 10.0 | weight for cycle loss (B -> A -> B) |
| --alg_cyclegan_lambda_identity | float | 0.5 | use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1 |
| --alg_cyclegan_rec_noise | float | 0.0 | whether to add noise to reconstruction |
| --madgrad | flag |  | if true madgrad optim will be used |


## re_cut_semantic_mask

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --alg_cut_lambda_GAN | float | 1.0 | weight for GAN loss：GAN(G(X)) |
| --alg_cut_lambda_NCE | float | 1.0 | weight for NCE loss: NCE(G(X), X) |
| --alg_cut_nce_idt | str2bool | True | use NCE loss for identity mapping: NCE(G(Y), Y)) |
| --alg_cut_nce_layers | str | 0,4,8,12,16 | compute NCE loss on which layers |
| --alg_cut_nce_includes_all_negatives_from_minibatch | str2bool | False | (used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details. |
| --alg_cut_netF | str | mlp_sample | how to downsample the feature map |
| --alg_cut_netF_nc | int | 256 |  |
| --alg_cut_netF_norm | str | instance | instance normalization or batch normalization for F |
| --alg_cut_netF_dropout | flag |  | whether to use dropout with F |
| --alg_cut_nce_T | float | 0.07 | temperature for NCE loss |
| --alg_cut_num_patches | int | 256 | number of patches per layer |
| --alg_cut_flip_equivariance | str2bool | False | Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT |


## re_cycle_gan_semantic_mask

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --alg_cyclegan_lambda_A | float | 10.0 | weight for cycle loss (A -> B -> A) |
| --alg_cyclegan_lambda_B | float | 10.0 | weight for cycle loss (B -> A -> B) |
| --alg_cyclegan_lambda_identity | float | 0.5 | use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1 |
| --alg_cyclegan_rec_noise | float | 0.0 | whether to add noise to reconstruction |
| --madgrad | flag |  | if true madgrad optim will be used |


## segmentation

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |


## template

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --lambda_regression | float | 1.0 | weight for the regression loss |
