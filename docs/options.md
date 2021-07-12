
# JoliGAN Options

Here are all the available options to call with `train.py`

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --dataroot | str |  | path to images (should have subfolders trainA, trainB, valA, valB, etc) |
| --name | str | experiment_name | name of the experiment. It decides where to store samples and models |
| --gpu_ids | str | 0 | gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU |
| --checkpoints_dir | str | ./checkpoints | models are saved here |
| --model | str | cycle_gan | chooses which model to use. [cut_semantic_mask \| cycle_gan_semantic_mask_sty2 \| segmentation \| cycle_gan_sty2 \| test \| cycle_gan \| cut_semantic \| cycle_gan_semantic_mask \| re_cycle_gan_semantic_mask \| cycle_gan_semantic \| attention_gan \| cycle_gan_semantic_mask_input \| cut \| template \| re_cut_semantic_mask] |
| --input_nc | int | 3 | # of input image channels: 3 for RGB and 1 for grayscale |
| --output_nc | int | 3 | # of output image channels: 3 for RGB and 1 for grayscale |
| --ngf | int | 64 | # of gen filters in the last conv layer |
| --ndf | int | 64 | # of discrim filters in the first conv layer |
| --netD | str | basic | specify discriminator architecture [basic \| n_layers \| pixel] or any torchvision model [resnet18...]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator |
| --netD_global | str | none | specify discriminator architecture, any torchvision model can be used [resnet18...]. By default no global discriminator will be used. |
| --netG | str | resnet_9blocks | specify generator architecture [resnet_9blocks \| resnet_6blocks \| resnet_attn \| unet_256 \| unet_128 \| stylegan2 \| smallstylegan2] |
| --n_layers_D | int | 3 | only used if netD==n_layers |
| --norm | str | instance | instance normalization or batch normalization [instance \| batch \| none] |
| --init_type | str | normal | network initialization [normal \| xavier \| kaiming \| orthogonal] |
| --init_gain | float | 0.02 | scaling factor for normal, xavier and orthogonal. |
| --no_dropout | flag |  | no dropout for the generator |
| --D_dropout | flag |  | whether to use dropout in the discriminator |
| --D_spectral | flag |  | whether to use spectral norm in the discriminator |
| --G_spectral | flag |  | whether to use spectral norm in the generator |
| --G_padding_type | str | reflect | whether to use padding in the generator, zeros or reflect |
| --dataset_mode | str | unaligned | chooses how datasets are loaded. [unaligned \| aligned \| single \| colorization] |
| --direction | str | AtoB | AtoB or BtoA |
| --serial_batches | flag |  | if true, takes images in order to make batches, otherwise takes them randomly |
| --num_threads | int | 4 | # threads for loading data |
| --batch_size | int | 1 | input batch size |
| --load_size | int | 286 | scale images to this size |
| --crop_size | int | 256 | then crop to this size |
| --max_dataset_size | int | inf | Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded. |
| --preprocess | str | resize_and_crop | scaling and cropping of images at load time [resize_and_crop \| crop \| scale_width \| scale_width_and_crop \| none] |
| --no_flip | flag |  | if specified, do not flip the images for data augmentation |
| --no_rotate | flag |  | if specified, do not rotate the images for data augmentation |
| --affine | float | 0.0 | if specified, apply random affine transforms to the images for data augmentation |
| --affine_translate | float | 0.2 | if random affine specified, translation range (-value*img_size,+value*img_size) value |
| --affine_scale_min | float | 0.8 | if random affine specified, min scale range value |
| --affine_scale_max | float | 1.2 | if random affine specified, max scale range value |
| --affine_shear | int | 45 | if random affine specified, shear range (0,value) |
| --imgaug | flag |  | whether to apply random image augmentation |
| --display_winsize | int | 256 | display window size for both visdom and HTML |
| --epoch | str | latest | which epoch to load? set to latest to use latest cached model |
| --load_iter | int | 0 | which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch] |
| --verbose | flag |  | if specified, print more debugging information |
| --suffix | str |  | customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size} |
| --semantic_nclasses | int | 10 | number of classes of the semantic loss classifier |
| --semantic_threshold | float | 1.0 | threshold of the semantic classifier loss below with semantic loss is applied |
| --display_networks | flag |  | Set True if you want to display networks on port 8000 |
| --compute_fid | flag |  |  |
| --fid_every | int | 1000 |  |
| --normG | str | instance | instance normalization or batch normalization for G |
| --normD | str | instance | instance normalization or batch normalization for D |
| --no_antialias | flag |  | if specified, use stride=2 convs instead of antialiased-downsampling (sad) |
| --no_antialias_up | flag |  | if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv] |
| --stylegan2_G_num_downsampling | int | 1 | Number of downsampling layers used by StyleGAN2Generator |
| --D_label_smooth | flag |  | whether to use one-sided label smoothing with discriminator |
| --D_noise | float | 0.0 | whether to add instance noise to discriminator inputs |


## attention_gan

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --lambda_A | float | 10.0 | weight for cycle loss (A -> B -> A) |
| --lambda_B | float | 10.0 | weight for cycle loss (B -> A -> B) |
| --lambda_identity | float | 0.5 | use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1 |
| --nb_attn | int | 10 | number of attention masks |
| --nb_mask_input | int | 1 | number of attention masks which will be applied on the input image |


## cut

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --CUT_mode | str | CUT |  |
| --lambda_GAN | float | 1.0 | weight for GAN loss：GAN(G(X)) |
| --lambda_NCE | float | 1.0 | weight for NCE loss: NCE(G(X), X) |
| --nce_idt | str2bool | True | use NCE loss for identity mapping: NCE(G(Y), Y)) |
| --nce_layers | str | 0,4,8,12,16 | compute NCE loss on which layers |
| --nce_includes_all_negatives_from_minibatch | str2bool | False | (used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details. |
| --netF | str | mlp_sample | how to downsample the feature map |
| --netF_nc | int | 256 |  |
| --nce_T | float | 0.07 | temperature for NCE loss |
| --num_patches | int | 256 | number of patches per layer |
| --flip_equivariance | str2bool | False | Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT |
| --use_label_B | flag |  | if true domain B has labels too |


## cut_semantic

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --CUT_mode | str | CUT |  |
| --lambda_GAN | float | 1.0 | weight for GAN loss：GAN(G(X)) |
| --lambda_NCE | float | 1.0 | weight for NCE loss: NCE(G(X), X) |
| --nce_idt | str2bool | True | use NCE loss for identity mapping: NCE(G(Y), Y)) |
| --nce_layers | str | 0,4,8,12,16 | compute NCE loss on which layers |
| --nce_includes_all_negatives_from_minibatch | str2bool | False | (used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details. |
| --netF | str | mlp_sample | how to downsample the feature map |
| --netF_nc | int | 256 |  |
| --nce_T | float | 0.07 | temperature for NCE loss |
| --num_patches | int | 256 | number of patches per layer |
| --flip_equivariance | str2bool | False | Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT |
| --use_label_B | flag |  | if true domain B has labels too |
| --train_cls_B | flag |  | if true cls will be trained not only on domain A but also on domain B |
| --cls_template | str | basic | classifier/regressor model type, from torchvision (resnet18, ...), default is custom simple model |
| --cls_pretrained | flag |  | whether to use a pretrained model, available for non "basic" model only |
| --lr_f_s | float | 0.0002 | f_s learning rate |
| --contrastive_noise | float | 0.0 | noise on constrastive classifier |
| --regression | flag |  | if true cls will be a regressor and not a classifier |
| --lambda_sem | float | 1.0 | weight for semantic loss |
| --l1_regression | flag |  | if true l1 loss will be used to compute regressor loss |


## cut_semantic_mask

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --CUT_mode | str | CUT |  |
| --lambda_GAN | float | 1.0 | weight for GAN loss：GAN(G(X)) |
| --lambda_NCE | float | 1.0 | weight for NCE loss: NCE(G(X), X) |
| --nce_idt | str2bool | True | use NCE loss for identity mapping: NCE(G(Y), Y)) |
| --nce_layers | str | 0,4,8,12,16 | compute NCE loss on which layers |
| --nce_includes_all_negatives_from_minibatch | str2bool | False | (used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details. |
| --netF | str | mlp_sample | how to downsample the feature map |
| --netF_nc | int | 256 |  |
| --nce_T | float | 0.07 | temperature for NCE loss |
| --num_patches | int | 256 | number of patches per layer |
| --flip_equivariance | str2bool | False | Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT |
| --use_label_B | flag |  | if true domain B has labels too |
| --train_f_s_B | flag |  | if true f_s will be trained not only on domain A but also on domain B |
| --fs_light | flag |  | whether to use a light (unet) network for f_s |
| --lr_f_s | float | 0.0002 | f_s learning rate |
| --out_mask | flag |  | use loss out mask |
| --lambda_out_mask | float | 10.0 | weight for loss out mask |
| --loss_out_mask | str | L1 | loss mask |
| --contrastive_noise | float | 0.0 | noise on constrastive classifier |
| --lambda_sem | float | 1.0 | weight for semantic loss |


## cycle_gan

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --lambda_A | float | 10.0 | weight for cycle loss (A -> B -> A) |
| --lambda_B | float | 10.0 | weight for cycle loss (B -> A -> B) |
| --lambda_identity | float | 0.5 | use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1 |
| --use_label_B | flag |  | if true domain B has labels too |
| --rec_noise | float | 0.0 | whether to add noise to reconstruction |


## cycle_gan_semantic

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --lambda_A | float | 10.0 | weight for cycle loss (A -> B -> A) |
| --lambda_B | float | 10.0 | weight for cycle loss (B -> A -> B) |
| --lambda_identity | float | 0.5 | use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1 |
| --use_label_B | flag |  | if true domain B has labels too |
| --rec_noise | float | 0.0 | whether to add noise to reconstruction |
| --train_cls_B | flag |  | if true cls will be trained not only on domain A but also on domain B, if true use_label_B needs to be True |
| --cls_template | str | basic | classifier/regressor model type, from torchvision (resnet18, ...), default is custom simple model |
| --cls_pretrained | flag |  | whether to use a pretrained model, available for non "basic" model only |
| --lr_f_s | float | 0.0002 | f_s learning rate |
| --regression | flag |  | if true cls will be a regressor and not a classifier |
| --lambda_sem | float | 1.0 | weight for semantic loss |
| --lambda_CLS | float | 1.0 | weight for CLS loss |
| --l1_regression | flag |  | if true l1 loss will be used to compute regressor loss |


## cycle_gan_semantic_mask

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --lambda_A | float | 10.0 | weight for cycle loss (A -> B -> A) |
| --lambda_B | float | 10.0 | weight for cycle loss (B -> A -> B) |
| --lambda_identity | float | 0.5 | use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1 |
| --use_label_B | flag |  | if true domain B has labels too |
| --rec_noise | float | 0.0 | whether to add noise to reconstruction |
| --out_mask | flag |  | use loss out mask |
| --lambda_out_mask | float | 10.0 | weight for loss out mask |
| --loss_out_mask | str | L1 | loss mask |
| --charbonnier_eps | float | 1e-06 | Charbonnier loss epsilon value |
| --disc_in_mask | flag |  | use in-mask discriminator |
| --train_f_s_B | flag |  | if true f_s will be trained not only on domain A but also on domain B |
| --fs_light | flag |  | whether to use a light (unet) network for f_s |
| --lr_f_s | float | 0.0002 | f_s learning rate |
| --nb_attn | int | 10 | number of attention masks |
| --nb_mask_input | int | 1 | number of attention masks which will be applied on the input image |
| --lambda_sem | float | 1.0 | weight for semantic loss |
| --madgrad | flag |  | if true madgrad optim will be used |


## cycle_gan_semantic_mask_input

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --lambda_A | float | 10.0 | weight for cycle loss (A -> B -> A) |
| --lambda_B | float | 10.0 | weight for cycle loss (B -> A -> B) |
| --lambda_identity | float | 0.5 | use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1 |


## cycle_gan_semantic_mask_sty2

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --lambda_A | float | 10.0 | weight for cycle loss (A -> B -> A) |
| --lambda_B | float | 10.0 | weight for cycle loss (B -> A -> B) |
| --lambda_identity | float | 0.5 | use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1 |
| --lambda_G | float | 1.0 | weight for generator loss |
| --out_mask | flag |  | use loss out mask |
| --lambda_out_mask | float | 10.0 | weight for loss out mask |
| --loss_out_mask | str | L1 | loss mask |
| --charbonnier_eps | float | 1e-06 | Charbonnier loss epsilon value |
| --train_f_s_B | flag |  | if true f_s will be trained not only on domain A but also on domain B |
| --fs_light | flag |  | whether to use a light (unet) network for f_s |
| --lr_f_s | float | 0.0002 | f_s learning rate |
| --D_noise | flag |  | whether to add instance noise to discriminator inputs |
| --D_label_smooth | flag |  | whether to use one-sided label smoothing with discriminator |
| --rec_noise | float | 0.0 | whether to add noise to reconstruction |
| --wplus | flag |  | whether to work in W+ latent space |
| --wskip | flag |  | whether to use skip connections to latent wplus heads |
| --truncation | float | 1 | whether to use truncation trick (< 1) |
| --decoder_size | int | 512 |  |
| --d_reg_every | int | 16 | regularize discriminator each x iterations, no reg if set to 0 |
| --g_reg_every | int | 4 | regularize decider sty2 each x iterations, no reg if set to 0 |
| --r1 | float | 10 |  |
| --mixing | float | 0.9 |  |
| --path_batch_shrink | int | 2 |  |
| --path_regularize | float | 2 |  |
| --no_init_weight_D_sty2 | flag |  |  |
| --no_init_weight_dec_sty2 | flag |  |  |
| --no_init_weight_G | flag |  |  |
| --load_weight_decoder | flag |  |  |
| --percept_loss | flag |  | whether to use perceptual loss for reconstruction and identity |
| --randomize_noise | flag |  | whether to use random noise in sty2 decoder |
| --D_lightness | int | 1 | sty2 discriminator lightness, 1: normal, then 2, 4, 8 for less parameters |
| --w_loss | flag |  |  |
| --lambda_w_loss | float | 10.0 |  |
| --n_loss | flag |  |  |
| --lambda_n_loss | float | 10.0 |  |
| --cam_loss | flag |  |  |
| --lambda_cam | float | 10.0 |  |
| --sty2_clamp | flag |  |  |


## cycle_gan_sty2

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --lambda_A | float | 10.0 | weight for cycle loss (A -> B -> A) |
| --lambda_B | float | 10.0 | weight for cycle loss (B -> A -> B) |
| --lambda_identity | float | 0.5 | use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1 |
| --lambda_G | float | 1.0 | weight for generator loss |
| --D_noise | flag |  | whether to add instance noise to discriminator inputs |
| --D_label_smooth | flag |  | whether to use one-sided label smoothing with discriminator |
| --rec_noise | float | 0.0 | whether to add noise to reconstruction |
| --wplus | flag |  | whether to work in W+ latent space |
| --wskip | flag |  | whether to use skip connections to latent wplus heads |
| --truncation | float | 1 | whether to use truncation trick (< 1) |
| --decoder_size | int | 512 |  |
| --d_reg_every | int | 16 |  |
| --g_reg_every | int | 4 |  |
| --r1 | float | 10 |  |
| --path_batch_shrink | int | 2 |  |
| --path_regularize | float | 2 |  |
| --no_init_weight_D_sty2 | flag |  |  |
| --no_init_weight_dec_sty2 | flag |  |  |
| --no_init_weight_G | flag |  |  |
| --load_weight_decoder | flag |  |  |
| --percept_loss | flag |  | whether to use perceptual loss for reconstruction and identity |
| --D_lightness | int | 1 | sty2 discriminator lightness, 1: normal, then 2, 4, 8 for less parameters |
| --w_loss | flag |  |  |
| --lambda_w_loss | float | 10.0 |  |
| --n_loss | flag |  |  |
| --lambda_n_loss | float | 10.0 |  |
| --cam_loss | flag |  |  |
| --lambda_cam | float | 10.0 |  |
| --sty2_clamp | flag |  |  |


## re_cut_semantic_mask

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --CUT_mode | str | CUT |  |
| --lambda_GAN | float | 1.0 | weight for GAN loss：GAN(G(X)) |
| --lambda_NCE | float | 1.0 | weight for NCE loss: NCE(G(X), X) |
| --nce_idt | str2bool | True | use NCE loss for identity mapping: NCE(G(Y), Y)) |
| --nce_layers | str | 0,4,8,12,16 | compute NCE loss on which layers |
| --nce_includes_all_negatives_from_minibatch | str2bool | False | (used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details. |
| --netF | str | mlp_sample | how to downsample the feature map |
| --netF_nc | int | 256 |  |
| --nce_T | float | 0.07 | temperature for NCE loss |
| --num_patches | int | 256 | number of patches per layer |
| --flip_equivariance | str2bool | False | Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT |
| --use_label_B | flag |  | if true domain B has labels too |
| --train_f_s_B | flag |  | if true f_s will be trained not only on domain A but also on domain B |
| --fs_light | flag |  | whether to use a light (unet) network for f_s |
| --lr_f_s | float | 0.0002 | f_s learning rate |
| --out_mask | flag |  | use loss out mask |
| --lambda_out_mask | float | 10.0 | weight for loss out mask |
| --loss_out_mask | str | L1 | loss mask |
| --contrastive_noise | float | 0.0 | noise on constrastive classifier |
| --lambda_sem | float | 1.0 | weight for semantic loss |
| --adversarial_loss_p | flag |  | if True, also train the prediction model with an adversarial loss |
| --nuplet_size | int | 3 | Number of frames loaded |
| --netP | str | unet_128 | specify P architecture [resnet_9blocks \| resnet_6blocks \| resnet_attn \| unet_256 \| unet_128] |
| --no_train_P_fake_images | flag |  | if True, P wont be trained over fake images projections |
| --projection_threshold | float | 1.0 | threshold of the real images projection loss below with fake projection and fake reconstruction losses are applied |
| --P_lr | float | 0.0002 | initial learning rate for P networks |


## re_cycle_gan_semantic_mask

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --lambda_A | float | 10.0 | weight for cycle loss (A -> B -> A) |
| --lambda_B | float | 10.0 | weight for cycle loss (B -> A -> B) |
| --lambda_identity | float | 0.5 | use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1 |
| --use_label_B | flag |  | if true domain B has labels too |
| --rec_noise | float | 0.0 | whether to add noise to reconstruction |
| --out_mask | flag |  | use loss out mask |
| --lambda_out_mask | float | 10.0 | weight for loss out mask |
| --loss_out_mask | str | L1 | loss mask |
| --charbonnier_eps | float | 1e-06 | Charbonnier loss epsilon value |
| --disc_in_mask | flag |  | use in-mask discriminator |
| --train_f_s_B | flag |  | if true f_s will be trained not only on domain A but also on domain B |
| --fs_light | flag |  | whether to use a light (unet) network for f_s |
| --lr_f_s | float | 0.0002 | f_s learning rate |
| --nb_attn | int | 10 | number of attention masks |
| --nb_mask_input | int | 1 | number of attention masks which will be applied on the input image |
| --lambda_sem | float | 1.0 | weight for semantic loss |
| --madgrad | flag |  | if true madgrad optim will be used |
| --adversarial_loss_p | flag |  | if True, also train the prediction model with an adversarial loss |
| --nuplet_size | int | 3 | Number of frames loaded |
| --netP | str | unet_128 | specify P architecture [resnet_9blocks \| resnet_6blocks \| resnet_attn \| unet_256 \| unet_128] |
| --no_train_P_fake_images | flag |  | if True, P wont be trained over fake images projections |
| --projection_threshold | float | 1.0 | threshold of the real images projection loss below with fake projection and fake reconstruction losses are applied |
| --P_lr | float | 0.0002 | initial learning rate for P networks |


## segmentation

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --lambda_A | float | 10.0 | weight for cycle loss (A -> B -> A) |
| --lambda_B | float | 10.0 | weight for cycle loss (B -> A -> B) |
| --lambda_identity | float | 0.5 | use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1 |


## template

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| --lambda_regression | float | 1.0 | weight for the regression loss |

