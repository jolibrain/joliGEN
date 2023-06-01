JoliGEN Options
===============

Here are all the available options to call with :code:`train.py`

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –               | string          | ./checkpoints   | models are      |
| checkpoints_dir |                 |                 | saved here      |
+-----------------+-----------------+-----------------+-----------------+
| –dataroot       | string          | None            | path to images  |
|                 |                 |                 | (should have    |
|                 |                 |                 | subfolders      |
|                 |                 |                 | trainA, trainB, |
|                 |                 |                 | valA, valB,     |
|                 |                 |                 | etc)            |
+-----------------+-----------------+-----------------+-----------------+
| –ddp_port       | string          | 12355           |                 |
+-----------------+-----------------+-----------------+-----------------+
| –gpu_ids        | string          | 0               | gpu ids: e.g. 0 |
|                 |                 |                 | 0,1,2, 0,2. use |
|                 |                 |                 | -1 for CPU      |
+-----------------+-----------------+-----------------+-----------------+
| –model_type     | string          | cut             | chooses which   |
|                 |                 |                 | model to        |
|                 |                 |                 | use.\ **V       |
|                 |                 |                 | alues:**\ *cut, |
|                 |                 |                 | cycle_gan,      |
|                 |                 |                 | palette*        |
+-----------------+-----------------+-----------------+-----------------+
| –name           | string          | experiment_name | name of the     |
|                 |                 |                 | experiment. It  |
|                 |                 |                 | decides where   |
|                 |                 |                 | to store        |
|                 |                 |                 | samples and     |
|                 |                 |                 | models          |
+-----------------+-----------------+-----------------+-----------------+
| –phase          | string          | train           | train, val,     |
|                 |                 |                 | test, etc       |
+-----------------+-----------------+-----------------+-----------------+
| –suffix         | string          |                 | customized      |
|                 |                 |                 | suffix:         |
|                 |                 |                 | opt.name =      |
|                 |                 |                 | opt.name +      |
|                 |                 |                 | suffix: e.g.,   |
|                 |                 |                 | {model}_{netG}_ |
|                 |                 |                 | size{load_size} |
+-----------------+-----------------+-----------------+-----------------+
| –               | int             | 1               | input batch     |
| test_batch_size |                 |                 | size            |
+-----------------+-----------------+-----------------+-----------------+
| –warning_mode   | flag            |                 | whether to      |
|                 |                 |                 | display warning |
+-----------------+-----------------+-----------------+-----------------+
| –with_amp       | flag            |                 | whether to      |
|                 |                 |                 | activate torch  |
|                 |                 |                 | amp on forward  |
|                 |                 |                 | passes          |
+-----------------+-----------------+-----------------+-----------------+
| –with_tf32      | flag            |                 | whether to      |
|                 |                 |                 | activate tf32   |
|                 |                 |                 | for faster      |
|                 |                 |                 | computations    |
|                 |                 |                 | (Ampere GPU and |
|                 |                 |                 | beyond only)    |
+-----------------+-----------------+-----------------+-----------------+
| –wit            | flag            |                 | whether to      |
| h_torch_compile |                 |                 | activate        |
|                 |                 |                 | torch.compile   |
|                 |                 |                 | for some        |
|                 |                 |                 | forward and     |
|                 |                 |                 | backward        |
|                 |                 |                 | functions       |
|                 |                 |                 | (experimental)  |
+-----------------+-----------------+-----------------+-----------------+

Discriminator
-------------

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –D_dropout      | flag            |                 | whether to use  |
|                 |                 |                 | dropout in the  |
|                 |                 |                 | discriminator   |
+-----------------+-----------------+-----------------+-----------------+
| –D_n_layers     | int             | 3               | only used if    |
|                 |                 |                 | netD==n_layers  |
+-----------------+-----------------+-----------------+-----------------+
| –D_ndf          | int             | 64              | # of discrim    |
|                 |                 |                 | filters in the  |
|                 |                 |                 | first conv      |
|                 |                 |                 | layer           |
+-----------------+-----------------+-----------------+-----------------+
| –D_netDs        | array           | [‘projected_d’, | specify         |
|                 |                 | ‘basic’]        | discriminator   |
|                 |                 |                 | architecture,   |
|                 |                 |                 | D_n_layers      |
|                 |                 |                 | allows you to   |
|                 |                 |                 | specify the     |
|                 |                 |                 | layers in the   |
|                 |                 |                 | discriminator.  |
|                 |                 |                 | NB: duplicated  |
|                 |                 |                 | arguments will  |
|                 |                 |                 | be ignored.     |
+-----------------+-----------------+-----------------+-----------------+
| –D_no_antialias | flag            |                 | if specified,   |
|                 |                 |                 | use stride=2    |
|                 |                 |                 | convs instead   |
|                 |                 |                 | of              |
|                 |                 |                 | antialias       |
|                 |                 |                 | ed-downsampling |
|                 |                 |                 | (sad)           |
+-----------------+-----------------+-----------------+-----------------+
| –D_             | flag            |                 | if specified,   |
| no_antialias_up |                 |                 | use             |
|                 |                 |                 | [upconv(learned |
|                 |                 |                 | filter)]        |
|                 |                 |                 | instead of      |
|                 |                 |                 | [up             |
|                 |                 |                 | conv(hard-coded |
|                 |                 |                 | [1,3,3,1]       |
|                 |                 |                 | filter), conv]  |
+-----------------+-----------------+-----------------+-----------------+
| –D_norm         | string          | instance        | instance        |
|                 |                 |                 | normalization   |
|                 |                 |                 | or batch        |
|                 |                 |                 | normalization   |
|                 |                 |                 | for             |
|                 |                 |                 | D\ **Values     |
|                 |                 |                 | :**\ *instance, |
|                 |                 |                 | batch, none*    |
+-----------------+-----------------+-----------------+-----------------+
| –D_proj_c       | string          | mode            | path to         |
| onfig_segformer |                 | ls/configs/segf | segformer       |
|                 |                 | ormer/segformer | configuration   |
|                 |                 | _config_b0.json | file            |
+-----------------+-----------------+-----------------+-----------------+
| –D_proj_interp  | int             | -1              | whether to      |
|                 |                 |                 | force projected |
|                 |                 |                 | discriminator   |
|                 |                 |                 | interpolation   |
|                 |                 |                 | to a value >    |
|                 |                 |                 | 224, -1 means   |
|                 |                 |                 | no              |
|                 |                 |                 | interpolation   |
+-----------------+-----------------+-----------------+-----------------+
| –D_pr           | string          | efficientnet    | projected       |
| oj_network_type |                 |                 | discriminator   |
|                 |                 |                 | architectur     |
|                 |                 |                 | e\ **Values:**\ |
|                 |                 |                 | *efficientnet,  |
|                 |                 |                 | segformer,      |
|                 |                 |                 | vitbase,        |
|                 |                 |                 | vitsmall,       |
|                 |                 |                 | vitsmall2,      |
|                 |                 |                 | vitclip16*      |
+-----------------+-----------------+-----------------+-----------------+
| –D_proj_w       | string          | models/co       | path to         |
| eight_segformer |                 | nfigs/segformer | segformer       |
|                 |                 | /pretrain/segfo | weight          |
|                 |                 | rmer_mit-b0.pth |                 |
+-----------------+-----------------+-----------------+-----------------+
| –D_spectral     | flag            |                 | whether to use  |
|                 |                 |                 | spectral norm   |
|                 |                 |                 | in the          |
|                 |                 |                 | discriminator   |
+-----------------+-----------------+-----------------+-----------------+
| –D              | int             | 4               |                 |
| _temporal_every |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –D_temp         | int             | 30              | how many frames |
| oral_frame_step |                 |                 | between         |
|                 |                 |                 | successive      |
|                 |                 |                 | frames selected |
+-----------------+-----------------+-----------------+-----------------+
| –D_temporal_    | int             | -1              | how many        |
| num_common_char |                 |                 | characters (the |
|                 |                 |                 | first ones) are |
|                 |                 |                 | used to         |
|                 |                 |                 | identify a      |
|                 |                 |                 | video; if =-1   |
|                 |                 |                 | natural sorting |
|                 |                 |                 | is used         |
+-----------------+-----------------+-----------------+-----------------+
| –D_tempora      | int             | 5               | how many        |
| l_number_frames |                 |                 | successive      |
|                 |                 |                 | frames use for  |
|                 |                 |                 | temporal loss   |
+-----------------+-----------------+-----------------+-----------------+
| –D_vision_      | string          | clip+dino+swin  | specify vision  |
| aided_backbones |                 |                 | aided           |
|                 |                 |                 | discriminators  |
|                 |                 |                 | architectures,  |
|                 |                 |                 | they are frozen |
|                 |                 |                 | then output are |
|                 |                 |                 | combined and    |
|                 |                 |                 | fitted with a   |
|                 |                 |                 | linear network  |
|                 |                 |                 | on top, choose  |
|                 |                 |                 | from dino,      |
|                 |                 |                 | clip, swin,     |
|                 |                 |                 | det_coco,       |
|                 |                 |                 | seg_ade and     |
|                 |                 |                 | combine them    |
|                 |                 |                 | with +          |
+-----------------+-----------------+-----------------+-----------------+
| –D_weight_sam   | string          |                 | path to sam     |
|                 |                 |                 | weight for D,   |
|                 |                 |                 | e.g. mod        |
|                 |                 |                 | els/configs/sam |
|                 |                 |                 | /pretrain/sam_v |
|                 |                 |                 | it_b_01ec64.pth |
+-----------------+-----------------+-----------------+-----------------+

Generator
---------

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –G_at           | int             | 10              |                 |
| tn_nb_mask_attn |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –G_att          | int             | 1               |                 |
| n_nb_mask_input |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –G_backward_com | flag            |                 | if true, feats  |
| patibility_twic |                 |                 | will go througt |
| e_resnet_blocks |                 |                 | resnet blocks   |
|                 |                 |                 | two times for   |
|                 |                 |                 | resnet_attn     |
|                 |                 |                 | generators.     |
|                 |                 |                 | This option     |
|                 |                 |                 | will be         |
|                 |                 |                 | deleted, it’s   |
|                 |                 |                 | for backward    |
|                 |                 |                 | compatibility   |
|                 |                 |                 | (old models     |
|                 |                 |                 | were trained    |
|                 |                 |                 | that way).      |
+-----------------+-----------------+-----------------+-----------------+
| –G_c            | string          | mode            | path to         |
| onfig_segformer |                 | ls/configs/segf | segformer       |
|                 |                 | ormer/segformer | configuration   |
|                 |                 | _config_b0.json | file for G      |
+-----------------+-----------------+-----------------+-----------------+
| –G_diff_        | int             | 1000            | Number of       |
| n_timestep_test |                 |                 | timesteps used  |
|                 |                 |                 | for UNET mha    |
|                 |                 |                 | inference (test |
|                 |                 |                 | time).          |
+-----------------+-----------------+-----------------+-----------------+
| –G_diff_n       | int             | 2000            | Number of       |
| _timestep_train |                 |                 | timesteps used  |
|                 |                 |                 | for UNET mha    |
|                 |                 |                 | training.       |
+-----------------+-----------------+-----------------+-----------------+
| –G_dropout      | flag            |                 | dropout for the |
|                 |                 |                 | generator       |
+-----------------+-----------------+-----------------+-----------------+
| –G_nblocks      | int             | 9               | # of layer      |
|                 |                 |                 | blocks in G,    |
|                 |                 |                 | applicable to   |
|                 |                 |                 | resnets         |
+-----------------+-----------------+-----------------+-----------------+
| –G_netE         | string          | resnet_256      | specify         |
|                 |                 |                 | multimodal      |
|                 |                 |                 | latent vector   |
|                 |                 |                 | encoder         |
|                 |                 |                 | **Values:*      |
|                 |                 |                 | *  *resnet_128, |
|                 |                 |                 | resnet_256,     |
|                 |                 |                 | resnet_512,     |
|                 |                 |                 | conv_128,       |
|                 |                 |                 | conv_256,       |
|                 |                 |                 | conv_512*       |
+-----------------+-----------------+-----------------+-----------------+
| –G_netG         | string          | mob             | specify         |
|                 |                 | ile_resnet_attn | generator       |
|                 |                 |                 | architecture    |
|                 |                 |                 | **Values:**     |
|                 |                 |                 | *resnet_9blocks,|
|                 |                 |                 | resnet_6blocks, |
|                 |                 |                 | resnet_3blocks, |
|                 |                 |                 | resnet_12blocks,|
|                 |                 |                 | mobile_         |
|                 |                 |                 | resnet_9blocks, |
|                 |                 |                 | mobile_         |
|                 |                 |                 | resnet_3blocks, |
|                 |                 |                 | resnet_attn,    |
|                 |                 |                 | mobi            |
|                 |                 |                 | le_resnet_attn, |
|                 |                 |                 | unet_256,       |
|                 |                 |                 | unet_128,       |
|                 |                 |                 | stylegan2,      |
|                 |                 |                 | smallstylegan2, |
|                 |                 |                 | segfo           |
|                 |                 |                 | rmer_attn_conv, |
|                 |                 |                 | segformer_conv, |
|                 |                 |                 | ittr, unet_mha, |
|                 |                 |                 | uvit*           |
+-----------------+-----------------+-----------------+-----------------+
| –G_ngf          | int             | 64              | # of gen        |
|                 |                 |                 | filters in the  |
|                 |                 |                 | last conv layer |
+-----------------+-----------------+-----------------+-----------------+
| –G_norm         | string          | instance        | instance        |
|                 |                 |                 | normalization   |
|                 |                 |                 | or batch        |
|                 |                 |                 | normalization   |
|                 |                 |                 | for             |
|                 |                 |                 | G\ **Values     |
|                 |                 |                 | :**\ *instance, |
|                 |                 |                 | batch, none*    |
+-----------------+-----------------+-----------------+-----------------+
| –G_padding_type | string          | reflect         | whether to use  |
|                 |                 |                 | padding in the  |
|                 |                 |                 | gen             |
|                 |                 |                 | erator\ **Value |
|                 |                 |                 | s:**\ *reflect, |
|                 |                 |                 | replicate,      |
|                 |                 |                 | zeros*          |
+-----------------+-----------------+-----------------+-----------------+
| –G_spectral     | flag            |                 | whether to use  |
|                 |                 |                 | spectral norm   |
|                 |                 |                 | in the          |
|                 |                 |                 | generator       |
+-----------------+-----------------+-----------------+-----------------+
| –G_stylegan2_n  | int             | 1               | Number of       |
| um_downsampling |                 |                 | downsampling    |
|                 |                 |                 | layers used by  |
|                 |                 |                 | Sty             |
|                 |                 |                 | leGAN2Generator |
+-----------------+-----------------+-----------------+-----------------+
| –G_un           | array           | [16]            | downrate        |
| et_mha_attn_res |                 |                 | samples at      |
|                 |                 |                 | which attention |
|                 |                 |                 | takes place     |
+-----------------+-----------------+-----------------+-----------------+
| –G_unet_mh      | array           | [1, 2, 4, 8]    | channel         |
| a_channel_mults |                 |                 | multiplier for  |
|                 |                 |                 | each level of   |
|                 |                 |                 | the UNET mha    |
+-----------------+-----------------+-----------------+-----------------+
| –G_unet_mha_    | int             | 32              |                 |
| group_norm_size |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –G_unet         | string          | groupnorm       | \ **Values:     |
| _mha_norm_layer |                 |                 | **\ *groupnorm, |
|                 |                 |                 | batchnorm,      |
|                 |                 |                 | layernorm,      |
|                 |                 |                 | instancenorm,   |
|                 |                 |                 | switchablenorm* |
+-----------------+-----------------+-----------------+-----------------+
| –G_unet_mha_nu  | int             | 32              |                 |
| m_head_channels |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –G_une          | int             | 1               |                 |
| t_mha_num_heads |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –G_unet         | array           | [2, 2, 2, 2]    | distribution of |
| _mha_res_blocks |                 |                 | resnet blocks   |
|                 |                 |                 | across the UNet |
|                 |                 |                 | stages, should  |
|                 |                 |                 | have same size  |
|                 |                 |                 | as              |
|                 |                 |                 | –G_unet_mh      |
|                 |                 |                 | a_channel_mults |
+-----------------+-----------------+-----------------+-----------------+
| –G_unet_mh      | flag            |                 | if true, use    |
| a_vit_efficient |                 |                 | efficient       |
|                 |                 |                 | attention in    |
|                 |                 |                 | UNet and UViT   |
+-----------------+-----------------+-----------------+-----------------+
| –G_uvit_num_tra | int             | 6               | Number of       |
| nsformer_blocks |                 |                 | transformer     |
|                 |                 |                 | blocks in UViT  |
+-----------------+-----------------+-----------------+-----------------+

Algorithm-specific
------------------

GAN model
~~~~~~~~~

=============== ===== ======= ==============================
Parameter       Type  Default Description
=============== ===== ======= ==============================
–alg_gan_lambda float 1.0     weight for GAN loss：GAN(G(X))
=============== ===== ======= ==============================

CUT model
~~~~~~~~~

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –alg            | float           | 1.0             |                 |
| _cut_HDCE_gamma |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –alg_cut        | float           | 1.0             |                 |
| _HDCE_gamma_min |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –               | flag            |                 | use MSENCE loss |
| alg_cut_MSE_idt |                 |                 | for identity    |
|                 |                 |                 | mapping:        |
|                 |                 |                 | MSE(G(Y), Y))   |
+-----------------+-----------------+-----------------+-----------------+
| –alg_cut_fl     | flag            |                 | Enforce         |
| ip_equivariance |                 |                 | fl              |
|                 |                 |                 | ip-equivariance |
|                 |                 |                 | as additional   |
|                 |                 |                 | regularization. |
|                 |                 |                 | It’s used by    |
|                 |                 |                 | FastCUT, but    |
|                 |                 |                 | not CUT         |
+-----------------+-----------------+-----------------+-----------------+
| –alg_cut        | float           | 1.0             | weight for MSE  |
| _lambda_MSE_idt |                 |                 | identity loss:  |
|                 |                 |                 | MSE(G(X), X)    |
+-----------------+-----------------+-----------------+-----------------+
| –alg            | float           | 1.0             | weight for NCE  |
| _cut_lambda_NCE |                 |                 | loss: NCE(G(X), |
|                 |                 |                 | X)              |
+-----------------+-----------------+-----------------+-----------------+
| –alg            | float           | 0.0             | weight for SRC  |
| _cut_lambda_SRC |                 |                 | (semantic       |
|                 |                 |                 | relation        |
|                 |                 |                 | consistency)    |
|                 |                 |                 | loss: NCE(G(X), |
|                 |                 |                 | X)              |
+-----------------+-----------------+-----------------+-----------------+
| –alg_cut_nce_T  | float           | 0.07            | temperature for |
|                 |                 |                 | NCE loss        |
+-----------------+-----------------+-----------------+-----------------+
| –               | flag            |                 | use NCE loss    |
| alg_cut_nce_idt |                 |                 | for identity    |
|                 |                 |                 | mapping:        |
|                 |                 |                 | NCE(G(Y), Y))   |
+-----------------+-----------------+-----------------+-----------------+
| –alg_           | flag            |                 | (used for       |
| cut_nce_include |                 |                 | single image    |
| s_all_negatives |                 |                 | translation) If |
| _from_minibatch |                 |                 | True, include   |
|                 |                 |                 | the negatives   |
|                 |                 |                 | from the other  |
|                 |                 |                 | samples of the  |
|                 |                 |                 | minibatch when  |
|                 |                 |                 | computing the   |
|                 |                 |                 | contrastive     |
|                 |                 |                 | loss. Please    |
|                 |                 |                 | see             |
|                 |                 |                 | mod             |
|                 |                 |                 | els/patchnce.py |
|                 |                 |                 | for more        |
|                 |                 |                 | details.        |
+-----------------+-----------------+-----------------+-----------------+
| –alg            | string          | 0,4,8,12,16     | compute NCE     |
| _cut_nce_layers |                 |                 | loss on which   |
|                 |                 |                 | layers          |
+-----------------+-----------------+-----------------+-----------------+
| –a              | string          | monce           | CUT contrastice |
| lg_cut_nce_loss |                 |                 | loss\ **Values  |
|                 |                 |                 | :**\ *patchnce, |
|                 |                 |                 | monce,          |
|                 |                 |                 | SRC_hDCE*       |
+-----------------+-----------------+-----------------+-----------------+
| –alg_cut_netF   | string          | mlp_sample      | how to          |
|                 |                 |                 | downsample the  |
|                 |                 |                 | feature         |
|                 |                 |                 | map\ **Valu     |
|                 |                 |                 | es:**\ *sample, |
|                 |                 |                 | mlp_sample,     |
|                 |                 |                 | sample_qsattn,  |
|                 |                 |                 | mlp             |
|                 |                 |                 | _sample_qsattn* |
+-----------------+-----------------+-----------------+-----------------+
| –alg_c          | flag            |                 | whether to use  |
| ut_netF_dropout |                 |                 | dropout with F  |
+-----------------+-----------------+-----------------+-----------------+
| –               | int             | 256             |                 |
| alg_cut_netF_nc |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –al             | string          | instance        | instance        |
| g_cut_netF_norm |                 |                 | normalization   |
|                 |                 |                 | or batch        |
|                 |                 |                 | normalization   |
|                 |                 |                 | for             |
|                 |                 |                 | F\ **Values     |
|                 |                 |                 | :**\ *instance, |
|                 |                 |                 | batch, none*    |
+-----------------+-----------------+-----------------+-----------------+
| –alg_           | int             | 256             | number of       |
| cut_num_patches |                 |                 | patches per     |
|                 |                 |                 | layer           |
+-----------------+-----------------+-----------------+-----------------+

CycleGAN model
~~~~~~~~~~~~~~

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –alg_cy         | float           | 10.0            | weight for      |
| clegan_lambda_A |                 |                 | cycle loss (A   |
|                 |                 |                 | -> B -> A)      |
+-----------------+-----------------+-----------------+-----------------+
| –alg_cy         | float           | 10.0            | weight for      |
| clegan_lambda_B |                 |                 | cycle loss (B   |
|                 |                 |                 | -> A -> B)      |
+-----------------+-----------------+-----------------+-----------------+
| –alg_cyclegan_  | float           | 0.5             | use identity    |
| lambda_identity |                 |                 | mapping.        |
|                 |                 |                 | Setting         |
|                 |                 |                 | lambda_identity |
|                 |                 |                 | other than 0    |
|                 |                 |                 | has an effect   |
|                 |                 |                 | of scaling the  |
|                 |                 |                 | weight of the   |
|                 |                 |                 | identity        |
|                 |                 |                 | mapping loss.   |
|                 |                 |                 | For example, if |
|                 |                 |                 | the weight of   |
|                 |                 |                 | the identity    |
|                 |                 |                 | loss should be  |
|                 |                 |                 | 10 times        |
|                 |                 |                 | smaller than    |
|                 |                 |                 | the weight of   |
|                 |                 |                 | the             |
|                 |                 |                 | reconstruction  |
|                 |                 |                 | loss, please    |
|                 |                 |                 | set             |
|                 |                 |                 | lambda_identity |
|                 |                 |                 | = 0.1           |
+-----------------+-----------------+-----------------+-----------------+
| –alg_cyc        | float           | 0.0             | whether to add  |
| legan_rec_noise |                 |                 | noise to        |
|                 |                 |                 | reconstruction  |
+-----------------+-----------------+-----------------+-----------------+

ReCUT / ReCycleGAN
~~~~~~~~~~~~~~~~~~

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –alg_re_P_lr    | float           | 0.0002          | initial         |
|                 |                 |                 | learning rate   |
|                 |                 |                 | for P networks  |
+-----------------+-----------------+-----------------+-----------------+
| –alg_re_adv     | flag            |                 | if True, also   |
| ersarial_loss_p |                 |                 | train the       |
|                 |                 |                 | prediction      |
|                 |                 |                 | model with an   |
|                 |                 |                 | adversarial     |
|                 |                 |                 | loss            |
+-----------------+-----------------+-----------------+-----------------+
| –alg_re_netP    | string          | unet_128        | specify P       |
|                 |                 |                 | architecture\   |
|                 |                 |                 |  **Values:**\ * |
|                 |                 |                 | resnet_9blocks, |
|                 |                 |                 | resnet_6blocks, |
|                 |                 |                 | resnet_attn,    |
|                 |                 |                 | unet_256,       |
|                 |                 |                 | unet_128*       |
+-----------------+-----------------+-----------------+-----------------+
| –alg_re_no_trai | flag            |                 | if True, P wont |
| n_P_fake_images |                 |                 | be trained over |
|                 |                 |                 | fake images     |
|                 |                 |                 | projections     |
+-----------------+-----------------+-----------------+-----------------+
| –alg            | int             | 3               | Number of       |
| _re_nuplet_size |                 |                 | frames loaded   |
+-----------------+-----------------+-----------------+-----------------+
| –alg_re_proje   | float           | 1.0             | threshold of    |
| ction_threshold |                 |                 | the real images |
|                 |                 |                 | projection loss |
|                 |                 |                 | below with fake |
|                 |                 |                 | projection and  |
|                 |                 |                 | fake            |
|                 |                 |                 | reconstruction  |
|                 |                 |                 | losses are      |
|                 |                 |                 | applied         |
+-----------------+-----------------+-----------------+-----------------+

Diffusion model
~~~~~~~~~~~~~~~

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –al             | array           | [‘canny’,       | what to use for |
| g_palette_compu |                 | ‘hed’]          | random sketch   |
| ted_sketch_list |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –alg_palette    | int             | 32              | nb of examples  |
| _cond_embed_dim |                 |                 | processed for   |
|                 |                 |                 | inference       |
+-----------------+-----------------+-----------------+-----------------+
| –a              | string          | y_t             | how cond_image  |
| lg_palette_cond |                 |                 | is              |
| _image_creation |                 |                 | created\ **V    |
|                 |                 |                 | alues:**\ *y_t, |
|                 |                 |                 | previous_frame, |
|                 |                 |                 | c               |
|                 |                 |                 | omputed_sketch* |
+-----------------+-----------------+-----------------+-----------------+
| –alg_palet      | string          |                 | whether to use  |
| te_conditioning |                 |                 | conditioning or |
|                 |                 |                 | not\            |
|                 |                 |                 | **Values:**\ *, |
|                 |                 |                 | mask, class,    |
|                 |                 |                 | mask_and_class* |
+-----------------+-----------------+-----------------+-----------------+
| –               | flag            |                 | whether to      |
| alg_palette_gen |                 |                 | generate        |
| erate_per_class |                 |                 | samples of each |
|                 |                 |                 | images          |
+-----------------+-----------------+-----------------+-----------------+
| –alg_palett     | int             | -1              | nb of examples  |
| e_inference_num |                 |                 | processed for   |
|                 |                 |                 | inference       |
+-----------------+-----------------+-----------------+-----------------+
| –alg_p          | float           | 1.0             | weight for      |
| alette_lambda_G |                 |                 | supervised loss |
+-----------------+-----------------+-----------------+-----------------+
| –a              | string          | MSE             | loss for        |
| lg_palette_loss |                 |                 | denoising       |
|                 |                 |                 | model\ **       |
|                 |                 |                 | Values:**\ *L1, |
|                 |                 |                 | MSE,            |
|                 |                 |                 | multiscale*     |
+-----------------+-----------------+-----------------+-----------------+
| –alg_p          | float           | 0.5             | prob to use     |
| alette_prob_use |                 |                 | previous frame  |
| _previous_frame |                 |                 | as y cond       |
+-----------------+-----------------+-----------------+-----------------+
| –alg_palette_   | string          | ddpm            | choose the      |
| sampling_method |                 |                 | sampling method |
|                 |                 |                 | between ddpm    |
|                 |                 |                 | and             |
|                 |                 |                 | ddim\ **Va      |
|                 |                 |                 | lues:**\ *ddpm, |
|                 |                 |                 | ddim*           |
+-----------------+-----------------+-----------------+-----------------+
| –               | array           | [0, 765]        | range for Canny |
| alg_palette_ske |                 |                 | thresholds      |
| tch_canny_range |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+

Datasets
--------

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –data_crop_size | int             | 256             | then crop to    |
|                 |                 |                 | this size       |
+-----------------+-----------------+-----------------+-----------------+
| –da             | string          | unaligned       | chooses how     |
| ta_dataset_mode |                 |                 | datasets are    |
|                 |                 |                 | loa             |
|                 |                 |                 | ded.\ **Values: |
|                 |                 |                 | **\ *unaligned, |
|                 |                 |                 | unalign         |
|                 |                 |                 | ed_labeled_cls, |
|                 |                 |                 | unaligne        |
|                 |                 |                 | d_labeled_mask, |
|                 |                 |                 | self_supervise  |
|                 |                 |                 | d_labeled_mask, |
|                 |                 |                 | unaligned_la    |
|                 |                 |                 | beled_mask_cls, |
|                 |                 |                 | sel             |
|                 |                 |                 | f_supervised_la |
|                 |                 |                 | beled_mask_cls, |
|                 |                 |                 | unaligned_label |
|                 |                 |                 | ed_mask_online, |
|                 |                 |                 | self_s          |
|                 |                 |                 | upervised_label |
|                 |                 |                 | ed_mask_online, |
|                 |                 |                 | unal            |
|                 |                 |                 | igned_labeled_m |
|                 |                 |                 | ask_cls_online, |
|                 |                 |                 | self_super      |
|                 |                 |                 | vised_labeled_m |
|                 |                 |                 | ask_cls_online, |
|                 |                 |                 | aligned,        |
|                 |                 |                 | nuplet_unaligne |
|                 |                 |                 | d_labeled_mask, |
|                 |                 |                 | temporal,       |
|                 |                 |                 | self_super      |
|                 |                 |                 | vised_temporal* |
+-----------------+-----------------+-----------------+-----------------+
| –data_direction | string          | AtoB            | AtoB or         |
|                 |                 |                 | BtoA\ **Va      |
|                 |                 |                 | lues:**\ *AtoB, |
|                 |                 |                 | BtoA*           |
+-----------------+-----------------+-----------------+-----------------+
| –dat            | flag            |                 | whether to      |
| a_inverted_mask |                 |                 | invert the      |
|                 |                 |                 | mask,           |
|                 |                 |                 | i.e. around the |
|                 |                 |                 | bbox            |
+-----------------+-----------------+-----------------+-----------------+
| –data_load_size | int             | 286             | scale images to |
|                 |                 |                 | this size       |
+-----------------+-----------------+-----------------+-----------------+
| –data_m         | int             | 1000000000      | Maximum number  |
| ax_dataset_size |                 |                 | of samples      |
|                 |                 |                 | allowed per     |
|                 |                 |                 | dataset. If the |
|                 |                 |                 | dataset         |
|                 |                 |                 | directory       |
|                 |                 |                 | contains more   |
|                 |                 |                 | than            |
|                 |                 |                 | ma              |
|                 |                 |                 | x_dataset_size, |
|                 |                 |                 | only a subset   |
|                 |                 |                 | is loaded.      |
+-----------------+-----------------+-----------------+-----------------+
| –d              | int             | 4               | # threads for   |
| ata_num_threads |                 |                 | loading data    |
+-----------------+-----------------+-----------------+-----------------+
| –data_online    | int             | 0               | context pixel   |
| _context_pixels |                 |                 | band around the |
|                 |                 |                 | crop, unused    |
|                 |                 |                 | for generation, |
|                 |                 |                 | only for disc   |
+-----------------+-----------------+-----------------+-----------------+
| –data_online_   | int             | -1              | if >0, it will  |
| fixed_mask_size |                 |                 | be used as      |
|                 |                 |                 | fixed bbox size |
|                 |                 |                 | (warning: in    |
|                 |                 |                 | dataset         |
|                 |                 |                 | resolution ie   |
|                 |                 |                 | before          |
|                 |                 |                 | resizing)       |
+-----------------+-----------------+-----------------+-----------------+
| –data_online_   | int             | -1              | category to     |
| select_category |                 |                 | select for      |
|                 |                 |                 | bounding boxes, |
|                 |                 |                 | -1 means all    |
|                 |                 |                 | boxes selected  |
+-----------------+-----------------+-----------------+-----------------+
| –data_onl       | flag            |                 | whether to only |
| ine_single_bbox |                 |                 | allow a single  |
|                 |                 |                 | bbox per online |
|                 |                 |                 | crop            |
+-----------------+-----------------+-----------------+-----------------+
| –               | string          | resize_and_crop | scaling and     |
| data_preprocess |                 |                 | cropping of     |
|                 |                 |                 | images at load  |
|                 |                 |                 | time\           |
|                 |                 |                 | **Values:**\ *r |
|                 |                 |                 | esize_and_crop, |
|                 |                 |                 | crop,           |
|                 |                 |                 | scale_width,    |
|                 |                 |                 | scale_          |
|                 |                 |                 | width_and_crop, |
|                 |                 |                 | none*           |
+-----------------+-----------------+-----------------+-----------------+
| –data           | flag            |                 | whether paths   |
| _relative_paths |                 |                 | to images are   |
|                 |                 |                 | relative to     |
|                 |                 |                 | dataroot        |
+-----------------+-----------------+-----------------+-----------------+
| –data           | flag            |                 | if true, wrong  |
| _sanitize_paths |                 |                 | images or       |
|                 |                 |                 | labels paths    |
|                 |                 |                 | will be removed |
|                 |                 |                 | before training |
+-----------------+-----------------+-----------------+-----------------+
| –data           | flag            |                 | if true, takes  |
| _serial_batches |                 |                 | images in order |
|                 |                 |                 | to make         |
|                 |                 |                 | batches,        |
|                 |                 |                 | otherwise takes |
|                 |                 |                 | them randomly   |
+-----------------+-----------------+-----------------+-----------------+

Online created datasets
~~~~~~~~~~~~~~~~~~~~~~~

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –dat            | flag            |                 | Perform task of |
| a_online_creati |                 |                 | replacing       |
| on_color_mask_A |                 |                 | color-filled    |
|                 |                 |                 | masks by        |
|                 |                 |                 | objects         |
+-----------------+-----------------+-----------------+-----------------+
| –dat            | int             | 50              | size of crops   |
| a_online_creati |                 |                 | are random,     |
| on_crop_delta_A |                 |                 | values allowed  |
|                 |                 |                 | are             |
|                 |                 |                 | online_cre      |
|                 |                 |                 | ation_crop_size |
|                 |                 |                 | more or less    |
|                 |                 |                 | online_crea     |
|                 |                 |                 | tion_crop_delta |
|                 |                 |                 | for domain A    |
+-----------------+-----------------+-----------------+-----------------+
| –dat            | int             | 50              | size of crops   |
| a_online_creati |                 |                 | are random,     |
| on_crop_delta_B |                 |                 | values allowed  |
|                 |                 |                 | are             |
|                 |                 |                 | online_cre      |
|                 |                 |                 | ation_crop_size |
|                 |                 |                 | more or less    |
|                 |                 |                 | online_crea     |
|                 |                 |                 | tion_crop_delta |
|                 |                 |                 | for domain B    |
+-----------------+-----------------+-----------------+-----------------+
| –da             | int             | 512             | crop to this    |
| ta_online_creat |                 |                 | size during     |
| ion_crop_size_A |                 |                 | online          |
|                 |                 |                 | creation, it    |
|                 |                 |                 | needs to be     |
|                 |                 |                 | greater than    |
|                 |                 |                 | bbox size for   |
|                 |                 |                 | domain A        |
+-----------------+-----------------+-----------------+-----------------+
| –da             | int             | 512             | crop to this    |
| ta_online_creat |                 |                 | size during     |
| ion_crop_size_B |                 |                 | online          |
|                 |                 |                 | creation, it    |
|                 |                 |                 | needs to be     |
|                 |                 |                 | greater than    |
|                 |                 |                 | bbox size for   |
|                 |                 |                 | domain B        |
+-----------------+-----------------+-----------------+-----------------+
| –da             | array           | []              | load to this    |
| ta_online_creat |                 |                 | size during     |
| ion_load_size_A |                 |                 | online          |
|                 |                 |                 | creation,       |
|                 |                 |                 | format : width  |
|                 |                 |                 | height or only  |
|                 |                 |                 | one size if     |
|                 |                 |                 | square          |
+-----------------+-----------------+-----------------+-----------------+
| –da             | array           | []              | load to this    |
| ta_online_creat |                 |                 | size during     |
| ion_load_size_B |                 |                 | online          |
|                 |                 |                 | creation,       |
|                 |                 |                 | format : width  |
|                 |                 |                 | height or only  |
|                 |                 |                 | one size if     |
|                 |                 |                 | square          |
+-----------------+-----------------+-----------------+-----------------+
| –dat            | array           | [0]             | ratio mask      |
| a_online_creati |                 |                 | offset to allow |
| on_mask_delta_A |                 |                 | generation of a |
|                 |                 |                 | bigger object   |
|                 |                 |                 | in domain B     |
|                 |                 |                 | (for semantic   |
|                 |                 |                 | loss) for       |
|                 |                 |                 | domain A,       |
|                 |                 |                 | format : width  |
|                 |                 |                 | (x) height (y)  |
|                 |                 |                 | or only one     |
|                 |                 |                 | size if square  |
+-----------------+-----------------+-----------------+-----------------+
| –dat            | array           | [0]             | mask offset to  |
| a_online_creati |                 |                 | allow           |
| on_mask_delta_B |                 |                 | genaration of a |
|                 |                 |                 | bigger object   |
|                 |                 |                 | in domain B     |
|                 |                 |                 | (for semantic   |
|                 |                 |                 | loss) for       |
|                 |                 |                 | domain B,       |
|                 |                 |                 | format : width  |
|                 |                 |                 | (y) height (x)  |
|                 |                 |                 | or only one     |
|                 |                 |                 | size if square  |
+-----------------+-----------------+-----------------+-----------------+
| –data_online    | array           | [0.0]           | ratio mask size |
| _creation_mask_ |                 |                 | randomization   |
| random_offset_A |                 |                 | (only to make   |
|                 |                 |                 | bigger one) to  |
|                 |                 |                 | robustify the   |
|                 |                 |                 | image           |
|                 |                 |                 | generation in   |
|                 |                 |                 | domain A,       |
|                 |                 |                 | format : width  |
|                 |                 |                 | (x) height (y)  |
|                 |                 |                 | or only one     |
|                 |                 |                 | size if square  |
+-----------------+-----------------+-----------------+-----------------+
| –data_online    | array           | [0.0]           | mask size       |
| _creation_mask_ |                 |                 | randomization   |
| random_offset_B |                 |                 | (only to make   |
|                 |                 |                 | bigger one) to  |
|                 |                 |                 | robustify the   |
|                 |                 |                 | image           |
|                 |                 |                 | generation in   |
|                 |                 |                 | domain B,       |
|                 |                 |                 | format : width  |
|                 |                 |                 | (y) height (x)  |
|                 |                 |                 | or only one     |
|                 |                 |                 | size if square  |
+-----------------+-----------------+-----------------+-----------------+
| –data           | flag            |                 | whether masks   |
| _online_creatio |                 |                 | should be       |
| n_mask_square_A |                 |                 | squared for     |
|                 |                 |                 | domain A        |
+-----------------+-----------------+-----------------+-----------------+
| –data           | flag            |                 | whether masks   |
| _online_creatio |                 |                 | should be       |
| n_mask_square_B |                 |                 | squared for     |
|                 |                 |                 | domain B        |
+-----------------+-----------------+-----------------+-----------------+
| –da             | flag            |                 | Perform task of |
| ta_online_creat |                 |                 | replacing       |
| ion_rand_mask_A |                 |                 | noised masks by |
|                 |                 |                 | objects         |
+-----------------+-----------------+-----------------+-----------------+

Semantic segmentation network
-----------------------------

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –f_s_all        | flag            |                 | if true, all    |
| _classes_as_one |                 |                 | classes will be |
|                 |                 |                 | considered as   |
|                 |                 |                 | the same one    |
|                 |                 |                 | (ie foreground  |
|                 |                 |                 | vs background)  |
+-----------------+-----------------+-----------------+-----------------+
| –f_             | array           | []              | class weights   |
| s_class_weights |                 |                 | for imbalanced  |
|                 |                 |                 | semantic        |
|                 |                 |                 | classes         |
+-----------------+-----------------+-----------------+-----------------+
| –f_s_c          | string          | mode            | path to         |
| onfig_segformer |                 | ls/configs/segf | segformer       |
|                 |                 | ormer/segformer | configuration   |
|                 |                 | _config_b0.json | file for f_s    |
+-----------------+-----------------+-----------------+-----------------+
| –f_s_dropout    | flag            |                 | dropout for the |
|                 |                 |                 | semantic        |
|                 |                 |                 | network         |
+-----------------+-----------------+-----------------+-----------------+
| –f_s_net        | string          | vgg             | specify f_s     |
|                 |                 |                 | network [vgg    |
+-----------------+-----------------+-----------------+-----------------+
| –f_s_nf         | int             | 64              | # of filters in |
|                 |                 |                 | the first conv  |
|                 |                 |                 | layer of        |
|                 |                 |                 | classifier      |
+-----------------+-----------------+-----------------+-----------------+
| –f_s_se         | int             | 2               | number of       |
| mantic_nclasses |                 |                 | classes of the  |
|                 |                 |                 | semantic loss   |
|                 |                 |                 | classifier      |
+-----------------+-----------------+-----------------+-----------------+
| –f_s_sem        | float           | 1.0             | threshold of    |
| antic_threshold |                 |                 | the semantic    |
|                 |                 |                 | classifier loss |
|                 |                 |                 | below with      |
|                 |                 |                 | semantic loss   |
|                 |                 |                 | is applied      |
+-----------------+-----------------+-----------------+-----------------+
| –f_s_weight_sam | string          |                 | path to sam     |
|                 |                 |                 | weight for f_s, |
|                 |                 |                 | e.g. mod        |
|                 |                 |                 | els/configs/sam |
|                 |                 |                 | /pretrain/sam_v |
|                 |                 |                 | it_b_01ec64.pth |
+-----------------+-----------------+-----------------+-----------------+
| –f_s_w          | string          |                 | path to         |
| eight_segformer |                 |                 | segformer       |
|                 |                 |                 | weight for f_s, |
|                 |                 |                 | e.g. models/co  |
|                 |                 |                 | nfigs/segformer |
|                 |                 |                 | /pretrain/segfo |
|                 |                 |                 | rmer_mit-b0.pth |
+-----------------+-----------------+-----------------+-----------------+

Semantic classification network
-------------------------------

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –cls_all        | flag            |                 | if true, all    |
| _classes_as_one |                 |                 | classes will be |
|                 |                 |                 | considered as   |
|                 |                 |                 | the same one    |
|                 |                 |                 | (ie foreground  |
|                 |                 |                 | vs background)  |
+-----------------+-----------------+-----------------+-----------------+
| –cl             | array           | []              | class weights   |
| s_class_weights |                 |                 | for imbalanced  |
|                 |                 |                 | semantic        |
|                 |                 |                 | classes         |
+-----------------+-----------------+-----------------+-----------------+
| –cls_c          | string          | mo              | path to         |
| onfig_segformer |                 | dels/configs/se | segformer       |
|                 |                 | gformer/segform | configuration   |
|                 |                 | er_config_b0.py | file for cls    |
+-----------------+-----------------+-----------------+-----------------+
| –cls_dropout    | flag            |                 | dropout for the |
|                 |                 |                 | semantic        |
|                 |                 |                 | network         |
+-----------------+-----------------+-----------------+-----------------+
| –cls_net        | string          | vgg             | specify cls     |
|                 |                 |                 | network [vgg    |
+-----------------+-----------------+-----------------+-----------------+
| –cls_nf         | int             | 64              | # of filters in |
|                 |                 |                 | the first conv  |
|                 |                 |                 | layer of        |
|                 |                 |                 | classifier      |
+-----------------+-----------------+-----------------+-----------------+
| –cls_se         | int             | 2               | number of       |
| mantic_nclasses |                 |                 | classes of the  |
|                 |                 |                 | semantic loss   |
|                 |                 |                 | classifier      |
+-----------------+-----------------+-----------------+-----------------+
| –cls_sem        | float           | 1.0             | threshold of    |
| antic_threshold |                 |                 | the semantic    |
|                 |                 |                 | classifier loss |
|                 |                 |                 | below with      |
|                 |                 |                 | semantic loss   |
|                 |                 |                 | is applied      |
+-----------------+-----------------+-----------------+-----------------+
| –cls_w          | string          |                 | path to         |
| eight_segformer |                 |                 | segformer       |
|                 |                 |                 | weight for cls, |
|                 |                 |                 | e.g. models/co  |
|                 |                 |                 | nfigs/segformer |
|                 |                 |                 | /pretrain/segfo |
|                 |                 |                 | rmer_mit-b0.pth |
+-----------------+-----------------+-----------------+-----------------+

Output
------

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –output_no_html | flag            |                 | do not save     |
|                 |                 |                 | intermediate    |
|                 |                 |                 | training        |
|                 |                 |                 | results to      |
|                 |                 |                 | [opt.ch         |
|                 |                 |                 | eckpoints_dir]/ |
|                 |                 |                 | [opt.name]/web/ |
+-----------------+-----------------+-----------------+-----------------+
| –ou             | int             | 100             | frequency of    |
| tput_print_freq |                 |                 | showing         |
|                 |                 |                 | training        |
|                 |                 |                 | results on      |
|                 |                 |                 | console         |
+-----------------+-----------------+-----------------+-----------------+
| –output_u       | int             | 1000            | frequency of    |
| pdate_html_freq |                 |                 | saving training |
|                 |                 |                 | results to html |
+-----------------+-----------------+-----------------+-----------------+
| –output_verbose | flag            |                 | if specified,   |
|                 |                 |                 | print more      |
|                 |                 |                 | debugging       |
|                 |                 |                 | information     |
+-----------------+-----------------+-----------------+-----------------+

Visdom display
~~~~~~~~~~~~~~

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –ou             | flag            |                 |                 |
| tput_display_G_ |                 |                 |                 |
| attention_masks |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –output_d       | int             | 53800           | aim port of the |
| isplay_aim_port |                 |                 | web display     |
+-----------------+-----------------+-----------------+-----------------+
| –output_dis     | string          | h               | aim server of   |
| play_aim_server |                 | ttp://localhost | the web display |
+-----------------+-----------------+-----------------+-----------------+
| –output_display | flag            |                 | if True x -     |
| _diff_fake_real |                 |                 | G(x) is         |
|                 |                 |                 | displayed       |
+-----------------+-----------------+-----------------+-----------------+
| –out            | string          |                 | visdom display  |
| put_display_env |                 |                 | environment     |
|                 |                 |                 | name (default   |
|                 |                 |                 | is “main”)      |
+-----------------+-----------------+-----------------+-----------------+
| –outp           | int             | 400             | frequency of    |
| ut_display_freq |                 |                 | showing         |
|                 |                 |                 | training        |
|                 |                 |                 | results on      |
|                 |                 |                 | screen          |
+-----------------+-----------------+-----------------+-----------------+
| –ou             | int             | 1               | window id of    |
| tput_display_id |                 |                 | the web display |
+-----------------+-----------------+-----------------+-----------------+
| –outpu          | int             | 0               | if positive,    |
| t_display_ncols |                 |                 | display all     |
|                 |                 |                 | images in a     |
|                 |                 |                 | single visdom   |
|                 |                 |                 | web panel with  |
|                 |                 |                 | certain number  |
|                 |                 |                 | of images per   |
|                 |                 |                 | row.(if == 0    |
|                 |                 |                 | ncols will be   |
|                 |                 |                 | computed        |
|                 |                 |                 | automatically)  |
+-----------------+-----------------+-----------------+-----------------+
| –output_d       | flag            |                 | Set True if you |
| isplay_networks |                 |                 | want to display |
|                 |                 |                 | networks on     |
|                 |                 |                 | port 8000       |
+-----------------+-----------------+-----------------+-----------------+
| –outp           | array           | [‘visdom’]      | output display, |
| ut_display_type |                 |                 | either visdom   |
|                 |                 |                 | or              |
|                 |                 |                 | aim\ **Valu     |
|                 |                 |                 | es:**\ *visdom, |
|                 |                 |                 | aim*            |
+-----------------+-----------------+-----------------+-----------------+
| –output_disp    | int             | 8097            | visdom port of  |
| lay_visdom_port |                 |                 | the web display |
+-----------------+-----------------+-----------------+-----------------+
| –output_displa  | string          | h               | visdom server   |
| y_visdom_server |                 | ttp://localhost | of the web      |
|                 |                 |                 | display         |
+-----------------+-----------------+-----------------+-----------------+
| –output_        | int             | 256             | display window  |
| display_winsize |                 |                 | size for both   |
|                 |                 |                 | visdom and HTML |
+-----------------+-----------------+-----------------+-----------------+

Model
-----

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –mode           | string          | DPT_Large       | specify depth   |
| l_depth_network |                 |                 | prediction      |
|                 |                 |                 | network         |
|                 |                 |                 | architec        |
|                 |                 |                 | ture\ **Values: |
|                 |                 |                 | **\ *DPT_Large, |
|                 |                 |                 | DPT_Hybrid,     |
|                 |                 |                 | MiDaS_small,    |
|                 |                 |                 | DPT_BEiT_L_512, |
|                 |                 |                 | DPT_BEiT_L_384, |
|                 |                 |                 | DPT_BEiT_B_384, |
|                 |                 |                 | DP              |
|                 |                 |                 | T_SwinV2_L_384, |
|                 |                 |                 | DP              |
|                 |                 |                 | T_SwinV2_B_384, |
|                 |                 |                 | DP              |
|                 |                 |                 | T_SwinV2_T_256, |
|                 |                 |                 | DPT_Swin_L_384, |
|                 |                 |                 | DPT_            |
|                 |                 |                 | Next_ViT_L_384, |
|                 |                 |                 | DPT_LeViT_224*  |
+-----------------+-----------------+-----------------+-----------------+
| –               | float           | 0.02            | scaling factor  |
| model_init_gain |                 |                 | for normal,     |
|                 |                 |                 | xavier and      |
|                 |                 |                 | orthogonal.     |
+-----------------+-----------------+-----------------+-----------------+
| –               | string          | normal          | network         |
| model_init_type |                 |                 | initial         |
|                 |                 |                 | ization\ **Valu |
|                 |                 |                 | es:**\ *normal, |
|                 |                 |                 | xavier,         |
|                 |                 |                 | kaiming,        |
|                 |                 |                 | orthogonal*     |
+-----------------+-----------------+-----------------+-----------------+
| –model_input_nc | int             | 3               | # of input      |
|                 |                 |                 | image channels: |
|                 |                 |                 | 3 for RGB and 1 |
|                 |                 |                 | for             |
|                 |                 |                 | grayscale\ *    |
|                 |                 |                 | *Values:**\ *1, |
|                 |                 |                 | 3*              |
+-----------------+-----------------+-----------------+-----------------+
| –m              | flag            |                 | multimodal      |
| odel_multimodal |                 |                 | model with      |
|                 |                 |                 | random latent   |
|                 |                 |                 | input vector    |
+-----------------+-----------------+-----------------+-----------------+
| –               | int             | 3               | # of output     |
| model_output_nc |                 |                 | image channels: |
|                 |                 |                 | 3 for RGB and 1 |
|                 |                 |                 | for             |
|                 |                 |                 | grayscale\ *    |
|                 |                 |                 | *Values:**\ *1, |
|                 |                 |                 | 3*              |
+-----------------+-----------------+-----------------+-----------------+

Training
--------

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –train_D        | int             | 1000            |                 |
| _accuracy_every |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –train_D_lr     | float           | 0.0002          | discriminator   |
|                 |                 |                 | separate        |
|                 |                 |                 | learning rate   |
+-----------------+-----------------+-----------------+-----------------+
| –train_G_ema    | flag            |                 | whether to      |
|                 |                 |                 | build G via     |
|                 |                 |                 | exponential     |
|                 |                 |                 | moving average  |
+-----------------+-----------------+-----------------+-----------------+
| –t              | float           | 0.999           | exponential     |
| rain_G_ema_beta |                 |                 | decay for ema   |
+-----------------+-----------------+-----------------+-----------------+
| –train_G_lr     | float           | 0.0002          | initial         |
|                 |                 |                 | learning rate   |
|                 |                 |                 | for generator   |
+-----------------+-----------------+-----------------+-----------------+
| –t              | int             | 1               | input batch     |
| rain_batch_size |                 |                 | size            |
+-----------------+-----------------+-----------------+-----------------+
| –train_beta1    | float           | 0.9             | momentum term   |
|                 |                 |                 | of adam         |
+-----------------+-----------------+-----------------+-----------------+
| –train_beta2    | float           | 0.999           | momentum term   |
|                 |                 |                 | of adam         |
+-----------------+-----------------+-----------------+-----------------+
| –train_cl       | flag            |                 | if true l1 loss |
| s_l1_regression |                 |                 | will be used to |
|                 |                 |                 | compute         |
|                 |                 |                 | regressor loss  |
+-----------------+-----------------+-----------------+-----------------+
| –train          | flag            |                 | if true cls     |
| _cls_regression |                 |                 | will be a       |
|                 |                 |                 | regressor and   |
|                 |                 |                 | not a           |
|                 |                 |                 | classifier      |
+-----------------+-----------------+-----------------+-----------------+
| –train_com      | flag            |                 |                 |
| pute_D_accuracy |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –train_         | flag            |                 |                 |
| compute_metrics |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –train_compu    | flag            |                 |                 |
| te_metrics_test |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –train_continue | flag            |                 | continue        |
|                 |                 |                 | training: load  |
|                 |                 |                 | the latest      |
|                 |                 |                 | model           |
+-----------------+-----------------+-----------------+-----------------+
| –train_epoch    | string          | latest          | which epoch to  |
|                 |                 |                 | load? set to    |
|                 |                 |                 | latest to use   |
|                 |                 |                 | latest cached   |
|                 |                 |                 | model           |
+-----------------+-----------------+-----------------+-----------------+
| –tr             | int             | 1               | the starting    |
| ain_epoch_count |                 |                 | epoch count, we |
|                 |                 |                 | save the model  |
|                 |                 |                 | by              |
|                 |                 |                 | <epoch_count>,  |
|                 |                 |                 | <ep             |
|                 |                 |                 | och_count>+<sav |
|                 |                 |                 | e_latest_freq>, |
|                 |                 |                 | …               |
+-----------------+-----------------+-----------------+-----------------+
| –t              | flag            |                 | whether to      |
| rain_export_jit |                 |                 | export model in |
|                 |                 |                 | jit format      |
+-----------------+-----------------+-----------------+-----------------+
| –train_gan_mode | string          | lsgan           | the type of GAN |
|                 |                 |                 | objective.      |
|                 |                 |                 | vanilla GAN     |
|                 |                 |                 | loss is the     |
|                 |                 |                 | cross-entropy   |
|                 |                 |                 | objective used  |
|                 |                 |                 | in the original |
|                 |                 |                 | GAN             |
|                 |                 |                 | paper.\ **Value |
|                 |                 |                 | s:**\ *vanilla, |
|                 |                 |                 | lsgan, wgangp,  |
|                 |                 |                 | projected*      |
+-----------------+-----------------+-----------------+-----------------+
| –               | int             | 1               | backward will   |
| train_iter_size |                 |                 | be apllied each |
|                 |                 |                 | iter_size       |
|                 |                 |                 | iterations, it  |
|                 |                 |                 | simulate a      |
|                 |                 |                 | greater batch   |
|                 |                 |                 | size : its      |
|                 |                 |                 | value is        |
|                 |                 |                 | batch           |
|                 |                 |                 | _size*iter_size |
+-----------------+-----------------+-----------------+-----------------+
| –               | int             | 0               | which iteration |
| train_load_iter |                 |                 | to load? if     |
|                 |                 |                 | load_iter > 0,  |
|                 |                 |                 | the code will   |
|                 |                 |                 | load models by  |
|                 |                 |                 | it              |
|                 |                 |                 | er_[load_iter]; |
|                 |                 |                 | otherwise, the  |
|                 |                 |                 | code will load  |
|                 |                 |                 | models by       |
|                 |                 |                 | [epoch]         |
+-----------------+-----------------+-----------------+-----------------+
| –train          | int             | 50              | multiply by a   |
| _lr_decay_iters |                 |                 | gamma every     |
|                 |                 |                 | lr_decay_iters  |
|                 |                 |                 | iterations      |
+-----------------+-----------------+-----------------+-----------------+
| –               | string          | linear          | learning rate   |
| train_lr_policy |                 |                 | policy.\ **Valu |
|                 |                 |                 | es:**\ *linear, |
|                 |                 |                 | step, plateau,  |
|                 |                 |                 | cosine*         |
+-----------------+-----------------+-----------------+-----------------+
| –trai           | int             | 1000            |                 |
| n_metrics_every |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –tr             | float           | 0.5             | weight for      |
| ain_mm_lambda_z |                 |                 | random z loss   |
+-----------------+-----------------+-----------------+-----------------+
| –train_mm_nz    | int             | 8               | number of       |
|                 |                 |                 | latent vectors  |
+-----------------+-----------------+-----------------+-----------------+
| –train_n_epochs | int             | 100             | number of       |
|                 |                 |                 | epochs with the |
|                 |                 |                 | initial         |
|                 |                 |                 | learning rate   |
+-----------------+-----------------+-----------------+-----------------+
| –train          | int             | 100             | number of       |
| _n_epochs_decay |                 |                 | epochs to       |
|                 |                 |                 | linearly decay  |
|                 |                 |                 | learning rate   |
|                 |                 |                 | to zero         |
+-----------------+-----------------+-----------------+-----------------+
| –train          | int             | 1000000000      | Maximum number  |
| _nb_img_max_fid |                 |                 | of samples      |
|                 |                 |                 | allowed per     |
|                 |                 |                 | dataset to      |
|                 |                 |                 | compute fid. If |
|                 |                 |                 | the dataset     |
|                 |                 |                 | directory       |
|                 |                 |                 | contains more   |
|                 |                 |                 | than            |
|                 |                 |                 | nb_img_max_fid, |
|                 |                 |                 | only a subset   |
|                 |                 |                 | is used.        |
+-----------------+-----------------+-----------------+-----------------+
| –train_optim    | string          | adam            | optimizer       |
|                 |                 |                 | (adam, radam,   |
|                 |                 |                 | adamw,          |
|                 |                 |                 | …)\ **Va        |
|                 |                 |                 | lues:**\ *adam, |
|                 |                 |                 | radam, adamw,   |
|                 |                 |                 | lion*           |
+-----------------+-----------------+-----------------+-----------------+
| –               | int             | 50              | the size of     |
| train_pool_size |                 |                 | image buffer    |
|                 |                 |                 | that stores     |
|                 |                 |                 | previously      |
|                 |                 |                 | generated       |
|                 |                 |                 | images          |
+-----------------+-----------------+-----------------+-----------------+
| –tra            | flag            |                 | whether saves   |
| in_save_by_iter |                 |                 | model by        |
|                 |                 |                 | iteration       |
+-----------------+-----------------+-----------------+-----------------+
| –train_         | int             | 1               | frequency of    |
| save_epoch_freq |                 |                 | saving          |
|                 |                 |                 | checkpoints at  |
|                 |                 |                 | the end of      |
|                 |                 |                 | epochs          |
+-----------------+-----------------+-----------------+-----------------+
| –train_s        | int             | 5000            | frequency of    |
| ave_latest_freq |                 |                 | saving the      |
|                 |                 |                 | latest results  |
+-----------------+-----------------+-----------------+-----------------+
| –tra            | flag            |                 | if true         |
| in_semantic_cls |                 |                 | semantic class  |
|                 |                 |                 | losses will be  |
|                 |                 |                 | used            |
+-----------------+-----------------+-----------------+-----------------+
| –trai           | flag            |                 | if true         |
| n_semantic_mask |                 |                 | semantic mask   |
|                 |                 |                 | losses will be  |
|                 |                 |                 | used            |
+-----------------+-----------------+-----------------+-----------------+
| –train_tem      | flag            |                 | if true, MSE    |
| poral_criterion |                 |                 | loss will be    |
|                 |                 |                 | computed        |
|                 |                 |                 | between         |
|                 |                 |                 | successive      |
|                 |                 |                 | frames          |
+-----------------+-----------------+-----------------+-----------------+
| –t              | float           | 1.0             | lambda for MSE  |
| rain_temporal_c |                 |                 | loss that will  |
| riterion_lambda |                 |                 | be computed     |
|                 |                 |                 | between         |
|                 |                 |                 | successive      |
|                 |                 |                 | frames          |
+-----------------+-----------------+-----------------+-----------------+
| –train_use_con  | flag            |                 |                 |
| trastive_loss_D |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+

Semantic training
~~~~~~~~~~~~~~~~~

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –               | flag            |                 | if true cls     |
| train_sem_cls_B |                 |                 | will be trained |
|                 |                 |                 | not only on     |
|                 |                 |                 | domain A but    |
|                 |                 |                 | also on domain  |
|                 |                 |                 | B               |
+-----------------+-----------------+-----------------+-----------------+
| –train          | float           | 1.0             | weight for      |
| _sem_cls_lambda |                 |                 | semantic class  |
|                 |                 |                 | loss            |
+-----------------+-----------------+-----------------+-----------------+
| –train_sem      | flag            |                 | whether to use  |
| _cls_pretrained |                 |                 | a pretrained    |
|                 |                 |                 | model,          |
|                 |                 |                 | available for   |
|                 |                 |                 | non “basic”     |
|                 |                 |                 | model only      |
+-----------------+-----------------+-----------------+-----------------+
| –train_s        | string          | basic           | class           |
| em_cls_template |                 |                 | ifier/regressor |
|                 |                 |                 | model type,     |
|                 |                 |                 | from            |
|                 |                 |                 | torchvision     |
|                 |                 |                 | (resnet18, …),  |
|                 |                 |                 | default is      |
|                 |                 |                 | custom simple   |
|                 |                 |                 | model           |
+-----------------+-----------------+-----------------+-----------------+
| –train_sem_idt  | flag            |                 | if true apply   |
|                 |                 |                 | semantic loss   |
|                 |                 |                 | on identity     |
+-----------------+-----------------+-----------------+-----------------+
| –t              | float           | 0.0002          | cls learning    |
| rain_sem_lr_cls |                 |                 | rate            |
+-----------------+-----------------+-----------------+-----------------+
| –t              | float           | 0.0002          | f_s learning    |
| rain_sem_lr_f_s |                 |                 | rate            |
+-----------------+-----------------+-----------------+-----------------+
| –train_         | float           | 1.0             | weight for      |
| sem_mask_lambda |                 |                 | semantic mask   |
|                 |                 |                 | loss            |
+-----------------+-----------------+-----------------+-----------------+
| –train          | flag            |                 | if true apply   |
| _sem_net_output |                 |                 | generator       |
|                 |                 |                 | semantic loss   |
|                 |                 |                 | on network      |
|                 |                 |                 | output for real |
|                 |                 |                 | image rather    |
|                 |                 |                 | than on label.  |
+-----------------+-----------------+-----------------+-----------------+
| –train_         | flag            |                 | if true domain  |
| sem_use_label_B |                 |                 | B has labels    |
|                 |                 |                 | too             |
+-----------------+-----------------+-----------------+-----------------+

Semantic training with masks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –train_mask_    | float           | 1e-06           | Charbonnier     |
| charbonnier_eps |                 |                 | loss epsilon    |
|                 |                 |                 | value           |
+-----------------+-----------------+-----------------+-----------------+
| –train_ma       | flag            |                 |                 |
| sk_compute_miou |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –train_ma       | flag            |                 | whether to use  |
| sk_disjoint_f_s |                 |                 | a disjoint f_s  |
|                 |                 |                 | with the same   |
|                 |                 |                 | exact structure |
+-----------------+-----------------+-----------------+-----------------+
| –t              | flag            |                 | if true f_s     |
| rain_mask_f_s_B |                 |                 | will be trained |
|                 |                 |                 | not only on     |
|                 |                 |                 | domain A but    |
|                 |                 |                 | also on domain  |
|                 |                 |                 | B               |
+-----------------+-----------------+-----------------+-----------------+
| –train_m        | flag            |                 | if true, object |
| ask_for_removal |                 |                 | removal mode,   |
|                 |                 |                 | domain B images |
|                 |                 |                 | with label 0,   |
|                 |                 |                 | cut models only |
+-----------------+-----------------+-----------------+-----------------+
| –train_mask_    | float           | 10.0            | weight for loss |
| lambda_out_mask |                 |                 | out mask        |
+-----------------+-----------------+-----------------+-----------------+
| –train_mas      | string          | L1              | loss for out    |
| k_loss_out_mask |                 |                 | mask content    |
|                 |                 |                 | (which should   |
|                 |                 |                 | not             |
|                 |                 |                 | change).\ **    |
|                 |                 |                 | Values:**\ *L1, |
|                 |                 |                 | MSE,            |
|                 |                 |                 | Charbonnier*    |
+-----------------+-----------------+-----------------+-----------------+
| –train_         | int             | 1000            |                 |
| mask_miou_every |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –train_mask     | flag            |                 | if true f_s     |
| _no_train_f_s_A |                 |                 | wont be trained |
|                 |                 |                 | on domain A     |
+-----------------+-----------------+-----------------+-----------------+
| –trai           | flag            |                 | use loss out    |
| n_mask_out_mask |                 |                 | mask            |
+-----------------+-----------------+-----------------+-----------------+

Data augmentation
-----------------

+-----------------+-----------------+-----------------+-----------------+
| Parameter       | Type            | Default         | Description     |
+=================+=================+=================+=================+
| –dataaug_APA    | flag            |                 | if true, G will |
|                 |                 |                 | be used as      |
|                 |                 |                 | augmentation    |
|                 |                 |                 | during D        |
|                 |                 |                 | training        |
|                 |                 |                 | adaptively to D |
|                 |                 |                 | overfitting     |
|                 |                 |                 | between real    |
|                 |                 |                 | and fake images |
+-----------------+-----------------+-----------------+-----------------+
| –da             | int             | 4               | How often to    |
| taaug_APA_every |                 |                 | perform APA     |
|                 |                 |                 | adjustment?     |
+-----------------+-----------------+-----------------+-----------------+
| –d              | int             | 50              | APA adjustment  |
| ataaug_APA_nimg |                 |                 | speed, measured |
|                 |                 |                 | in how many     |
|                 |                 |                 | images it takes |
|                 |                 |                 | for p to        |
|                 |                 |                 | in              |
|                 |                 |                 | crease/decrease |
|                 |                 |                 | by one unit.    |
+-----------------+-----------------+-----------------+-----------------+
| –dataaug_APA_p  | int             | 0               | initial value   |
|                 |                 |                 | of probability  |
|                 |                 |                 | APA             |
+-----------------+-----------------+-----------------+-----------------+
| –dat            | float           | 0.6             |                 |
| aaug_APA_target |                 |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| –data           | flag            |                 | whether to      |
| aug_D_diffusion |                 |                 | apply diffusion |
|                 |                 |                 | noise           |
|                 |                 |                 | augmentation to |
|                 |                 |                 | discriminator   |
|                 |                 |                 | inputs,         |
|                 |                 |                 | projected       |
|                 |                 |                 | discriminator   |
|                 |                 |                 | only            |
+-----------------+-----------------+-----------------+-----------------+
| –dataaug_D_     | int             | 4               | How often to    |
| diffusion_every |                 |                 | perform         |
|                 |                 |                 | diffusion       |
|                 |                 |                 | augmentation    |
|                 |                 |                 | adjustment      |
+-----------------+-----------------+-----------------+-----------------+
| –dataaug        | flag            |                 | whether to use  |
| _D_label_smooth |                 |                 | one-sided label |
|                 |                 |                 | smoothing with  |
|                 |                 |                 | discriminator   |
+-----------------+-----------------+-----------------+-----------------+
| –               | float           | 0.0             | whether to add  |
| dataaug_D_noise |                 |                 | instance noise  |
|                 |                 |                 | to              |
|                 |                 |                 | discriminator   |
|                 |                 |                 | inputs          |
+-----------------+-----------------+-----------------+-----------------+
| –dataaug_affine | float           | 0.0             | if specified,   |
|                 |                 |                 | apply random    |
|                 |                 |                 | affine          |
|                 |                 |                 | transforms to   |
|                 |                 |                 | the images for  |
|                 |                 |                 | data            |
|                 |                 |                 | augmentation    |
+-----------------+-----------------+-----------------+-----------------+
| –dataaug_a      | float           | 1.2             | if random       |
| ffine_scale_max |                 |                 | affine          |
|                 |                 |                 | specified, max  |
|                 |                 |                 | scale range     |
|                 |                 |                 | value           |
+-----------------+-----------------+-----------------+-----------------+
| –dataaug_a      | float           | 0.8             | if random       |
| ffine_scale_min |                 |                 | affine          |
|                 |                 |                 | specified, min  |
|                 |                 |                 | scale range     |
|                 |                 |                 | value           |
+-----------------+-----------------+-----------------+-----------------+
| –dataa          | int             | 45              | if random       |
| ug_affine_shear |                 |                 | affine          |
|                 |                 |                 | specified,      |
|                 |                 |                 | shear range     |
|                 |                 |                 | (0,value)       |
+-----------------+-----------------+-----------------+-----------------+
| –dataaug_a      | float           | 0.2             | if random       |
| ffine_translate |                 |                 | affine          |
|                 |                 |                 | specified,      |
|                 |                 |                 | translation     |
|                 |                 |                 | range           |
|                 |                 |                 | (-v             |
|                 |                 |                 | alue*img_size,+ |
|                 |                 |                 | value*img_size) |
|                 |                 |                 | value           |
+-----------------+-----------------+-----------------+-----------------+
| –dataaug_       | string          |                 | choose the      |
| diff_aug_policy |                 |                 | augmentation    |
|                 |                 |                 | policy : color  |
|                 |                 |                 | randaffine      |
|                 |                 |                 | r               |
|                 |                 |                 | andperspective. |
|                 |                 |                 | If you want     |
|                 |                 |                 | more than one,  |
|                 |                 |                 | please write    |
|                 |                 |                 | them separated  |
|                 |                 |                 | by a comma with |
|                 |                 |                 | no space        |
|                 |                 |                 | (e.g. co        |
|                 |                 |                 | lor,randaffine) |
+-----------------+-----------------+-----------------+-----------------+
| –dataaug        | float           | 0.5             | proba of using  |
| _diff_aug_proba |                 |                 | each            |
|                 |                 |                 | transformation  |
+-----------------+-----------------+-----------------+-----------------+
| –dataaug_imgaug | flag            |                 | whether to      |
|                 |                 |                 | apply random    |
|                 |                 |                 | image           |
|                 |                 |                 | augmentation    |
+-----------------+-----------------+-----------------+-----------------+
| –               | flag            |                 | if specified,   |
| dataaug_no_flip |                 |                 | do not flip the |
|                 |                 |                 | images for data |
|                 |                 |                 | augmentation    |
+-----------------+-----------------+-----------------+-----------------+
| –da             | flag            |                 | if specified,   |
| taaug_no_rotate |                 |                 | do not rotate   |
|                 |                 |                 | the images for  |
|                 |                 |                 | data            |
|                 |                 |                 | augmentation    |
+-----------------+-----------------+-----------------+-----------------+

JoliGEN Models
==============

Models
------

+------------+----------------------------------+
| Name       | Paper                            |
+============+==================================+
| CycleGAN   | https://arxiv.org/abs/1703.10593 |
+------------+----------------------------------+
| CyCADA     | https://arxiv.org/abs/1711.03213 |
+------------+----------------------------------+
| CUT        | https://arxiv.org/abs/2007.15651 |
+------------+----------------------------------+
| RecycleGAN | https://arxiv.org/abs/1808.05174 |
+------------+----------------------------------+
| StyleGAN2  | https://arxiv.org/abs/1912.04958 |
+------------+----------------------------------+

Generator architectures
-----------------------

+------------------------+----------------------+
| Architecture           | Number of parameters |
+========================+======================+
| Resnet 9 blocks        | 11.378M              |
+------------------------+----------------------+
| Mobile resnet 9 blocks | 1.987M               |
+------------------------+----------------------+
| Resnet attn            | 11.823M              |
+------------------------+----------------------+
| Mobile resnet attn     | 2.432M               |
+------------------------+----------------------+
| Segformer b0           | 4.158M               |
+------------------------+----------------------+
| Segformer attn b0      | 4.60M                |
+------------------------+----------------------+
| Segformer attn b1      | 14.724M              |
+------------------------+----------------------+
| Segformer attn b5      | 83.016M              |
+------------------------+----------------------+
| UNet with mha          | ~60M configurable    |
+------------------------+----------------------+
| ITTR                   | ~30M configurable    |
+------------------------+----------------------+
