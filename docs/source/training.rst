.. _training:

##################
 JoliGEN Training
##################

Training requires the following:

-  GPU

-  a ``checkpoints`` directory to be specified in which model weights
   are stored

-  a `Visdom <https://github.com/fossasia/visdom>`_ server, by default
   the training script starts a Visdom server on http://0.0.0.0:8097 if
   none is running

-  Go to http://localhost:8097 to follow training losses and image
   result samples

JoliGEN has (too) many options, for finer grained control, see the
:doc:`full option list <options>`.

.. _training-im2im-without-semantics:

*******************************************
 Training image to image without semantics
*******************************************

Modify as required and run with the following line command:

.. code:: bash

   python3 train.py --dataroot /path/to/horse2zebra --checkpoints_dir /path/to/checkpoints --name horse2zebra --output_display_env horse2zebra --data_load_size 256 --data_crop_size 256 --train_n_epochs 200 --data_dataset_mode unaligned --train_n_epochs_decay 0 --model_type cut --G_netG mobile_resnet_attn

.. _training-im2im-with-class-semantics:

*******************************
 Training with class semantics
*******************************

.. code:: bash

   python3 train.py --dataroot /path/to/mnist2USPS --checkpoints_dir /path/to/checkpoints --name mnist2USPS --output_display_env mnist2USPS --data_load_size 180 --data_crop_size 180 --train_n_epochs 200 --data_dataset_mode unaligned_labeled_cls --train_n_epochs_decay 0 --model_type cut --cls_semantic_nclasses 10 --train_sem_use_label_B --train_semantic_cls --dataaug_no_rotate --dataaug_D_noise 0.001 --G_netG mobile_resnet_attn

.. _training-im2im-with-mask-semantics:

******************************
 Training with mask semantics
******************************

.. code:: bash

   python3 train.py --dataroot /path/to/noglasses2glasses_ffhq/ --checkpoints_dir /path/to/checkpoints/ --name noglasses2glasses --output_display_env noglasses2glasses --output_display_freq 200 --output_print_freq 200 --train_G_lr 0.0002 --train_D_lr 0.0001 --train_sem_lr_f_s 0.0002 --data_crop_size 256 --data_load_size 256 --data_dataset_mode unaligned_labeled_mask --model_type cut --train_semantic_mask --train_batch_size 2 --train_iter_size 1 --model_input_nc 3 --model_output_nc 3 --f_s_net unet --train_mask_f_s_B --train_mask_out_mask --f_s_semantic_nclasses 2 --G_netG mobile_resnet_attn --alg_cut_nce_idt --train_sem_use_label_B --D_netDs projected_d basic vision_aided --D_proj_interp 256 --D_proj_network_type efficientnet --train_G_ema --G_padding_type reflect --dataaug_no_rotate --data_relative_paths

.. _training-im2im-with-bbox-semantics-and-online-sampling-boxes-dataaug:

********************************************************************************************
 Training with bounding box semantics and online sampling around boxes as data augmentation
********************************************************************************************

.. code:: bash

   python3 train.py --dataroot /path/to/online_mario2sonic/ --checkpoints_dir /path/to/checkpoints/ --name mario2sonic --output_display_env mario2sonic --output_display_freq 200 --output_print_freq 200 --train_G_lr 0.0002 --train_D_lr 0.0001 --train_sem_lr_f_s 0.0002 --data_crop_size 128 --data_load_size 180 --data_dataset_mode unaligned_labeled_mask_online --model_type cut --train_semantic_m --train_batch_size 2 --train_iter_size 1 --model_input_nc 3 --model_output_nc 3 --f_s_net unet --train_mask_f_s_B --train_mask_out_mask --data_online_creation_crop_size_A 128 --data_online_creation_crop_delta_A 50 --data_online_creation_mask_delta_A 50 --data_online_creation_crop_size_B 128 --data_online_creation_crop_delta_B 15 --data_online_creation_mask_delta_B 15 --f_s_semantic_nclasses 2 --G_netG segformer_attn_conv --G_config_segformer models/configs/segformer/segformer_config_b0.py --alg_cut_nce_idt --train_sem_use_label_B --D_netDs projected_d basic vision_aided --D_proj_interp 256 --D_proj_network_type vitsmall --train_G_ema --G_padding_type reflect --dataaug_no_rotate --data_relative_paths

.. _training-object-insertion:

***************************
 Training object insertion
***************************

Trains a diffusion model to insert glasses onto faces.

.. code:: bash

   python3 train.py --dataroot /path/to/noglasses2glasses_ffhq/ --checkpoints_dir /path/to/checkpoints/ --name noglasses2glasses --data_direction BtoA --output_display_env noglasses2glasses --gpu_ids 0,1 --model_type palette --train_batch_size 4 --train_iter_size 16 --model_input_nc 3 --model_output_nc 3 --data_relative_paths --train_G_ema --train_optim radam --data_dataset_mode self_supervised_labeled_mask --data_load_size 256 --data_crop_size 256 --G_netG unet_mha --data_online_creation_rand_mask_A --train_G_lr 0.00002 --train_n_epochs 400 --dataaug_no_rotate --output_display_freq 10000 --train_optim adamw --G_nblocks 2

.. _training-cyclegan:

******************
 Train a cycleGAN
******************

You can tune the hyperparameters in ``./scripts/train_cyclegan.sh`` and
then use the following line command.

.. code:: bash

   bash ./scripts/train_cyclegan.sh dataroot

.. _training-cyclegan-with-labels:

******************************
 Train a cycleGAN with labels
******************************

You can tune the hyperparameters in
``./scripts/train_cyclegan_semantic.sh`` and then use the following line
command.

.. code:: bash

   bash ./scripts/train_cyclegan_semantic.sh dataroot

.. _training-cyclegan-with-mask-labels:

***********************************
 Train a cycleGAN with mask labels
***********************************

You can tune the hyperparameters in
``./scripts/train_cyclegan_semantic_mask.sh`` and then use the following
line command.

.. code:: bash

   bash ./scripts/train_cyclegan_semantic_mask.sh dataroot

.. _training-visualize losses:

******************
 Visualize losses
******************

To display losses from previous training, please run

.. code:: bash

   python3 util/load_display_losses.py --loss_log_file_path path_to_repo_of_loss.json --port 8097 --env_name visdom_environment_name
