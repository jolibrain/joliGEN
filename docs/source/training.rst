.. _training:

##################
 JoliGEN Training
##################

JoliGEN allows training custom generative models with GANs and DDPMs.

Every training run consists running `train.py` with the appropriate
set of training and data processing options.

There are three ways of passing the options to the training procedure:
- Command line
- A JSON configuration file
- REST API of the JoliGEN server

Below, the recommended manner is through a JSON configuration file,
from the `examples/` directory.

Training involves:

- One or more Generator neural networks (GANs and DDPMs)
- One or more Discriminators (GANs only)
- One or more supervised neural networks acting as constraints, e.g. on labels (GANs only)
- One or more frozen neural networks for input data processing, e.g. segmentation (GANs and DDPMs)

Training requires the following:

- one or more GPUs
- a labeled/unlabled, paired/unpaired dataset of images
- a `checkpoints` directory to be specified in which model weights are stored
- Optional: a `Visdom <https://github.com/fossasia/visdom>`_ server. Go to http://localhost:8097 to follow training losses and image result samples

JoliGEN has (too) many options, for finer grained control, see the
:doc:`full option list <options>`.

.. _training-im2im-without-semantics:

*******************************
 GAN training without semantics
*******************************

Modify as required and run with the following line command:

Dataset: https://joligen.com/datasets/horse2zebra.zip

.. code:: bash

  python3 train.py --dataroot /path/to/horse2zebra --checkpoints_dir /path/to/checkpoints --name horse2zebra --config_json examples/example_gan_horse2zebra.json

.. _training-im2im-with-class-semantics:

**********************************
 GAN training with class semantics
**********************************

Dataset: https://joligen.com/datasets/mnist2USPS.zip

.. code:: bash
	  
  python3 train.py --dataroot /path/to/mnist2USPS --checkpoints_dir /path/to/checkpoints --name mnist2USPS --config_json examples/example_gan_mnist2USPS.json

.. _training-im2im-with-mask-semantics:

*********************************
 GAN Training with mask semantics
*********************************

Dataset: https://joligen.com/datasets/noglasses2glasses_ffhq.zip

.. code:: bash

   python3 train.py --dataroot /path/to/noglasses2glasses_ffhq/ --checkpoints_dir /path/to/checkpoints/ --name noglasses2glasses --config_json examples/example_gan_noglasses2glasses.json

.. _training-im2im-with-mask-semantics:

***************************************
 CUT_Turbo Training with mask semantics
***************************************

Trains a GAN model to insert glasses onto faces using a pretrained SD-Turbo model with LoRA adapter

Dataset: https://joligen.com/datasets/noglasses2glasses_ffhq.zip

.. code:: bash

   python3 train.py --dataroot /path/to/noglasses2glasses_ffhq/ --checkpoints_dir /path/to/checkpoints/ --name noglasses2glasses --config_json examples/example_cut_turbo_noglasses2glasses.json

.. _training-im2im-with-bbox-semantics-and-online-sampling-boxes-dataaug:

************************************************
 GAN Training with online bounding box semantics
************************************************

Dataset: https://joligen.com/datasets/online_mario2sonic_full.tar

.. code:: bash

   python3 train.py --dataroot /path/to/online_mario2sonic/ --checkpoints_dir /path/to/checkpoints/ --name mario2sonic --config_json examples/example_gan_mario2sonic.json

.. _training-object-insertion:

************************************************
 DDPM training for object insertion / inpainting
************************************************

Dataset: https://joligen.com/datasets/noglasses2glasses_ffhq.zip

Trains a diffusion model to insert glasses onto faces.

.. code:: bash

   python3 train.py --dataroot /path/to/data/noglasses2glasses_ffhq --checkpoints_dir /path/to/checkpoints --name noglasses2glasses --config_json examples/example_ddpm_noglasses2glasses.json

******************************************
 DDPM training with random bbox inpainting
******************************************

Dataset: https://joligen.com/datasets/xview_inpainting_flat256_full.zip (12Gb)

Trains a diffusion model to fill up random boxes from satellite imagery

.. code:: bash

   python3 train.py --dataroot /path/to/data/xview_inpainting_flat256_full --checkpoints_dir /path/to/checkpoints --name xview_inpaint --config_json examples/example_ddpm_xview.json

.. image:: _static/xview_inpainting_train1.png
   
**************************************
 DDPM training with class conditioning
**************************************

Dataset: https://joligen.com/datasets/online_mario2sonic_lite.zip

Trains a diffusion model to generate Marios conditioned by pose (standing, walking, jumping, swimming, crouching).

.. code:: bash

   python3 train.py --dataroot /path/to/data/online_mario2sonic_full --checkpoints_dir /path/to/checkpoints --name mario --config_json examples/example_ddpm_mario.json

*********************************************
 DDPM training with Canny sketch conditioning
*********************************************

Dataset: https://joligen.com/datasets/mapillary_full.zip (85 GB)

Trains a diffusion model to generate traffic signs conditioned by a Canny sketch.

.. code:: bash

   python3 train.py --dataroot /path/to/data/mapillary_full --checkpoints_dir /path/to/checkpoints --name mapillary --config_json examples/example_ddpm_mapillary.json

Open http://localhost:8097/env/mapillary (or alternatively http://<your-server-address>:8097 to have a look at your training logs: loss curves, model output and inputs, and the options used to train.

.. image:: _static/mapillary_visdom.png


************************************************
 DDPM training with image reference conditioning
************************************************

Dataset: https://joligen.com/datasets/viton_bbox_ref_mini.zip

Trains a diffusion model to generate tried on clothing items conditioned by a reference image.

.. code:: bash

   python3 train.py --dataroot /path/to/data/viton_bbox_ref_mini --checkpoints_dir /path/to/checkpoints --name viton --config_json examples/example_ddpm_unetref_viton.json

Open http://localhost:8097/env/viton to have a look at the training output: loss curves, model output and inputs, and the options used to train.

.. image:: _static/viton_ref_visdom.png

********************************
 DDPM training for pix2pix task
********************************

Can be used for style transfer or paired super-resolution.

Dataset: https://joligen.com/datasets/SEN2VEN_mini.zip

Trains a diffusion model to generate an image conditioned by another image (super-resolution in this example).

.. code:: bash

   python3 train.py --dataroot /path/to/data/SEN2VEN_mini --checkpoints_dir /path/to/checkpoints --name SEN2VEN --config_json examples/example_ddpm_SEN2VEN.json

*************************************************************
 Consistency Model training for object insertion / inpainting
*************************************************************

Dataset: https://joligen.com/datasets/noglasses2glasses_ffhq.zip

Trains a consistency model to insert glasses onto faces.

.. code:: bash

   python3 train.py --dataroot /path/to/data/noglasses2glasses_ffhq --checkpoints_dir /path/to/checkpoints --name noglasses2glasses --config_json examples/example_cm_noglasses2glasses.json

***************************************************
 DDPM training for video generation with inpainting
***************************************************

Dataset: https://joligen.com/datasets/online_mario2sonic_full.tar

Train a DDPM model to generate a sequence of frame images for inpainting, ensuring temporal consistency throughout the series of frames.

.. code:: bash

   python3 train.py --dataroot /path/to/data/online_mario2sonic_full  --checkpoints_dir /path/to/checkpoints  --name mario_vid  --config_json examples/example_ddpm_vid_mario.json

***********************************************************************
 Fine-tuning image-trained DDPM with motion module for video inpainting
***********************************************************************
Starts from a DDPM model pretrained on image data using a U-Net backbone (e.g., single-frame inpainting with Canny sketch conditioning). The pretrained U-Net weights are transferred into a new architecture, `unet_vid`, which extends the original U-Net with a motion-aware module.

While image-level DDPM models (such as those using `unet_mha`) process each frame independently, they often fail to maintain temporal coherence when applied to video sequences. To address this, we fine-tune the model on sequential video frames using the `unet_vid` architecture, which incorporates temporal modeling through motion modules.

This fine-tuning process results in a video-aware DDPM model capable of producing temporally consistent inpainting results across entire video sequences.

Dataset: https://joligen.com/datasets/online_mario2sonic_lite2.zip

.. code:: bash

   python3 train.py \
     --dataroot /path/to/data/online_mario2sonic_full \
     --checkpoints_dir /path/to/checkpoints \
     --name mario_vid_ft_vid_from_img \
     --config_json examples/example_ddpm_vid_mario_ft_from_image.json \
     --train_epoch specifie/starting/epoch/number/for/fine-tuning \
     --train_continue \

*************************************************************
 Fine-tuning DDPM video model on domain-similar video dataset
*************************************************************
Fine-tunes a pretrained video-based DDPM model on a target dataset from a related domain. This method leverages existing spatiotemporal representations to accelerate convergence and improve generalization, especially when labeled data is limited or there is a domain shift.

Dataset: https://joligen.com/datasets/online_mario2sonic_lite2.zip

.. code:: bash

   python3 train.py \
     --dataroot /path/to/data/online_mario2sonic_full \
     --checkpoints_dir /path/to/checkpoints \
     --name mario_vid_ft_vid \
     --config_json examples/example_ddpm_vid_mario_ft.json \
     --train_epoch specifie/starting/epoch/number/for/fine-tuning \
     --train_continue \

.. image:: _static/ddpm_vid_ft_ref_visdom.png

