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

  python3 train.py --dataroot /home/beniz/tmp/joligan/datasets/horse2zebra --checkpoints_dir /home/beniz/tmp/joligan/checkpoints --name horse2zebra --config_json examples/example_gan_horse2zebra.json

.. _training-im2im-with-class-semantics:

**********************************
 GAN training with class semantics
**********************************

Dataset: https://joligen.com/datasets/mnist2USPS.zip

.. code:: bash
	  
  python3 train.py --dataroot /path/to/mnist2USPS --checkpoints_dir
  /path/to/checkpoints --name mnist2USPS --config_json examples/example_gan_mnist2usps.json

.. _training-im2im-with-mask-semantics:

*********************************
 GAN Training with mask semantics
*********************************

Dataset: https://joligen.com/datasets/noglasses2glasses_ffhq.zip

.. code:: bash

   python3 train.py --dataroot /path/to/noglasses2glasses_ffhq/
   --checkpoints_dir /path/to/checkpoints/ --name noglasses2glasses
   --config_json examples/example_gan_noglasses2glasses.json

.. _training-im2im-with-bbox-semantics-and-online-sampling-boxes-dataaug:

************************************************
 GAN Training with online bounding box semantics
************************************************

Dataset: https://joligen.com/datasets/online_mario2sonic_full.tar

.. code:: bash

   python3 train.py --dataroot /path/to/online_mario2sonic/ --checkpoints_dir /path/to/checkpoints/ --name mario2sonic --config_json examples/example_gan_mario2sonic.json

.. _training-object-insertion:

***********************************
 DDPM training for object insertion
***********************************

Dataset: https://joligen.com/datasets/noglasses2glasses_ffhq.zip

Trains a diffusion model to insert glasses onto faces.

.. code:: bash

   python3 train.py --dataroot noglasses2glasses_ffhq --checkpoints_dir ./checkpoints --name glasses2noglasses --output_display_env glasses2noglasses --config_json examples/example_ddpm_glasses2noglasses.json
   


