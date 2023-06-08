**************************************************************
 Quickstart GAN: Train a model that removes glasses from faces
**************************************************************

.. _quickstart-gan-dataset:

This section builds a model that removes glasses from faces. Once trained,
the model is given a portrait image of someone wearing glasses, and
produces a portrait without glasses.

Download the Dataset
====================

Download the dataset (3.3 Gb), unzip it, place it in the ``datasets`` directory, and create your ``checkpoints`` directory:

.. code:: bash

   wget https://www.joligen.com/datasets/noglasses2glasses_ffhq.zip
   unzip noglasses2glasses_ffhq.zip
   mkdir datasets
   mv noglasses2glasses_ffhq datasets/noglasses2glasses_ffhq
   rm noglasses2glasses_ffhq.zip
   mkdir checkpoints

This dataset contains two subdirectories with portraits: one with
glasses, one without glasses. Note that of course every directory
contains different glasses. This is an *unpaired* dataset.

For every face in the dataset, there's a corresponding mask location
(bounding boxes) that contains either the eyes or the glasses location.

Train your GAN
==============

We train a GAN with JoliGEN `cut` model (
https://github.com/https://arxiv.org/abs/2007.15651), with
`semantic_mask` constraints. This means that the training algorithm
uses the available masks (around eyes and glasses) to help the GAN
modify only the relevant elements from the input portrait. Note that
this mask is not required at inference, when using the trained model!

Use ``train.py`` along with the `example JoliGEN config file
<https://github.com/jolibrain/joliGEN/examples/example_gan_glasses2noglasses.json>`_
to launch the training:

.. code:: bash

   python3 train.py --dataroot noglasses2glasses_ffhq --checkpoints_dir ./checkpoints --name glasses2noglasses --output_display_env glasses2noglasses --config_json examples/example_glasses2noglasses.json

The training run can be monitored from the terminal. Lines
like the ones below are printed every x iterations (according to the
``--output_print_freq`` option, which is set to 200 iterations in this
example, we recommend setting its value to at least
``train_batch_size * train_iter_size`` to produce smooth curves):

.. code::

   (epoch: 1, iters: 2800, time comput per image: 0.114, time data mini batch: 0.002)
   G_tot_avg: 6.772019 G_NCE_avg: 0.410732 G_NCE_Y_avg: 0.411012
   G_GAN_D_B_projected_d_avg: 2.265606 G_GAN_D_B_basic_avg: 0.465516
   G_GAN_D_B_vision_aided_avg: 1.158908 D_tot_avg: 1.014235
   D_GAN_D_B_projected_d_avg: 0.212351 D_GAN_D_B_basic_avg: 0.394115
   D_GAN_D_B_vision_aided_avg: 0.407769 G_sem_mask_AB_avg: 0.525168
   G_out_mask_AB_avg: 1.945949 f_s_avg: 1.039919

Alternatively, you can :ref:`monitor your training
<quickstart-visdom-gan>` through a local web page to which training logs
are sent.

The GAN converges in around 20 hours on a single RTX A5000 after
training for ~24 epochs, batch size of 16, iter size 16, equivalent to
a full batch size of 256.

.. _quickstart-visdom-gan:

GAN Training Visualization
====================

Open http://localhost:8097/env/glasses2noglasses (or alternatively
``http://<your-server-address>:8097`` to have a look at your training
logs: loss curves, model output and inputs, and the options used to
train.

.. image:: _static/quickstart_visdom_gan.png
	   
