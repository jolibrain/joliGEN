.. _tips:

##############
 JoliGEN tips
##############

.. _tips-options:

**********
 Training
**********

-  JoliGEN has many options, start from examples in :ref:`training`, and
   modify them by adding/removing options.

-  Before initiating a long training run:

   -  set a small batch size, and increase it until GPU memory is
      filled.
   -  set `--output_display_freq` to a small value and verify visually
      on Visdom that inputs are correct.

-  Killing a training run:

   -  From the server: use the client to properly stop the job,
      :ref:`client_stop`.

   -  From the control line: hit Ctrl-C to kill the current processes.
      Beware: some processes can remain alive, this is a by-product of
      Python's multi-processing. In this case use:

      .. code:: bash

         ps aux | grep train.py

      to list all running jobs, and kill them one by one.

-  Learning rate: it is useful to start with the default learning rates.
   Though the following recipe appears to produce the best results:

   -  start from default learning rate

   -  once the model appears to have stabilized, i.e. quality does not
      improve (can take up to several weeks on some models!): stop the
      training run, lower the learning rate ten-fold and resume.

*********
 CPU/GPU
*********

.. _tips-cpu-gpu:

-  GANs and DDPMs are too intensive to train on CPU
-  Default GPU is 0, i.e. `--gpu_ids 0`
-  Set ``--gpu_ids -1`` to use CPU mode
-  Set ``--gpu_ids 0,1,2`` for multi-GPU mode. You need a large batch
   size (e.g., ``--train_batch_size 32``) to benefit from multiple GPUs.
-  Multiple GPUs: batch size is per GPU, i.e. batch size 64 on two GPUs
   yields an effective batch size of 128

.. _tips-visualization:

***************
 Visualization
***************

-  Use Visdom or AIM for visualization. Image generation is visual, and
   difficult to assess otherwise.

.. _tips-preprocessing:

***************
 Preprocessing
***************

.. _tips-finetune-resume-training:

*****************************
 Fine-tuning/resume training
*****************************

-  To fine-tune a pre-trained model, or resume the previous training,
   use the ``--train_continue`` flag. The program will then load the
   last checkpoint from the model and resume from the last epoch.

.. _tips-train-test-high-res-images:

***************************************
 Training/Testing with high res images
***************************************

JoliGEN is quite memory-intensive, most especially with GANs as it
trains multiple networks in parallel.

This can be an issue when training from large images.

-  Applying a model to large images: all models are fully convolutional,
   i.e. once trained in resolution XxY, they can apply seamlessly to
   higher resolutions. Notes, depending on applications:

      -  Results may be of lower quality

      -  It is a good practice to keep relevant elements in large images
         at the same resolutions as at training time, e.g. cars can be
         processed in larger images correctly as long as they are made
         of the same approximate number of pixels as in training images.

-  Online cropping with the `--data_online` options and `online`
   dataloaders automatically focuses the bulk of the work on relevant
   locations (based on labels). This prevents costly pre-processing
   steps, especially useful on large datasets, while allowing to easily
   experiment with various input parameters.

-  Progressive finetuning of increasing resolutions works very well:
   start training in 256x256, then resume training at 360x360 and
   512x512 once stabilized. This lowers the computing training costs by
   speeding up training.

   -  GANs: when using a `projected_d` discriminator, anticipate the
      final full resolution by setting up `--D_proj_interp` to its size,
      e.g. 512 while initially training at 256x256.

-  Preprocessing large images does speed up dataloaders: when processing
   at 512x512 from 4k images, lowering the dataset resolution once
   before training saves lots of compute, preventing the dataloaders to
   resize every image in an online manner, many times.

.. _tips-loss-curve:

******************
 About loss curve
******************

Unfortunately, the loss curve does not reveal much information in
training GANs, and joliGEN is no exception. To check whether the
training has converged or not, we recommend periodically generating a
few samples and looking at them.

For DDPMs, the loss is descending and easy to assess. In practice, the
loss noisily reaches the 1e-4 zone while the image output keeps
improving.

.. _tips-batch-size:

******************
 About batch size
******************

JoliGEN is designed to accommodate consumer-grade GPUs with as low as a
few GB of VRAM.

-  When the batch size does not fit the GPU memory, lower it and use
   `--iter_size` to accumulate the gradients over multiple passes. This
   is equivalent to larger batch size.

-  Large batch sizes do stabilize the losses, however do not appear to
   improve results significantly. Running with batch sizes as low as 2
   or 4, and `iter_size` around 4 to 8 appears to be enough in practice.
