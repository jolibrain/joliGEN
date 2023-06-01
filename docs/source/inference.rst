###################
 JoliGEN Inference
###################

**************
 Using python
**************

JoliGEN reads the model configuration from a generated
``train_config.json`` file that is stored in the model directory. When
loading a previously trained model, make sure the the
``train_config.json`` file is in the directory.

Python scripts are provided for inference, that can be used as a
baseline for using a model in another codebase.

Generate an image with a GAN generator model
============================================

.. code:: bash

   cd scripts
   python3 gen_single_image.py --model-in-file /path/to/model/latest_net_G_A.pth --img-in /path/to/source.jpg --img-out target.jpg

Generate an image with a diffusion model
========================================

Using a pretrained glasses insertion model (see above):

.. code:: bash

   python3 gen_single_image_diffusion.py --model-in-file /path/to/model/latest_net_G_A.pth --img-in /path/to/source.jpg --mask-in /path/to/mask.jpg --img-out target.jpg  --img-size 256

The mask image has 1 where to insert the object and 0 elsewhere.

**************
 Export model
**************

.. code:: bash

   python3 -m scripts.export_jit_model --model-in-file "/path/to/model_checkpoint.pth" --model-out-file exported_model.pt --model-type mobile_resnet_9blocks --img-size 360

Then ``exported_model.pt`` can be reloaded without JoliGEN to perform
inference with an external software, e.g. `DeepDetect
<https://github.com/jolibrain/deepdetect>`_ with torch backend.
