##############
 Model export
##############

.. code:: bash

   python3 -m scripts.export_jit_model --model_in_file "/path/to/model_checkpoint.pth" --model_out_file exported_model.pt --model_type mobile_resnet_9blocks --img_size 360

Then ``exported_model.pt`` can be reloaded without JoliGEN to perform
inference with an external software, e.g. `DeepDetect
<https://github.com/jolibrain/deepdetect>`_ with torch backend.
