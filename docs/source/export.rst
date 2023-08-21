##############
 Model export
##############

.. code:: bash

   python3 -m scripts.export_jit_model --model-in-file "/path/to/model_checkpoint.pth" --model-out-file exported_model.pt --model-type mobile_resnet_9blocks --img-size 360

Then ``exported_model.pt`` can be reloaded without JoliGEN to perform
inference with an external software, e.g. `DeepDetect
<https://github.com/jolibrain/deepdetect>`_ with torch backend.
