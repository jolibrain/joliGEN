#################
 JoliGEN Metrics
#################

JoliGEN reads the model configuration from a generated ``train_config.json`` file that is stored in the model directory.
When testing a previously trained model, make sure the ``train_config.json`` file is in the directory.

.. code:: bash

   python3 test.py \
        --test_model_dir /path/to/model/directory \
        --test_epoch 1 \
        --test_metrics_list FID KID MSID PSNR \
        --test_nb_img 1000 \
        --test_batch_size 16 \
        --test_seed 42

This will output the selected metrics:

.. code:: text

   fidB_test: 136.3628652179921
   msidB_test: 32.10317674393986
   kidB_test: 0.036237239837646484
   psnr_test: 20.68259048461914

The metrics are also saved in a ``/path/to/model/directory/metrics/date.json`` file:

.. code:: json

 {
    "fidB_test": 136.3628652179921,
    "msidB_test": 32.10317674393986,
    "kidB_test": 0.036237239837646484,
    "psnr_test": 20.68259048461914
 }

The following options are available:

- ``test_model_dir``: path to the checkpoints for the model, should contain a ``train_config.json`` file
- ``test_epoch``: which epoch to load, defaults to latest checkpoint
- ``test_metrics``: list of metrics to compute, defaults to all metrics
- ``test_nb_img``: number of images to generate to compute metrics, defaults to dataset size
- ``test_batch_size``: input batch size
- ``test_seed``: seed to use for reproducible results
