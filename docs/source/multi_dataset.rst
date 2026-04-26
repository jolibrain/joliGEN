.. _multi-dataset:

######################
 Multi-Dataset Loading
######################

The multi-dataset loader trains one model from several compatible
datasets. It is intended for B2B video online training where the
network shape is fixed, but each source dataset may need its own online
crop parameters.

Use it with:

.. code-block:: bash

   --data_dataset_mode multi_dataset \
   --data_multi_dataset_config /path/to/multi_dataset_config.json

The wrapper samples one child dataset for every training item. Normal
PyTorch collation then builds mixed batches containing samples from
different datasets.

*****************
 Design overview
*****************

The multi-dataset wrapper keeps existing dataset classes isolated:

- the global options are deep-copied for every child dataset;
- per-dataset overrides are applied to the child copy only;
- the real child dataset is instantiated normally;
- training length is the sum of child dataset lengths;
- sampling is weighted random, with equal weights by default;
- returned training samples include ``dataset_name`` and
  ``dataset_index`` metadata;
- returned test samples also include ``dataset_test_name``.

The current implementation is scoped to
``self_supervised_vid_mask_online`` children. It does not add dataset
conditioning or class labels to the model.

Mixed batches require every child dataset to return the same dictionary
keys and tensor shapes. The wrapper validates one dry sample per child
at construction time and fails early if a child is incompatible.

********************
 Configuration file
********************

The multi-dataset config contains one entry per child dataset:

.. code-block:: json

   {
     "datasets": [
       {
         "name": "dataset_a",
         "dataset_mode": "self_supervised_vid_mask_online",
         "dataroot": "/path/to/dataset_a",
         "weight": 1.0,
         "overrides": {
           "data_online_creation_crop_size_A": 304,
           "data_online_creation_crop_delta_A": 30
         }
       },
       {
         "name": "dataset_b",
         "dataset_mode": "self_supervised_vid_mask_online",
         "dataroot": "/path/to/dataset_b",
         "weight": 2.0,
         "overrides": {
           "data_online_creation_load_size_A": [1280, 720]
         }
       }
     ]
   }

The ``weight`` controls the expected training composition. It is not an
exact per-epoch quota.

The supported per-child overrides are deliberately narrow:

- ``dataroot``;
- ``data_online_creation_crop_size_A``;
- ``data_online_creation_crop_delta_A``;
- ``data_online_creation_load_size_A``;
- ``data_online_creation_mask_delta_A``;
- ``data_online_creation_mask_delta_A_ratio``;
- ``data_online_creation_mask_random_offset_A``;
- ``data_online_creation_mask_square_A``;
- ``data_temporal_num_common_char``.

Shape-defining and model-defining options must remain global because the
B2B model is created once for the full run. Do not override options such
as ``data_load_size``, ``data_crop_size``, channel counts,
``data_temporal_number_frames``, ``data_temporal_frame_step``,
``G_netG``, or B2B model options per dataset.

****************
 Config generator
****************

``scripts/gen_multi_dataset_b2b_config.py`` builds the multi-dataset
JSON from a CSV or TSV manifest. Required columns are:

- ``name``;
- ``dataroot``.

Optional columns are:

- ``weight``;
- ``data_online_creation_crop_size_A``;
- ``data_online_creation_crop_delta_A``;
- ``data_online_creation_load_size_A``;
- ``data_online_creation_mask_delta_A``;
- ``data_online_creation_mask_delta_A_ratio``;
- ``data_online_creation_mask_random_offset_A``;
- ``data_online_creation_mask_square_A``;
- ``data_temporal_num_common_char``.

The generator writes:

- ``multi_dataset_config.json``;
- ``train_config.json`` configured with
  ``data_dataset_mode: "multi_dataset"``;
- preview grids for every child dataset, unless ``--skip-preview`` is
  used.

Example:

.. code-block:: bash

   python scripts/gen_multi_dataset_b2b_config.py \
     --manifest datasets.tsv \
     --output-dir runs/b2b_multi_dataset_configs \
     --data-load-size 256 \
     --data-crop-size 256 \
     --data-temporal-number-frames 2 \
     --data-temporal-frame-step 1

*****************
 Test set support
*****************

Multi-dataset test support is designed around per-dataset metrics. Each
child dataset should expose its own test set, and training metrics should
report separate values for each dataset/test split rather than one pooled
score.

The generator behavior is:

- if a child dataset already has ``testA`` or ``testA*`` directories,
  use those predefined test sets;
- if a child dataset has no predefined test set, generate a reproducible
  true holdout under the generator output directory;
- generated holdouts use absolute paths back to source images and masks,
  so source datasets are not modified;
- generated training paths exclude frames used by the sampled test
  windows.

For temporal video datasets, automatic test sampling must sample complete
valid temporal windows using the global ``data_temporal_number_frames``
and ``data_temporal_frame_step`` settings, plus any per-dataset
``data_temporal_num_common_char`` override.

The generated multi-dataset config contains a top-level ``test_sets``
list. Each entry maps the public test-set id used by ``test.py`` and
training metrics to one child dataset and one child test split:

.. code-block:: json

   {
     "test_sets": [
       {
         "id": "dataset_a",
         "dataset_name": "dataset_a",
         "dataroot": "/path/to/generated_or_existing_root",
         "child_test_name": "",
         "generated": true
       }
     ]
   }

During test, the wrapper receives one of these ids and instantiates only
the matching child dataset. Metrics and visualizations are therefore
reported per dataset/test split.

*************
 Limitations
*************

- V1 support is for B2B video online training with
  ``self_supervised_vid_mask_online`` children.
- Exact distributed composition is not guaranteed: DDP samples wrapper
  indices, while child selection happens inside each worker.
- Test-set generation is intended to make small dataset differences easy
  to handle, but all generated and predefined test sets must still return
  tensors compatible with the global model shape.
