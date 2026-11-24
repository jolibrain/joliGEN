#################
 JoliGEN Metrics
#################

JoliGEN reads the model configuration from a generated ``train_config.json`` file that is stored in the model directory.
When testing a previously trained model, make sure the ``train_config.json`` file is in the directory.

.. code:: bash

   python3 test.py \
        --test_model_dir /path/to/model/directory \
        --test_epoch 1 \
        --test_metrics_list FID KID MSID PSNR LPIPS DINOv2 \
        --test_nb_img 1000 \
        --test_batch_size 16 \
        --test_seed 42

This will output the selected metrics:

.. code:: text

   fidB_test: 136.3628652179921
   msidB_test: 32.10317674393986
   kidB_test: 0.036237239837646484
   psnr_test: 20.68259048461914
   dinov2_test: 0.08321705460548401

The metrics are also saved in a ``/path/to/model/directory/metrics/date.json`` file:

.. code:: json

 {
    "fidB_test": 136.3628652179921,
    "msidB_test": 32.10317674393986,
    "kidB_test": 0.036237239837646484,
    "psnr_test": 20.68259048461914,
    "dinov2_test": 0.08321705460548401
 }

The following options are available:

- ``test_model_dir``: path to the checkpoints for the model, should contain a ``train_config.json`` file
- ``test_epoch``: which epoch to load, defaults to latest checkpoint
- ``test_metrics``: list of metrics to compute, defaults to all metrics
- ``test_nb_img``: number of images to generate to compute metrics, defaults to dataset size
- ``test_batch_size``: input batch size
- ``test_seed``: seed to use for reproducible results

Object-region temporal warping error
===================================

``scripts/eval_warping_error.py`` evaluates existing inference outputs
without loading a JoliGEN generation model. It requires consecutive RGB/RGBA
``000000_generated.png`` images and matching ``000000_orig_bbox.json`` files
in one directory. Annotations contain full-image, half-open ``[x0,y0,x1,y1]``
coordinates (or dictionaries using ``x0,y0,x1,y1`` or
``xmin,ymin,xmax,ymax``). All frames must have the same dimensions.
The annotated region can contain any object category; the evaluator uses no
class labels or object-specific model. Each frame has one bounding box defining
the region of interest (ROI).

Validate inputs and crop geometry first:

.. code:: bash

   python scripts/eval_warping_error.py \
       --input-dir /path/to/inference --output-dir /path/to/evaluation \
       --start-frame 0 --num-frames 101 --dry-run

Compute 100 consecutive transitions from those 101 frames:

.. code:: bash

   python scripts/eval_warping_error.py \
       --input-dir /path/to/inference --output-dir /path/to/evaluation \
       --start-frame 0 --num-frames 101 --model large --device cuda

Dependencies are torch, torchvision, numpy, and Pillow, already in the project
requirements. Pretrained torchvision RAFT weights may download on first use;
later runs can use the cached weights. ``--device cpu`` is supported.
``--self-test`` runs deterministic CPU checks without weights or input data.
``--dry-run`` validates files and writes ``config.json`` without loading RAFT.
Use a fresh output directory to preserve previous results: existing result
files with the same names are overwritten.

The evaluator uses one native-resolution crop for the selected sequence: the
union of its annotated boxes, expanded by 25% of the union width and height on
each side and clamped to the image. It pads right and bottom for RAFT, without
resizing, and removes padding from the resulting flows. Forward and backward
flow are estimated on the generated images.

The primary score is the mean, over valid pixels, of the sum of squared RGB
channel differences between a frame and its flow-warped successor, with RGB
values in [0,1]. Valid pixels must be inside the source box and warped target
box, in bounds, and forward/backward consistent. Consistency requires
``||f + warped(b)||^2 <= alpha * (||f||^2 + ||warped(b)||^2) + beta``;
``--fb-alpha`` defaults to 0.01 and ``--fb-beta`` to 0.5 pixel squared.

Outputs are ``per_transition.csv``, ``summary.csv``, and ``config.json``.
``mean_roi_warp_mse_rgb_sum`` averages valid transition scores equally;
transitions with no valid correspondences are invalid/NaN, never zero.
An entirely invalid sequence has a NaN primary mean. Inspect
``invalid_transitions`` and ``mean_roi_total_valid_coverage`` alongside error.
Configuration records crop geometry, thresholds, weights, and runtime versions.
``--save-diagnostics N`` optionally saves debug images for the first N transitions.

This is a custom bbox-based temporal metric. Rectangles can include background and other objects; consistency filtering is not ground-truth occlusion segmentation.
Compare matched frames, annotations, crop geometry, and RAFT settings. Lower
error alone does not establish realism or absence of flicker. The standalone
CLI does not change training metrics or ``--test_metrics_list``.
