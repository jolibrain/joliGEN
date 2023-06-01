.. _datasets:

#################
 Dataset formats
#################

.. _datasets-unlabeled:

*******************
 Unlabeled Datasets
*******************

To train a model on your own datasets, you need to create a data folder
with two subdirectories ``trainA`` and ``trainB`` that contain images
from domain A and B. You can test your model on your training set by
setting ``--phase train`` in ``test.py``. You can also create
subdirectories ``testA``` and ``testB`` if you have test data.

.. _datasets-labels:

**********************
 Datasets with labels
**********************

Create ``trainA`` and ``trainB`` directories as described for CycleGAN
datasets. In ``trainA``, you have to separate your data into
directories, each directory belongs to a class.

.. _datasets-masks:

*********************
 Datasets with masks
*********************

You can use a dataset made of images and their mask labels (it can be
segmentation or attention masks). To do so, you have to generate masks
which are pixel labels for the images. If you have n different classes,
pixel values in the mask have to be between 0 and n-1. You can specify
the number of classes with the flag ``--semantic_nclasses n``.

.. _datasets-example:

******************
 Example Datasets
******************

.. _datasets-example-im2im-without-semantics:

Image to image without semantics
================================

Example: horse to zebra from two sets of images

Dataset: https://joligen.com/datasets/horse2zebra.zip

.. code::

   horse2zebra/
   horse2zebra/trainA  # horse images
   horse2zebra/trainB  # zebra images
   horse2zebra/testA
   horse2zebra/testB

.. _datasets-example-im2im-with-class-semantics:

Image to image with class semantics
===================================

Example: font number conversion

Dataset: https://joligen.com/datasets/mnist2USPS.zip

.. code::

   mnist2USPS/
   mnist2USPS/trainA
   mnist2USPS/trainA/0  # images of number 0
   mnist2USPS/trainA/1  # images of number 1
   mnist2USPS/trainA/2  # images of number 2
   ...
   mnist2USPS/trainB
   mnist2USPS/trainB/0  # images of target number 0
   mnist2USPS/trainB/1  # images of target number 1
   mnist2USPS/trainB/2  # images of target number 2

.. _datasets-example-im2im-with-mask-semantics:

Image to image with mask semantics
==================================

Example: Add glasses to a face without modifying the rest of the face

Dataset:
https://joligen.com/datasets/noglasses2glasses_ffhq_mini.zip

Full dataset:
https://joligen.com/datasets/noglasses2glasses_ffhq.zip

.. code::

   noglasses2glasses_ffhq_mini
   noglasses2glasses_ffhq_mini/trainA
   noglasses2glasses_ffhq_mini/trainA/img
   noglasses2glasses_ffhq_mini/trainA/img/0000.png # source image, e.g. face without glasses
   ...
   noglasses2glasses_ffhq_mini/trainA/bbox
   noglasses2glasses_ffhq_mini/trainA/bbox/0000.png # source mask, e.g. mask around eyes
   ...
   noglasses2glasses_ffhq_mini/trainA/paths.txt # list of associated source / mask images
   noglasses2glasses_ffhq_mini/trainB
   noglasses2glasses_ffhq_mini/trainB/img
   noglasses2glasses_ffhq_mini/trainB/img/0000.png # target image, e.g. face with glasses
   ...
   noglasses2glasses_ffhq_mini/trainB/bbox
   noglasses2glasses_ffhq_mini/trainB/bbox/0000.png # target mask, e.g. mask around glasses
   ...
   noglasses2glasses_ffhq_mini/trainB/paths.txt # list of associated target / mask images

.. _datasets-example-im2im-with-bbox-semantics:

Image to image with bounding box semantics
==========================================

Example: Super Mario to Sonic while preserving the position and action,
e.g. crouch, jump, still, ...

Dataset:
https://joligen.com/datasets/online_mario2sonic_lite.zip

Full dataset:
https://joligen.com/datasets/online_mario2sonic_full.tar

.. code::

   online_mario2sonic_lite
   online_mario2sonic_lite/mario
   online_mario2sonic_lite/mario/bbox
   online_mario2sonic_lite/mario/bbox/r_mario_frame_19538.jpg.txt # contains bboxes, see format below
   online_mario2sonic_lite/mario/imgs
   online_mario2sonic_lite/mario/imgs/mario_frame_19538.jpg
   online_mario2sonic_lite/mario/all.txt # list of associated source image / bbox file,
   ...
   online_mario2sonic_lite/sonic
   online_mario2sonic_lite/sonic/bbox
   online_mario2sonic_lite/sonic/bbox/r_sonic_frame_81118.jpg.txt
   online_mario2sonic_lite/sonic/imgs
   online_mario2sonic_lite/sonic/imgs/sonic_frame_81118.jpg
   online_mario2sonic_lite/sonic/all.txt # list of associated target image / bbox file
   ...
   online_mario2sonic_lite/trainA
   online_mario2sonic_lite/trainA/paths.txt # symlink to ../mario/all.txt
   online_mario2sonic_lite/trainB
   online_mario2sonic_lite/trainB/paths.txt # symlink to ../sonic/all.txt

List file format:

.. code::

   cat online_mario2sonic_lite/mario/all.txt
   mario/imgs/mario_frame_19538.jpg mario/bbox/r_mario_frame_19538.jpg.txt

Bounding boxes format, e.g. ``r_mario_frame_19538.jpg.txt``:

.. code::

   2 132 167 158 218

in this order:

.. code::

   cls xmin ymin xmax ymax

where ``cls`` is the class, in this dataset ``2`` means ``running``.

.. _datasets-example-im2im-with-bbox-class-semantics:

Image to image with multiple semantics: bounding box and class
==============================================================

Example: Image seasonal modification while preserving objects with mask
(cars, pedestrians, ...) and overall image weather (snow, rain, clear,
...) with class

Dataset:
https://joligen.com/datasets/daytime2dawn_dusk_lite.zip

.. code::

   daytime2dawn_dusk_lite
   daytime2dawn_dusk_lite/dawn_dusk
   daytime2dawn_dusk_lite/dawn_dusk/img
   daytime2dawn_dusk_lite/dawn_dusk/mask
   daytime2dawn_dusk_lite/daytime
   daytime2dawn_dusk_lite/daytime/img
   daytime2dawn_dusk_lite/daytime/mask
   daytime2dawn_dusk_lite/trainA
   daytime2dawn_dusk_lite/trainA/paths.txt
   daytime2dawn_dusk_lite/trainB
   daytime2dawn_dusk_lite/trainB/paths.txt

``paths.txt`` format:

.. code::

   cat trainA/paths.txt
   daytime/img/00054602-3bf57337.jpg 2 daytime/mask/00054602-3bf57337.png

in this order: ``source image path``, ``image class``, ``image mask``,
where ``image class`` in this dataset represents the weather class.

.. _datasets-example-im2im-with-other-semantics:

Other semantics
===============

Other semantics are possible, i.e. an algorithm that runs on both source
and target.
