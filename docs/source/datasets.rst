.. _datasets:

#################
 Dataset formats
#################

JoliGEN supports datasets with and without labels. Labeled datasets
are useful because they allow for more fine-grained control of
generated images.

Broadly speaking, labels do help constrain the search space of all
possible combinations of generated pixels. For this reason labels are
sometimes refered to as semantic constraints.

A dataset contains :

- One (``trainA``) or two (``trainA`` and ``trainB``) **training sets**;

- One (``testA``) or more (``testAxxx``, ``testAyyy``, ``testBxxx``,
  ``testByyy``) **test sets**. Multiple test sets are useful to keep
  track of various set of examples during training, e.g. corner-cases,
  various difficulties, etc...

For instance:

- **class labels** allow to ensure matching between input and output
  image, e.g. turn Mario into Sonic while keeping the action (jump,
  kneel, run, ..)
  
- **mask labels** allow conserve or modify only appropriate areas,
  e.g. generate a car and conserve everything around it
  
  JoliGEN allows to derive rectangular masks from **bounding boxes**,
  and more precise masks automatically with SAM.
  
.. _datasets-unlabeled:

*******************
 Unlabeled Datasets
*******************

Unlabeled dataset comes as a data folder with two subdirectories
``trainA`` and ``trainB`` that contain images from domain A and B
respectively.
Subdirectories ``testA`` and ``testB`` can be added for test data.

Example: horse to zebra from two sets of images

Dataset: https://joligen.com/datasets/horse2zebra.zip

.. code-block:: bash

		horse2zebra/
		horse2zebra/trainA  # horse images
		horse2zebra/trainB  # zebra images
		horse2zebra/testA
		horse2zebra/testB

.. _datasets-labels:

***************************
 Datasets with class labels
***************************

A class label is a label that hold for the full image.

Dataset with class label has ``trainA`` and ``trainB`` directories. In
``trainA``, every class comes as a separate directory that holds
images for this class.

Example: font number conversion

Dataset: https://joligen.com/datasets/mnist2USPS.zip

.. code-block:: bash
		
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

.. _datasets-bbox:

*****************************
 Datasets with bounding boxes
*****************************

Bounding boxes are elements location in format

.. code::

   cls xmin ymin xmax ymax

where ``cls`` is an integer for the class, starting from 1.

Dataset with bounding boxes comes as a data folder with two subdirectories
``trainA`` and ``trainB`` that contain two subdirectories ``imgs`` and
``bbox``. In ``imgs`` the image files are stored, and ``bbox``
contains a .txt file per image, that lists the boxes for that image.

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


.. _datasets-masks:

*********************
 Datasets with masks
*********************

Dataset with mask labels contain the subdirectories ``trainA`` and
``trainB``, each with two subdirectories ``imgs`` and ``bbox``. In
``imgs`` are the image files. In ``masks`` are the mask files. 
A mask file is a single channel (B&W) image with labels as pixel
values. For n different classes, pixel values in the mask have to be
between 0 and n-1. The number of classes needs to be specified at
training time with ``--f_s_semantic_nclasses n``.

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


*************************************************
 Datasets with bounding box and image-level class
*************************************************

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

*****************************************************
 Datasets with mask and reference image conditioning
*****************************************************

Example: inpaint a garment from a catalog image onto a person

Dataset:
https://www.joligen.com/datasets/viton_mask_ref_mini.zip

.. code::

   viton_mask_ref_mini
   viton_mask_ref_mini/trainA
   viton_mask_ref_mini/trainA/imgs
   viton_mask_ref_mini/trainA/imgs/00000_00.jpg # source image, e.g. person with original garment
   ...
   viton_mask_ref_mini/trainA/mask
   viton_mask_ref_mini/trainA/mask/00000_00.png # mask for inpainting zone, e.g. original garment to remove
   ...
   viton_mask_ref_mini/trainA/ref
   viton_mask_ref_mini/trainA/ref/00000_00.jpg # reference image to inpaint, e.g. catalog image
   ...
   viton_mask_ref_mini/trainA/paths.txt # list of associated source / mask image
   viton_mask_ref_mini/trainA/conditions.txt # list of associated source / reference image

``paths.txt`` format:

.. code::

   cat trainA/paths.txt
   trainA/imgs/00000_00.jpg trainA/mask/00000_00.png

in this order: ``source image path``, ``image mask``.

``conditions.txt`` format:

.. code::

   cat trainA/conditions.txt
   trainA/imgs/00000_00.jpg trainA/ref/00000_00.jpg

in this order: ``source image path`` (same as ``paths.txt``), ``reference image``.

*****************************************************
 Datasets with bbox and reference image conditioning
*****************************************************

Example: inpaint garments from a catalog image onto a person

Dataset:
https://www.joligen.com/datasets/viton_bbox_ref_mini.zip

.. code::

   viton_bbox_ref_mini
   viton_bbox_ref_mini/trainA
   viton_bbox_ref_mini/trainA/imgs
   viton_bbox_ref_mini/trainA/imgs/00000_00.jpg # source image, e.g. person with original garments
   ...
   viton_bbox_ref_mini/trainA/bbox
   viton_bbox_ref_mini/trainA/bbox/00000_00.txt # list of bboxes for inpainting zone, e.g. original garments to remove
   ...
   viton_bbox_ref_mini/trainA/cond
   viton_bbox_ref_mini/trainA/cond/00000_00.txt # list of reference images to inpaint for each bbox
   ...
   viton_bbox_ref_mini/trainA/ref
   viton_bbox_ref_mini/trainA/ref/00000_00.jpg # reference image to inpaint, e.g. catalog image
   ...
   viton_bbox_ref_mini/trainA/paths.txt # list of associated source / bboxes
   viton_bbox_ref_mini/trainA/conditions.txt # list of associated source / reference images

``paths.txt`` format:

.. code::

   cat trainA/paths.txt
   trainA/imgs/00000_00.jpg trainA/bbox/00000_00.txt

in this order: ``source image path``, ``bboxes file``.

Bounding box format is the :ref:`same as above<datasets-bbox>`.

``conditions.txt`` format:

.. code::

   cat trainA/conditions.txt
   trainA/imgs/00000_00.jpg trainA/cond/00000_00.txt

in this order: ``source image path`` (same as ``paths.txt``), ``file containing list of reference images``.

List of reference images file format:

.. code::

   cat trainA/cond/00000_00.txt
   trainA/ref/00000_00.jpg # path to reference image (same number of lines and order as corresponding bbox file)
