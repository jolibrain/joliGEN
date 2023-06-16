#############
 Dataloaders
#############

Every dataset type requires a dedicated dataloader, for the classes,
masks or bounding boxes to be processed accordingly.

There are two types of datasets and thus dataloaders:

-  **Unaligned datasets**: images from domain A and B do not come as
   pairs
-  **Aligned datasets**: images from domain A and B are paired

Next, there's a special type of dataloaders for online pre-processing
and augmentation:

-  **Online**: dataloaders that automatically crop and zoom in / zoom
   out around various elements of images, most often according to masks
   and bounding boxes.

Online dataloaders are one of the great features of JoliGEN: they allow
to avoid a long pre-processing of the datasets, e.g. resizing images,
generating crops of interest, etc...

There's a special type of dataloaders for sequential data:

-  **Temporal**: dataloaders that load sequences of images, used for
   temporal discriminators in GANs, and temporal conditioning in DDPMs.

Finally, there's a special type of self-supervised dataloader, used by
DDPMs only:

-  **self-supervised**: dataloaders that modify the input data in order
   to generate self-supervised tasks, e.g. by removing portions of an
   image for training an inpainting DDPM.

To choose a dataloader please use the flag ``--dataset_mode
dataloader_name``.

*********************
 List of dataloaders
*********************

-  unaligned: basic unaligned, e.g. horse2zebra dataset
-  unaligned_labeled_cls: unaligned with classes
-  unaligned_labeled_mask: unaligned with masks
-  unaligned_labeled_mask_online: unaligned with masks with online
   croping around masks
-  unaligned_labeled_mask_cls_online: unaligned with masks and classes
   with online croping around masks
-  self_supervised_labeled_cls: with class labels
-  self_supervised_labeled_mask: with mask labels
-  self_supervised_labeled_mask_online: with mask labels and online
   croping around masks
-  self_supervised_labeled_mask_cls_online: with class and mask labels,
   and online croping around masks
-  temporal: basic temporal (sequential) loader
-  self_supervised_temporal: self-supervised version of the temporal
   loader, for DDPMs

********************************
 Online Dataloaders and Options
********************************

Online dataloaders are useful when:

-  Images are too large to be processed fully
-  Dataset is labeled and model training should concentrate on labeled
   areas
-  Dataset is small and can benefit from augmentation from random
   cropping

The online dataloader applies the following steps:

-  Loads the input image according to
   `--data_oinline_creation_load_size_{A,B}`

-  Pick a bounding box randomly

-  Builds a mask from the bounding box

-  Crop around the bounding box according to fixed size
   `--data_online_creation_crop_size_{A,B}`

-  Randomly pick and apply a positive or negative offset to the crop
   size according to `--data_online_creation_crop_delta_{A,B}`. This
   allows random variations around the fixed size of the crop.

-  Randomly pick and apply a positive or negative offset to the mask
   according to `--data_online_creation_mask_delta_{A,B}`. This step
   allows for an object in domain A to roughly match the size of an
   object in domain B. E.g. turning cars into buses requires an offset
   on masks from the car domain so that the mask can fit a bus.

**********************************
 Temporal Dataloaders and Options
**********************************

Temporal dataloaders read sequences of images. This is useful in two
cases:

-  Temporal smoothing with GANs using a temporal discriminator
-  Frame conditioning with DDPMs

The temporal dataloader applies the following steps:

-  Uses the `--data_temporal_num_common_char` to sort the frames in the
   dataset. The number of common characters defines the length of the
   filename that should not be used for sorting. E.g. sorting files
   named `image_xxxx` would use 6 as the number of common chars, sorting
   based on xxxx only.

-  Selects `--data_temporal_number_frames` interleaved with
   `--data_temporal_frame_step` frames. This is useful to control the
   temporal smoothing or conditioning independently from the video's
   true frames per second.
