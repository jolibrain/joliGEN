# Dataloader


To choose a dataloader please use the flag `--dataset_mode dataloader_name`.


## Unaligned dataset


Name : `unaligned`\
You need to create two directories to host images from domain A `/path/to/data/trainA` and from domain B `/path/to/data/trainB`. Then you can train the model with the dataset flag `--dataroot /path/to/data`. Optionally, you can create hold-out test datasets at `/path/to/data/testA` and `/path/to/data/testB` to test your model on unseen images.


## Unaligned and labeled dataset


Name : `unaligned_labeled`\
You need to create two directories to host images from domain A `/path/to/data/trainA` and from domain B `/path/to/data/trainB`. In `trainA`, you have to separate your data into directories, each directory belongs to a class. Then you can train the model with the dataset flag `--dataroot /path/to/data`. Optionally, you can create hold-out test datasets at `/path/to/data/testA` and `/path/to/data/testB` to test your model on unseen images.

## Unaligned and labeled (with masks) dataset


Name : `unaligned_labeled_mask`\
For each domain A and B, you have to create a file `paths.txt` which each line gives paths to the image and to the mask, separeted by space, e.g. `path/to/image path/to/mask`.\
You need two create two directories to host `paths.txt` from each domain A `/path/to/data/trainA` and from domain B `/path/to/data/trainB`. Then you can train the model with the dataset flag `--dataroot /path/to/data`. Optionally, you can create hold-out test datasets at `/path/to/data/testA` and `/path/to/data/testB` to test your model on unseen images.