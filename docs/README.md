# JoliGEN Documentation

+ [Datasets](#datasets)
+ [Dataloaders](#dataloaders)
+ [Training](#training)
+ [Inference](#inference)

+ [Models and Options](options.md)
+ [Project Source Overview](overview.md)

## Datasets

### CycleGAN Datasets

To train a model on your own datasets, you need to create a data folder with two subdirectories `trainA` and `trainB` that contain images from domain A and B. You can test your model on your training set by setting `--phase train` in `test.py`. You can also create subdirectories `testA` and `testB` if you have test data.

### Datasets with labels

Create `trainA` and `ŧrainB` directories as described for CycleGAN datasets.
In `trainA, you have to separate your data into directories, each directory belongs to a class.

### Datasets with masks

You can use a dataset made of images and their mask labels (it can be segmentation or attention masks). To do so, you have to generate masks which are pixel labels for the images. If you have n differents classes, pixel values in the mask have to be between 0 and n-1. You can specify the number of classes with the flag `--semantic_nclasses n`.

In the following example, the are two classes: an important zone and the background (0 are respresented in black and 1 in green).

## Dataloaders

To choose a dataloader please use the flag `--dataset_mode dataloader_name`.

### Unaligned dataset

Name : `unaligned`\
You need to create two directories to host images from domain A `/path/to/data/trainA` and from domain B `/path/to/data/trainB`. Then you can train the model with the dataset flag `--dataroot /path/to/data`. Optionally, you can create hold-out test datasets at `/path/to/data/testA` and `/path/to/data/testB` to test your model on unseen images.

### Unaligned and labeled dataset

Name : `unaligned_labeled`\
You need to create two directories to host images from domain A `/path/to/data/trainA` and from domain B `/path/to/data/trainB`. In `trainA`, you have to separate your data into directories, each directory belongs to a class. Then you can train the model with the dataset flag `--dataroot /path/to/data`. Optionally, you can create hold-out test datasets at `/path/to/data/testA` and `/path/to/data/testB` to test your model on unseen images.

### Unaligned and labeled (with masks) dataset

Name : `unaligned_labeled_mask`\
For each domain A and B, you have to create a file `paths.txt` which each line gives paths to the image and to the mask, separeted by space, e.g. `path/to/image path/to/mask`.\
You need two create two directories to host `paths.txt` from each domain A `/path/to/data/trainA` and from domain B `/path/to/data/trainB`. Then you can train the model with the dataset flag `--dataroot /path/to/data`. Optionally, you can create hold-out test datasets at `/path/to/data/testA` and `/path/to/data/testB` to test your model on unseen images.

## Training

All models and associated options are listed [here](options.md).

With `dataroot` the path of the dataset

### Train a cycleGAN
 
You can tune the hyperparameters in `./scripts/train_cyclegan.sh` and then use the following line command.
```
bash ./scripts/train_cyclegan.sh dataroot
```

### Train a cycleGAN with labels
 
You can tune the hyperparameters in `./scripts/train_cyclegan_semantic.sh` and then use the following line command.
```
bash ./scripts/train_cyclegan_semantic.sh dataroot
```

### Train a cycleGAN with mask labels
 
You can tune the hyperparameters in `./scripts/train_cyclegan_semantic_mask.sh` and then use the following line command.
```
bash ./scripts/train_cyclegan_semantic_mask.sh dataroot
```

### Visualize losses

To display losses from previous training, please run
```
python3 util/load_display_losses.py --loss_log_file_path path_to_repo_of_loss.json --port 8097 --env_name visdom_environment_name
```

## Inference

### Using python

```
python3 -m scripts.gen_single_image --model-in-file "/path/to/model_checkpoint.pth" --model-type mobile_resnet_9blocks --img-size 360 --img-in "/path/to/img_in.png" --img-out result.png
```

### Export model

```
python3 -m scripts.export_jit_model --model-in-file "/path/to/model_checkpoint.pth" --model-out-file exported_model.pt --model-type mobile_resnet_9blocks --img-size 360 
```

Then `exported_model.pt` can be reloaded without JoliGEN to perform inference with an external software, e.g. [DeepDetect](https://github.com/jolibrain/deepdetect) with torch backend.

<!-- Insert example with dede? -->
