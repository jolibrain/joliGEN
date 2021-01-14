### CycleGAN Datasets
To train a model on your own datasets, you need to create a data folder with two subdirectories `trainA` and `trainB` that contain images from domain A and B. You can test your model on your training set by setting `--phase train` in `test.py`. You can also create subdirectories `testA` and `testB` if you have test data.

You should **not** expect our method to work on just any random combination of input and output datasets (e.g. `cats<->keyboards`). From our experiments, we find it works better if two datasets share similar visual content. For example, `landscape painting<->landscape photographs` works much better than `portrait painting <-> landscape photographs`. `zebras<->horses` achieves compelling results while `cats<->dogs` completely fails.


### Datasets with labels
To train a model on your own datasets, you need to create a data folder with two subdirectories `trainA` and `trainB` that contain images from domain A and B. In `trainA`, you have to separate your data into directories, each directory belongs to a class.


### Datasets with masks
You can use a dataset made of images and their mask labels (it can be segmentation or attention masks). To do so, you have to generate masks which are pixel labels for the images. If you have n differents classes, pixel values in the mask have to be between 0 and n-1. You can specify the number of classes with the flag `--semantic_nclasses n`.\
In the following example, the are two classes: an important zone and the background (0 are respresented in black and 1 in green).

