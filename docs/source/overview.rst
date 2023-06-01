Overview of Code Structure
==========================

To help users better understand and use our codebase, we briefly
overview the functionality and implementation of each package and each
module. Please see the documentation in each file for more details. If
you have questions, you may find useful information in :doc:`training/test
tips <tips>` and `frequently asked questions <qa.md>`_.

`train.py <https://github.com/jolibrain/joliGEN/blob/master/train.py>`_ is a general-purpose training script. It
works for various models (with option :code:`--model`: e.g., :code:`cycle_gan`,
:code:`cut`, :code:`cycle_gan_semantic`, …) and different datasets (with option
:code:`--dataset_mode`: e.g., :code:`aligned`, :code:`unaligned`,
:code:`unaligned_labeled`, :code:`unaligned_labeled_mask`). See the section
:doc:`Datasets <datasets>` and :doc:`training/test tips <tips>` for more
details.

`test.py <https://github.com/jolibrain/joliGEN/blob/master/test.py>`_ is a general-purpose test script. Once you have
trained your model with :code:`train.py`, you can use this script to test
the model. It will load a saved model from :code:`--checkpoints_dir` and
save the results to :code:`--results_dir`. See :doc:`training/test tips <tips>`
for more details.

`data <https://github.com/jolibrain/joliGEN/blob/master/data>`_ directory contains all the modules related to data
loading and preprocessing. To add a custom dataset class called
:code:`dummy`, you need to add a file called :code:`dummy_dataset.py` and define
a subclass :code:`DummyDataset` inherited from :code:`BaseDataset`. You need to
implement four functions: :code:`__init__` (initialize the class, you need
to first call :code:`BaseDataset.__init__(self, opt)`), :code:`__len__` (return
the size of dataset), :code:`__getitem__` (get a data point), and
optionally :code:`modify_commandline_options` (add dataset-specific options
and set default options). Now you can use the dataset class by
specifying flag :code:`--dataset_mode dummy`. See our template dataset
`class <https://github.com/jolibrain/joliGEN/blob/master/data/template_dataset.py>`_ for an example. Below we explain
each file in details.

*  `\__init__.py <https://github.com/jolibrain/joliGEN/blob/master/data/__init__.py>`_ implements the interface
   between this package and training and test scripts. :code:`train.py` and
   :code:`test.py` call :code:`from data import create_dataset` and
   :code:`dataset = create_dataset(opt)` to create a dataset given the
   option :code:`opt`.
*  `base_dataset.py <https://github.com/jolibrain/joliGEN/blob/master/data/base_dataset.py>`_ implements an abstract
   base class (`ABC <https://docs.python.org/3/library/abc.html>`_ ) for
   datasets. It also includes common transformation functions (e.g.,
   :code:`get_transform`, :code:`__scale_width`), which can be later used in
   subclasses.
*  `image_folder.py <https://github.com/jolibrain/joliGEN/blob/master/data/image_folder.py>`_ implements an image
   folder class. We modify the official PyTorch image folder
   `code <https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py>`_
   so that this class can load images from both the current directory
   and its subdirectories.
*  `template_dataset.py <https://github.com/jolibrain/joliGEN/blob/master/data/template_dataset.py>`_ provides a
   dataset template with detailed documentation. Check out this file if
   you plan to implement your own dataset.
*  `aligned_dataset.py <https://github.com/jolibrain/joliGEN/blob/master/data/aligned_dataset.py>`_ includes a
   dataset class that can load image pairs. It assumes a single image
   directory :code:`/path/to/data/train`, which contains image pairs in the
   form of {A,B}. See
   `here <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#prepare-your-own-datasets-for-pix2pix>`_
   on how to prepare aligned datasets. During test time, you need to
   prepare a directory :code:`/path/to/data/test` as test data.
*  `unaligned_dataset.py <https://github.com/jolibrain/joliGEN/blob/master/data/unaligned_dataset.py>`_ includes a
   dataset class that can load unaligned/unpaired datasets. It assumes
   that two directories to host training images from domain A
   :code:`/path/to/data/trainA` and from domain B :code:`/path/to/data/trainB`
   respectively. Then you can train the model with the dataset flag
   :code:`--dataroot /path/to/data`. Similarly, you need to prepare two
   directories :code:`/path/to/data/testA` and :code:`/path/to/data/testB`
   during test time.
*  `single_dataset.py <https://github.com/jolibrain/joliGEN/blob/master/data/single_dataset.py>`_ includes a dataset
   class that can load a set of single images specified by the path
   :code:`--dataroot /path/to/data`. It can be used for generating CycleGAN
   results only for one side with the model option :code:`-model test`.
*  `colorization_dataset.py <https://github.com/jolibrain/joliGEN/blob/master/data/colorization_dataset.py>`_
   implements a dataset class that can load a set of nature images in
   RGB, and convert RGB format into (L, ab) pairs in
   `Lab <https://en.wikipedia.org/wiki/CIELAB_color_space>`_ color
   space. It is required by pix2pix-based colorization model
   (:code:`--model colorization`).
*  `unaligned_labeled_dataset.py <https://github.com/jolibrain/joliGEN/blob/master/data/unaligned_labeled_dataset/py>`_
   implements a dataset class that can load an unaligned/unpaired
   dataset with labels (for domain A). It assumes that two directories
   to host training images from domain A :code:`/path/to/data/trainA` and
   from domain B :code:`/path/to/data/trainB` respectively. In trainA,
   images are distributed into directories, each corresponding to a
   class. Then you can train the model with the dataset flag
   :code:`--dataroot /path/to/data`. Similarly, you need to prepare two
   directories :code:`/path/to/data/testA` and :code:`/path/to/data/testB`
   during test time.
*  `unaligned_labeled_mask_dataset.py <https://github.com/jolibrain/joliGEN/blob/master/data/unaligned_labeled_mask_dataset/py>`_
   implements a dataset class that can load an unaligned/unpaired
   dataset with pixel labels (segmentation/attention masks). It assumes
   two directories to host training images from domain A
   :code:`/path/to/data/trainA` and from domain B :code:`/path/to/data/trainB`
   respectively. In trainA and trainB, there is a :code:`paths.txt` file
   which contains paths to every image and to its label. Then you can
   train the model with the dataset flag :code:`--dataroot /path/to/data`.
   Similarly, you need to prepare two directories
   :code:`/path/to/data/testA` and :code:`/path/to/data/testB` during test time.

`models <https://github.com/jolibrain/joliGEN/blob/master/models>`_ directory
contains modules related to objective functions, optimizations,
and network architectures. To add a custom model class called ``dummy``,
you need to add a file called :code:`dummy_model.py` and define a subclass
:code:`DummyModel` inherited from :code:`BaseModel`. You need to implement four functions:
:code:`__init__` (initialize the class; you need to first call
:code:`BaseModel.__init__(self, opt)`), :code:`set_input` (unpack data from
dataset and apply preprocessing), :code:`forward` (generate intermediate
results), :code:`optimize_parameters` (calculate loss, gradients, and update
network weights), and optionally :code:`modify_commandline_options` (add
model-specific options and set default options). Now you can use the
model class by specifying flag :code:`--model dummy`. See our template model
`class <https://github.com/jolibrain/joliGEN/blob/master/models/template_model.py>`_
for an example. Below we explain each file in details.

*  `\__init__.py <https://github.com/jolibrain/joliGEN/blob/master/models/__init__.py>`_ implements the interface
   between this package and training and test scripts. :code:`train.py` and
   :code:`test.py` call :code:`from models import create_model` and
   :code:`model = create_model(opt)` to create a model given the option
   :code:`opt`. You also need to call :code:`model.setup(opt)` to properly
   initialize the model.
*  `base_model.py <https://github.com/jolibrain/joliGEN/blob/master/models/base_model.py>`_ implements an abstract
   base class (`ABC <https://docs.python.org/3/library/abc.html>`_ ) for
   models. It also includes commonly used helper functions (e.g.,
   :code:`setup`, :code:`test`, :code:`update_learning_rate`, :code:`save_networks`,
   :code:`load_networks`), which can be later used in subclasses.
*  `template_model.py <https://github.com/jolibrain/joliGEN/blob/master/models/template_model.py>`_ provides a model
   template with detailed documentation. Check out this file if you plan
   to implement your own model.
*  `cycle_gan_model.py <https://github.com/jolibrain/joliGEN/blob/master/models/cycle_gan_model.py>`_ implements the
   CycleGAN `model <https://junyanz.github.io/CycleGAN/>`_, for
   learning image-to-image translation without paired data. The model
   training requires :code:`--dataset_mode unaligned` dataset. By default,
   it uses a :code:`--netG resnet_9blocks` ResNet generator, a
   :code:`--netD basic` discriminator (PatchGAN introduced by pix2pix), and
   a least-square GANs `objective <https://arxiv.org/abs/1611.04076>`_
   (:code:`--gan_mode lsgan`).
*  `networks.py <https://github.com/jolibrain/joliGEN/blob/master/models/networks.py>`_ module implements network
   architectures (both generators and discriminators), as well as
   normalization layers, initialization methods, optimization scheduler
   (i.e., learning rate policy), and GAN objective function
   (:code:`vanilla`, :code:`lsgan`, :code:`wgangp`).
*  `test_model.py <https://github.com/jolibrain/joliGEN/blob/master/models/test_model.py>`_ implements a model that
   can be used to generate CycleGAN results for only one direction. This
   model will automatically set :code:`--dataset_mode single`, which only
   loads the images from one set. See the test
   `instruction <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#apply-a-pre-trained-model-cyclegan>`_
   for more details.
*  `cycle_gan_semantic_model.py <https://github.com/jolibrain/joliGEN/blob/master/models/cycle_gan_semantic_model.py>`_
   implements the CycleGAN
   `model <https://junyanz.github.io/CycleGAN/>`_, for learning
   image-to-image translation without paired data but with labels in
   domain A. The model training requires
   :code:`--dataset_mode unaligned_labeled` dataset. By default, it uses a
   :code:`--netG resnet_9blocks` ResNet generator, a :code:`--netD basic`
   discriminator (PatchGAN introduced by pix2pix), and a least-square
   GANs `objective <https://arxiv.org/abs/1611.04076>`_
   (:code:`--gan_mode lsgan`).
*  `cycle_gan_semantic_mask_model.py <https://github.com/jolibrain/joliGEN/blob/master/models/cycle_gan_semantic_mask_model.py>`_ 
   implements the CycleGAN
   `model <https://junyanz.github.io/CycleGAN/>`_, for learning
   image-to-image translation without paired data but with pixel labels
   in both domains. The model training requires
   :code:`--dataset_mode unaligned_labeled_mask` dataset. By default, it
   uses a :code:`--netG resnet_9blocks` ResNet generator, a :code:`--netD basic`
   discriminator (PatchGAN introduced by pix2pix), and a least-square
   GANs `objective <https://arxiv.org/abs/1611.04076>`_
   (:code:`--gan_mode lsgan`).
*  `cycle_gan_sty2_model.py <https://github.com/jolibrain/joliGEN/blob/master/models/cycle_gan_sty2_model.py>`_
   implements the CycleGAN
   `model <https://junyanz.github.io/CycleGAN/>`_, for learning
   image-to-image translation without paired data but with pixel labels
   in both domains. The model training requires
   :code:`--dataset_mode unaligned_labeled_mask` dataset. By default, it
   uses a :code:`--netG resnet_9blocks` ResNet generator, a :code:`--netD basic`
   discriminator (PatchGAN introduced by pix2pix), and a least-square
   GANs `objective <https://arxiv.org/abs/1611.04076>`_
   (:code:`--gan_mode lsgan`).

`options <https://github.com/jolibrain/joliGEN/blob/master/options>`_ directory includes our option modules: training
options, test options, and basic options (used in both training and
test). :code:`TrainOptions` and :code:`TestOptions` are both subclasses of
:code:`BaseOptions`. They will reuse the options defined in :code:`BaseOptions`.

*  `\__init__.py <https://github.com/jolibrain/joliGEN/blob/master/options/__init__.py>`_ is required to make Python
   treat the directory :code:`options` as containing packages,
*  `base_options.py <https://github.com/jolibrain/joliGEN/blob/master/options/base_options.py>`_ includes options that
   are used in both training and test. It also implements a few helper
   functions such as parsing, printing, and saving the options. It also
   gathers additional options defined in :code:`modify_commandline_options`
   functions in both dataset class and model class.
*  `train_options.py <https://github.com/jolibrain/joliGEN/blob/master/options/train_options.py>`_ includes options that
   are only used during training time.
*  `test_options.py <https://github.com/jolibrain/joliGEN/blob/master/options/test_options.py>`_ includes options that
   are only used during test time.

`util <https://github.com/jolibrain/joliGEN/blob/master/util>`_ directory includes a miscellaneous collection of
useful helper functions.

*  `\__init__.py <https://github.com/jolibrain/joliGEN/blob/master/util/__init__.py>`_ is
   required to make Python treat the directory :code:`util` as containing
   packages,
*  `get_data.py <https://github.com/jolibrain/joliGEN/blob/master/util/get_data.py>`_ provides a Python
   script for downloading CycleGAN and pix2pix datasets. Alternatively, You
   can also use bash scripts such as
   `download_pix2pix_model.sh <https://github.com/jolibrain/joliGEN/blob/master/scripts/download_pix2pix_model.sh>`_ and
   `download_cyclegan_model.sh <https://github.com/jolibrain/joliGEN/blob/master/scripts/download_cyclegan_model.sh>`_.
*  `html.py <https://github.com/jolibrain/joliGEN/blob/master/util/html.py>`_ implements a module that saves images
   into a single HTML file. It consists of functions such as :code:`add_header`
   (add a text header to the HTML file), :code:`add_images` (add a row of
   images to the HTML file), :code:`save` (save the HTML to the disk). It is
   based on Python library :code:`dominate`, a Python library for creating and
   manipulating HTML documents using a DOM API.
*  `image_pool.py <https://github.com/jolibrain/joliGEN/blob/master/util/image_pool.py>`_ implements an image buffer
   that stores previously generated images. This buffer enables us to
   update discriminators using a history of generated images rather than
   the ones produced by the latest generators. The original idea was
   discussed in this
   `paper <http://openaccess.thecvf.com/content_cvpr_2017/papers/Shrivastava_Learning_From_Simulated_CVPR_2017_paper.pdf>`_.
   The size of the buffer is controlled by the flag :code:`--pool_size`.
*  `visualizer.py <https://github.com/jolibrain/joliGEN/blob/master/util/visualizer.py>`_ includes several functions
   that can display/save images and print/save logging information. It uses
   a Python library :code:`visdom` for display and a Python library
   :code:`dominate` (wrapped in :code:`HTML`) for creating HTML files with images.
*  `util.py <https://github.com/jolibrain/joliGEN/blob/master/util/util.py>`_ consists of simple helper functions
   such as :code:`tensor2im` (convert a tensor array to a numpy image array),
   :code:`diagnose_network` (calculate and print the mean of average absolute
   value of gradients), and :code:`mkdirs` (create multiple directories).
