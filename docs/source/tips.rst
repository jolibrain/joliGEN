Training/test Tips
==================

Training/test options
---------------------

Please see :code:`options/train_options.py` and :code:`options/base_options.py`
for the training flags; see :code:`options/test_options.py` and
:code:`options/base_options.py` for the test flags. There are some
model-specific flags as well, which are added in the model files. The
default values of these options are also adjusted in the model files.

CPU/GPU 
-------

(default :code:`--gpu_ids 0`) Please set :code:`--gpu_ids -1` to
use CPU mode; set :code:`--gpu_ids 0,1,2` for multi-GPU mode. You need a
large batch size (e.g., :code:`--train_batch_size 32`) to benefit from
multiple GPUs.

Visualization
-------------

During training, the current results can be viewed using two methods.
First, if you set :code:`--output_display_id` > 0, the results and loss plot
will appear on a local graphics web server launched by
`visdom <https://github.com/facebookresearch/visdom>`_. To do this, you
should have :code:`visdom` installed and a server running by the command
:code:`python -m visdom.server`. The default server URL is
:code:`http://localhost:8097`. :code:`display_id` corresponds to the window ID
that is displayed on the :code:`visdom` server. The :code:`visdom` display
functionality is turned on by default. To avoid the extra overhead of
communicating with :code:`visdom` set :code:`--output_display_id -1`. Second,
the intermediate results are saved to
:code:`[opt.checkpoints_dir]/[opt.name]/web/` as an HTML file. To avoid
this, set :code:`--output_no_html`.

Preprocessing
-------------

Images can be resized and cropped in different ways using
:code:`--data_preprocess` option. The default option :code:`'resize_and_crop'`
resizes the image to be of size :code:`(opt.load_size, opt.load_size)` and
does a random crop of size :code:`(opt.crop_size, opt.crop_size)`.
:code:`'crop'` skips the resizing step and only performs random cropping.
:code:`'scale_width'` resizes the image to have width :code:`opt.crop_size`
while keeping the aspect ratio. :code:`'scale_width_and_crop'` first resizes
the image to have width :code:`opt.load_size` and then does random cropping
of size :code:`(opt.crop_size, opt.crop_size)`. :code:`'none'` tries to skip all
these preprocessing steps. However, if the image size is not a multiple
of some number depending on the number of downsamplings of the
generator, you will get an error because the size of the output image
may be different from the size of the input image. Therefore, :code:`'none'`
option still tries to adjust the image size to be a multiple of 4. You
might need a bigger adjustment if you change the generator architecture.
Please see :code:`data/base_datset.py` do see how all these were
implemented.

Fine-tuning/resume training
---------------------------

To fine-tune a pre-trained model, or resume the previous training, use
the :code:`--train_continue` flag. The program will then load the model
based on :code:`epoch`. By default, the program will initialize the epoch
count as 1. Set :code:`--train_epoch_count <int>` to specify a different
starting epoch count.

Training/Testing with high res images
-------------------------------------

JoliGEN is quite memory-intensive as at least four networks (two
generators and two discriminators) need to be loaded on one GPU, so a
large image cannot be entirely loaded. In this case, we recommend
training with cropped images. For example, to generate 1024px results,
you can train with
:code:`--data_preprocess scale_width_and_crop --data_load_size 1024 --data_crop_size 360`,
and test with :code:`--data_preprocess scale_width --data_load_size 1024`.
This way makes sure the training and test will be at the same scale. At
test time, you can afford higher resolution because you don’t need to
load all networks.

About loss curve
----------------

Unfortunately, the loss curve does not reveal much information in
training GANs, and joliGEN is no exception. To check whether the
training has converged or not, we recommend periodically generating a
few samples and looking at them.

About batch size
----------------

For all experiments in the paper, we set the batch size to be 1. If
there is room for memory, you can use higher batch size with batch norm
or instance norm. (Note that the default batchnorm does not work well
with multi-GPU training. You may consider using `synchronized
batchnorm <https://github.com/vacancy/Synchronized-BatchNorm-PyTorch>`_
instead). But please be aware that it can impact the training. In
particular, even with Instance Normalization, different batch sizes can
lead to different results. Moreover, increasing :code:`--crop_size` may be a
good alternative to increasing the batch size.
