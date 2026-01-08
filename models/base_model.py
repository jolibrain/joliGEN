import cv2
from util.fvd import FrechetVideoDistance
import copy
import os

import torch

if torch.__version__[0] == "2":
    import torch._dynamo

from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import ExitStack

import numpy as np
import torch.nn.functional as F
from thop import profile
from torchviz import make_dot

# for metrics
from data.base_dataset import get_transform
from util.metrics import _compute_statistics_of_dataloader

from tqdm import tqdm
from piq import MSID, KID, FID, psnr, ssim
from lpips import LPIPS

from util.util import save_image, tensor2im, delete_flop_param
from util.util import pad_to_lpips_safe

from util.diff_aug import DiffAugment
from util.discriminator import DiscriminatorInfo

# For export
from util.export import export

# for D accuracy
from util.image_pool import ImagePool

# Iter Calculator
from util.iter_calculator import IterCalculator
from util.network_group import NetworkGroup
from util.util import delete_flop_param, save_image, tensor2im, MAX_INT

from . import base_networks, semantic_networks

# For D loss computing
from .modules import loss
from .modules.utils import get_scheduler
from .modules.sam.sam_inference import predict_sam
from .modules.diffusion_utils import rearrange_5dto4d_bf
import logging


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt, rank):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.rank = rank
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.with_amp = opt.with_amp
        self.use_cuda = (
            opt.use_cuda
        )  # torch.cuda.is_available() and self.gpu_ids and self.gpu_ids[0] >= 0
        if self.use_cuda:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.with_amp)
        if hasattr(opt, "fs_light"):
            self.fs_light = opt.fs_light
        self.device = torch.device(
            "cuda:{}".format(self.gpu_ids[rank]) if self.use_cuda else "cpu"
        )  # get device name: CPU or GPU
        self.save_dir = os.path.join(
            opt.checkpoints_dir, opt.name
        )  # save all the checkpoints to save_dir
        if (
            opt.data_preprocess != "scale_width"
        ):  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.display_param = []
        self.set_display_param()
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

        # create image buffers to store  images

        self.real_A_pool = ImagePool(opt.train_pool_size)
        self.fake_B_pool = ImagePool(opt.train_pool_size)
        self.fake_A_pool = ImagePool(opt.train_pool_size)
        self.real_B_pool = ImagePool(opt.train_pool_size)

        self.niter = 0

        self.objects_to_update = []

        if opt.output_display_G_attention_masks:
            for i in range(opt.G_attn_nb_mask_attn):
                temp_visual_names_attn = []
                temp_visual_names_attn += ["attention_" + str(i)]
                temp_visual_names_attn += ["output_" + str(i)]
                if i < opt.G_attn_nb_mask_attn - opt.G_attn_nb_mask_input:
                    temp_visual_names_attn += ["image_" + str(i)]

                self.visual_names.append(temp_visual_names_attn)

        self.margin = self.opt.data_online_context_pixels * 2

        if "segformer" in self.opt.G_netG:
            self.onnx_opset_version = 11
        elif (
            "ittr" in self.opt.G_netG
            or "unet_mha" in self.opt.G_netG
            or "uvit" in self.opt.G_netG
            or "dit" in self.opt.G_netG
        ):
            self.onnx_opset_version = 12
        else:
            self.onnx_opset_version = 9

        if "FID" in self.opt.train_metrics_list:
            self.fid_metric = FID()
        if "MSID" in self.opt.train_metrics_list:
            self.msid_metric = MSID()
        if "KID" in self.opt.train_metrics_list:
            self.kid_metric = KID()
        if "LPIPS" in self.opt.train_metrics_list:
            self.lpips_metric = LPIPS().to(self.device)
        if "FVD" in self.opt.train_metrics_list:
            self.fvd_metric = FrechetVideoDistance()

    def init_metrics(self, dataloader_test, test_name=""):
        self.use_inception = any(
            metric in self.opt.train_metrics_list for metric in ["KID", "FID", "MSID"]
        )

        if self.opt.train_compute_metrics_test and self.use_inception:
            dims = 2048
            if self.use_cuda:
                test_device = self.gpu_ids[0]
            else:
                test_device = self.device  # cpu
            self.netFid = base_networks.define_inception(test_device, dims)

        if self.opt.data_relative_paths:
            self.root = self.opt.dataroot
        else:
            self.root = None

        if self.opt.train_compute_metrics_test:
            pathB = self.save_dir + "/fakeB"
            if not os.path.exists(pathB):
                os.mkdir(pathB)

            if self.use_inception:
                path_sv_B = os.path.join(
                    self.opt.checkpoints_dir, self.opt.name, "fid_mu_sigma_B_test.npz"
                )

                if self.use_cuda:
                    test_device = self.gpu_ids[0]
                else:
                    test_device = self.device  # cpu

                realactB_test = _compute_statistics_of_dataloader(
                    path_sv=path_sv_B,
                    model=self.netFid,
                    domain="B",
                    batch_size=self.opt.test_batch_size,
                    dims=dims,
                    device=test_device,
                    dataloader=dataloader_test,
                    nb_max_img=self.opt.train_nb_img_max_fid,
                    root=self.root,
                    data_image_bits=self.opt.data_image_bits,
                )
                setattr(self, "realactB_test" + test_name, realactB_test)

    def init_semantic_cls(self, opt):
        # specify the semantic training networks and losses.
        # The training/test scripts will call <BaseModel.get_current_losses>

        losses_G = ["G_sem_cls_AB"]

        if hasattr(self, "fake_A"):
            losses_G.append("G_sem_cls_BA")

        if self.opt.train_sem_idt:
            losses_G += ["G_sem_cls_idt_B"]
            if hasattr(self, "fake_A"):
                losses_G += ["G_sem_cls_idt_A"]

        losses_CLS = ["CLS"]

        self.loss_names_G += losses_G
        self.loss_names_CLS = losses_CLS
        self.loss_names += losses_G + losses_CLS

        # define network CLS
        if self.isTrain:
            self.netCLS = semantic_networks.define_C(**vars(opt))

            self.model_names += ["CLS"]

            # define loss functions
            self.criterionCLS = torch.nn.modules.CrossEntropyLoss()

            self.optimizer_CLS = opt.optim(
                opt,
                self.netCLS.parameters(),
                lr=opt.train_sem_lr_f_s,
                betas=(opt.train_beta1, opt.train_beta2),
                weight_decay=opt.train_optim_weight_decay,
                eps=opt.train_optim_eps,
            )

            if opt.train_cls_regression:
                if opt.train_cls_l1_regression:
                    self.criterionCLS = torch.nn.L1Loss()
                else:
                    self.criterionCLS = torch.nn.modules.MSELoss()
            else:
                self.criterionCLS = torch.nn.modules.CrossEntropyLoss()

            self.optimizers.append(self.optimizer_CLS)

            ###Making groups
            self.group_CLS = NetworkGroup(
                networks_to_optimize=["CLS"],
                forward_functions=None,
                backward_functions=["compute_CLS_loss"],
                loss_names_list=["loss_names_CLS"],
                optimizer=["optimizer_CLS"],
                loss_backward=["loss_CLS"],
            )
            self.networks_groups.append(self.group_CLS)

    def init_semantic_mask(self, opt):
        # specify the semantic training networks and losses.
        # The training/test scripts will call <BaseModel.get_current_losses>
        losses_G = ["G_sem_mask_AB"]

        if hasattr(self, "fake_A"):
            losses_G += ["G_sem_mask_BA"]

        if self.opt.train_sem_idt:
            losses_G += ["G_sem_mask_idt_B"]
            if hasattr(self, "fake_A"):
                losses_G += ["G_sem_mask_idt_A"]

        if opt.train_mask_out_mask:
            losses_G += ["G_out_mask_AB"]
            if hasattr(self, "fake_A"):
                losses_G += ["G_out_mask_BA"]

        if opt.f_s_net != "sam":
            losses_f_s = ["f_s"]
        else:
            losses_f_s = []

        self.loss_names_G += losses_G
        self.loss_names_f_s = losses_f_s

        self.loss_names += losses_G + losses_f_s

        # define networks (both generator and discriminator)
        if self.isTrain:
            networks_f_s = []
            if self.opt.f_s_net == "sam":
                self.netf_s, self.f_s_mg = semantic_networks.define_f(**vars(opt))
                networks_f_s.append("f_s")

            elif self.opt.train_mask_disjoint_f_s:
                self.opt.train_f_s_B = True

                self.netf_s_A = semantic_networks.define_f(**vars(opt))
                networks_f_s.append("f_s_A")

                self.netf_s_B = semantic_networks.define_f(**vars(opt))
                networks_f_s.append("f_s_B")
            else:
                self.netf_s = semantic_networks.define_f(**vars(opt))
                networks_f_s.append("f_s")

            self.model_names += networks_f_s

            # define loss functions
            tweights = None
            if opt.f_s_class_weights:
                print("Using f_s class weights=", opt.f_s_class_weights)
                tweights = torch.FloatTensor(opt.f_s_class_weights).to(self.device)
            if opt.f_s_net != "sam":
                self.criterionf_s = torch.nn.modules.CrossEntropyLoss(weight=tweights)
            else:
                self.criterionf_s = torch.nn.MSELoss()

            if opt.train_mask_out_mask:
                if opt.train_mask_loss_out_mask == "L1":
                    self.criterionMask = torch.nn.L1Loss()
                elif opt.train_mask_loss_out_mask == "MSE":
                    self.criterionMask = torch.nn.MSELoss()
                elif opt.train_mask_loss_out_mask == "Charbonnier":
                    self.criterionMask = L1_Charbonnier_loss(
                        opt.train_mask_charbonnier_eps
                    )

            if self.opt.f_s_net != "sam":
                if self.opt.train_mask_disjoint_f_s:
                    self.optimizer_f_s = opt.optim(
                        opt,
                        itertools.chain(
                            self.netf_s_A.parameters(), self.netf_s_B.parameters()
                        ),
                        lr=opt.train_sem_lr_f_s,
                        betas=(opt.train_beta1, opt.train_beta2),
                        weight_decay=opt.train_optim_weight_decay,
                        eps=opt.train_optim_eps,
                    )
                else:
                    self.optimizer_f_s = opt.optim(
                        opt,
                        self.netf_s.parameters(),
                        lr=opt.train_sem_lr_f_s,
                        betas=(opt.train_beta1, opt.train_beta2),
                        weight_decay=opt.train_optim_weight_decay,
                        eps=opt.train_optim_eps,
                    )

                self.optimizers.append(self.optimizer_f_s)

            ###Making groups
            if opt.f_s_net != "sam":
                self.group_f_s = NetworkGroup(
                    networks_to_optimize=networks_f_s,
                    forward_functions=None,
                    backward_functions=["compute_f_s_loss"],
                    loss_names_list=["loss_names_f_s"],
                    optimizer=["optimizer_f_s"],
                    loss_backward=["loss_f_s"],
                )
                self.networks_groups.append(self.group_f_s)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @staticmethod
    def modify_commandline_options_train(parser):
        return parser

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        if "A_ref" in data:  # accomodates multi-frames dataloader
            self.real_A_with_context = data["A_ref"].to(self.device)
        else:
            self.real_A_with_context = data["A"].to(self.device)
        if "real_B_prompt" in data:
            self.real_B_prompt = data["real_B_prompt"]
        self.real_A = self.real_A_with_context.clone()
        if self.opt.data_online_context_pixels > 0:
            self.real_A = self.real_A[
                :,
                :,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
            ]

            self.real_A_with_context_vis = torch.nn.functional.interpolate(
                self.real_A_with_context, size=self.real_A.shape[2:]
            )

        if "B_ref" in data:
            self.real_B_with_context = data["B_ref"].to(self.device)
        else:
            self.real_B_with_context = data["B"].to(self.device)

        self.real_B = self.real_B_with_context.clone()

        if self.opt.data_online_context_pixels > 0:
            self.real_B = self.real_B[
                :,
                :,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
            ]

            self.real_B_with_context_vis = torch.nn.functional.interpolate(
                self.real_B_with_context, size=self.real_B.shape[2:]
            )

        self.image_paths = data["A_img_paths"]

        self.input_A_ref_bbox = None
        self.input_B_ref_bbox = None

        if self.opt.train_semantic_mask:
            self.set_input_semantic_mask(data)
        if self.opt.train_semantic_cls:
            self.set_input_semantic_cls(data)

    def set_input_semantic_mask(self, data):
        if "A_label_mask" in data:
            self.input_A_label_mask = data["A_label_mask"].to(self.device).squeeze(1)
            self.input_A_ref_bbox = data["A_ref_bbox"]
            if "A_ref_label_mask" in data:
                self.input_A_ref_label_mask = (
                    data["A_ref_label_mask"].to(self.device).squeeze(1)
                )
                self.input_A_label_mask = self.input_A_ref_label_mask

            if self.opt.data_online_context_pixels > 0:
                self.input_A_label_mask = self.input_A_label_mask[
                    :,
                    self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                    self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                ]

        if "B_label_mask" in data:
            self.input_B_label_mask = data["B_label_mask"].to(self.device).squeeze(1)
            self.input_B_ref_bbox = data.get("B_ref_bbox", None)
            if "B_ref_label_mask" in data:
                self.input_B_ref_label_mask = (
                    data["B_ref_label_mask"].to(self.device).squeeze(1)
                )
                self.input_B_label_mask = self.input_B_ref_label_mask

            if self.opt.data_online_context_pixels > 0:
                self.input_B_label_mask = self.input_B_label_mask[
                    :,
                    self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                    self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                ]

    def set_input_semantic_cls(self, data):
        if "A_label_cls" in data:
            if not self.opt.train_cls_regression:
                self.input_A_label_cls = (
                    data["A_label_cls"].to(torch.long).to(self.device)
                )
            else:
                self.input_A_label_cls = (
                    data["A_label_cls"].to(torch.float).to(device=self.device)
                )
        if "B_label_cls" in data:
            if not self.opt.train_cls_regression:
                self.input_B_label_cls = (
                    data["B_label_cls"].to(torch.long).to(self.device)
                )
            else:
                self.input_B_label_cls = (
                    data["B_label_cls"].to(torch.float).to(device=self.device)
                )

    def set_input_temporal(self, data_temporal):
        self.temporal_real_A_with_context = data_temporal["A"].to(self.device)
        self.temporal_real_B_with_context = data_temporal["B"].to(self.device)

        if self.opt.data_online_context_pixels > 0:
            self.temporal_real_A = self.temporal_real_A_with_context[
                :,
                :,
                :,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
            ]
            self.temporal_real_B = self.temporal_real_B_with_context[
                :,
                :,
                :,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
            ]

        else:
            self.temporal_real_A = self.temporal_real_A_with_context
            self.temporal_real_B = self.temporal_real_B_with_context

        for i in range(self.opt.data_temporal_number_frames):
            setattr(
                self,
                "temporal_real_A_" + str(i) + "_with_context",
                self.temporal_real_A_with_context[:, i],
            )

            if self.opt.data_online_context_pixels > 0:
                setattr(
                    self,
                    "temporal_real_A_" + str(i),
                    self.temporal_real_A_with_context[
                        :,
                        i,
                        :,
                        self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                        self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                    ],
                )

                setattr(
                    self,
                    "temporal_real_A_" + str(i) + "_with_context_vis",
                    torch.nn.functional.interpolate(
                        getattr(self, "temporal_real_A_" + str(i) + "_with_context"),
                        # size=self.temporal_real_A.shape[2:]
                        size=getattr(self, "temporal_real_A_" + str(i)).shape[2:],
                    ),
                )
            else:
                setattr(
                    self,
                    "temporal_real_A_" + str(i),
                    self.temporal_real_A_with_context[
                        :,
                        i,
                    ],
                )

            # Temporal Real B

            setattr(
                self,
                "temporal_real_B_" + str(i) + "_with_context",
                self.temporal_real_B_with_context[:, i],
            )

            if self.opt.data_online_context_pixels > 0:
                setattr(
                    self,
                    "temporal_real_B_" + str(i),
                    self.temporal_real_B_with_context[
                        :,
                        i,
                        :,
                        self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                        self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                    ],
                )

                setattr(
                    self,
                    "temporal_real_B_" + str(i) + "_with_context_vis",
                    torch.nn.functional.interpolate(
                        getattr(self, "temporal_real_B_" + str(i) + "_with_context"),
                        size=getattr(self, "temporal_real_B_" + str(i)).shape[2:],
                    ),
                )
            else:
                setattr(
                    self,
                    "temporal_real_B_" + str(i),
                    self.temporal_real_B_with_context[
                        :,
                        i,
                    ],
                )

        self.image_paths = data_temporal["A_img_paths"]

        if self.opt.train_semantic_mask:
            self.set_input_semantic_mask(data_temporal)
        if self.opt.train_semantic_cls:
            self.set_input_semantic_cls(data_temporal)

    def forward(self):
        for forward_function in self.forward_functions:
            getattr(self, forward_function)()

    def compute_fake_with_context(self, fake_name, real_name):
        setattr(
            self,
            fake_name + "_with_context",
            torch.nn.functional.pad(
                getattr(self, fake_name),
                (
                    self.opt.data_online_context_pixels,
                    self.opt.data_online_context_pixels,
                    self.opt.data_online_context_pixels,
                    self.opt.data_online_context_pixels,
                ),
            ),
        )

        setattr(
            self,
            fake_name + "_with_context",
            getattr(self, fake_name + "_with_context")
            + self.mask_context * getattr(self, real_name + "_with_context"),
        )
        setattr(
            self,
            fake_name + "_with_context_vis",
            torch.nn.functional.interpolate(
                getattr(self, fake_name + "_with_context"), size=self.real_A.shape[2:]
            ),
        )

    def compute_temporal_fake_with_context(self, fake_name, real_name):
        setattr(
            self,
            fake_name + "_with_context",
            torch.nn.functional.pad(
                getattr(self, fake_name),
                (
                    self.opt.data_online_context_pixels,
                    self.opt.data_online_context_pixels,
                    self.opt.data_online_context_pixels,
                    self.opt.data_online_context_pixels,
                ),
            ),
        )

        setattr(
            self,
            fake_name + "_with_context",
            getattr(self, fake_name + "_with_context")
            + self.mask_context * getattr(self, real_name + "_with_context"),
        )
        setattr(
            self,
            fake_name + "_with_context_vis",
            torch.nn.functional.interpolate(
                getattr(self, fake_name + "_with_context"),
                size=getattr(self, real_name).shape[2:],
            ),
        )

    def compute_temporal_fake(self, objective_domain):
        origin_domain = "B" if objective_domain == "A" else "A"
        netG = getattr(self, "netG_" + origin_domain)
        temporal_fake = []

        for i in range(self.opt.data_temporal_number_frames):
            temporal_fake.append(
                netG(getattr(self, "temporal_real_" + origin_domain)[:, i])
            )

        temporal_fake = torch.stack(temporal_fake, dim=1)

        for i in range(self.opt.data_temporal_number_frames):
            setattr(
                self,
                "temporal_fake_" + objective_domain + "_" + str(i),
                temporal_fake[:, i],
            )
            if self.opt.data_online_context_pixels > 0:
                self.compute_temporal_fake_with_context(
                    fake_name="temporal_fake_" + objective_domain + "_" + str(i),
                    real_name="temporal_real_" + origin_domain + "_" + str(i),
                )

        setattr(self, "temporal_fake_" + objective_domain, temporal_fake)

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [
                get_scheduler(optimizer, opt) for optimizer in self.optimizers
            ]
        if not self.isTrain or opt.train_continue:
            load_suffix = (
                "iter_%d" % opt.train_load_iter
                if opt.train_load_iter > 0
                else opt.train_epoch
            )
            if opt.train_finetune:
                # allow network to not already exists
                try:
                    self.load_networks(load_suffix)
                except Exception as e:
                    print(e)
            else:
                self.load_networks(load_suffix)

    def parallelize(self, rank):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name).to(self.gpu_ids[rank])
                self.set_requires_grad(net, True)
                net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
                setattr(
                    self,
                    "net" + name,
                    torch.nn.parallel.DistributedDataParallel(
                        net, device_ids=[self.gpu_ids[rank]], broadcast_buffers=False
                    ),
                )

    def single_gpu(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name).to(self.gpu_ids[0])
                setattr(self, "net" + name, net)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals(self.opt.test_batch_size)

    def compute_visuals(self, nb_imgs):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.train_lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr_G = self.optimizers[0].param_groups[0]["lr"]
        # lr_D = self.optimizers[1].param_groups[0]['lr']
        # print('learning rate G = %.7f' % lr_G, ' / learning rate D = %.7f' % lr_D)

    def get_current_visuals(self, nb_imgs, phase="train", test_name=""):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = []
        for i, group in enumerate(self.visual_names):
            cur_visual = OrderedDict()
            for name in group:
                if phase == "test":
                    name = name + "_test_" + test_name
                if isinstance(name, str) and hasattr(self, name):
                    cur_visual[name] = getattr(self, name)
            visual_ret.append(cur_visual)
            if (
                self.opt.model_type != "cut"
                and self.opt.model_type != "cycle_gan"
                and not self.opt.G_netG == "unet_vid"
            ):  # GANs have more outputs in practice, including semantics
                if i == nb_imgs - 1:
                    break
        if phase == "test" and self.opt.G_netG == "unet_vid":
            visual_ret = visual_ret[
                : self.opt.test_batch_size * (self.opt.data_temporal_number_frames)
            ]

        return visual_ret

    def get_display_param(self):
        param = OrderedDict()
        for name in self.display_param:
            if isinstance(name, str):
                param[name] = getattr(self.opt, name)
        return param

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, "loss_" + name)

        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "%s_net_%s.pth" % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, "net" + name)

                if len(self.gpu_ids) > 1 and self.use_cuda:
                    if (
                        name == "G_A"
                        and hasattr(net.module, "unet")
                        and hasattr(net.module, "vae")
                        and any(
                            "lora" in n for n, _ in net.module.unet.named_parameters()
                        )
                    ):
                        net.module.save_lora_config(save_path)
                    else:
                        torch.save(net.module.state_dict(), save_path)
                else:
                    if (
                        name == "G_A"
                        and hasattr(net, "unet")
                        and hasattr(net, "vae")
                        and any("lora" in n for n, _ in net.unet.named_parameters())
                    ):
                        net.save_lora_config(save_path)
                    else:
                        torch.save(net.state_dict(), save_path)

    def export_networks(self, epoch):
        """Export chosen networks weights to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names_export:
            if isinstance(name, str):
                save_filename = "%s_net_%s.pth" % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)

                net = getattr(self, "net" + name)

                # onnx

                if not self.opt.model_type in [
                    "palette",
                    "cm",
                    "cm_gan",
                    "sc",
                    "b2b",
                ]:  # Note: export is for generators from GANs only at the moment
                    # For export
                    from util.export import export

                    input_nc = self.opt.model_input_nc
                    if self.opt.model_multimodal:
                        input_nc += self.opt.train_mm_nz

                    # onnx
                    if (
                        not self.opt.train_feat_wavelet
                        and not "ittr" in self.opt.G_netG
                        and not "hdit" in self.opt.G_netG
                        and not "img2img_turbo" in self.opt.G_netG
                        and not "hat" == self.opt.G_netG
                        and not (
                            torch.__version__[0] == "2"
                            and "segformer" in self.opt.G_netG
                        )
                    ):  # XXX: segformer export fails with ONNX and Pytorch2
                        export_path_onnx = save_path.replace(".pth", ".onnx")

                        export(
                            self.opt,
                            cuda=False,  # onnx export is made on cpu
                            model_in_file=save_path,
                            model_out_file=export_path_onnx,
                            opset_version=self.onnx_opset_version,
                            export_type="onnx",
                        )

                    # jit
                    if (
                        self.opt.train_export_jit
                        and not ("uvit" in self.opt.G_netG)
                        and not ("hdit" in self.opt.G_netG)
                        and not ("img2img_turbo" in self.opt.G_netG)
                    ):
                        export_path_jit = save_path.replace(".pth", ".pt")

                        export(
                            self.opt,
                            cuda=False,  # jit export is made on cpu
                            model_in_file=save_path,
                            model_out_file=export_path_jit,
                            opset_version=self.onnx_opset_version,
                            export_type="jit",
                        )

    def get_dummy_input(self, device=None):
        input_nc = self.opt.model_input_nc
        if self.opt.model_multimodal:
            input_nc += self.opt.train_mm_nz

        if device is None:
            device = self.device
        dummy_input = torch.randn(
            1,
            input_nc,
            self.opt.data_crop_size,
            self.opt.data_crop_size,
            device=device,
        )

        return dummy_input

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = "%s_net_%s.pth" % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, "net" + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                if not os.path.isfile(load_path) and "temporal" in load_path:
                    print("Skipping missing temporal discriminator pre-trained weights")
                    continue
                print("loading the model from %s" % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4

                if self.opt.model_prior_321_backwardcompatibility:
                    for key in list(
                        state_dict.keys()
                    ):  # need to copy keys here because we mutate in loop
                        if "cond_embed" in key:
                            new_key = key.replace("denoise_fn.cond_embed", "cond_embed")
                            state_dict[new_key] = state_dict[key].clone()
                            del state_dict[key]

                        elif "denoise_fn" in key:
                            new_key = key.replace("denoise_fn", "denoise_fn.model")
                            state_dict[new_key] = state_dict[key].clone()
                            del state_dict[key]

                if getattr(self.opt, "alg_diffusion_ddpm_cm_ft", False):
                    model_dict = net.state_dict()
                    filtered = {}

                    for k, v in state_dict.items():
                        if "denoise_fn.model.cond_embed" in k:
                            new_k = k.replace(
                                "denoise_fn.model.cond_embed",
                                "cm_cond_embed.projection",
                            )
                        elif k.startswith("cond_embed."):
                            new_k = k.replace("cond_embed", "cm_cond_embed.projection")
                        elif "denoise_fn.model." in k:
                            new_k = k.replace("denoise_fn.model.", "cm_model.")
                        else:
                            new_k = k

                        if new_k in model_dict and v.shape == model_dict[new_k].shape:
                            filtered[new_k] = v
                        else:
                            if "cond_embed" in k:
                                print(f"⚠️ unmatched cond_embed key {k} → {new_k}")
                            else:
                                print(
                                    f"⚠️ skipping {new_k}: shape {v.shape if hasattr(v, 'shape') else 'N/A'}"
                                )

                    missing = set(model_dict.keys()) - set(filtered.keys())
                    extra = set(state_dict.keys()) - set(model_dict.keys())

                    print(
                        f"Loaded {len(filtered)}/{len(model_dict)} params; {len(missing)} missing.",
                        flush=True,
                    )

                    if missing:
                        print("\n⚠️ Missing keys:")
                        for k in sorted(missing):
                            print("   ", k)

                    net.load_state_dict(filtered, strict=False)

                else:
                    state1 = list(state_dict.keys())
                    state2 = list(net.state_dict().keys())
                    state1.sort()
                    state2.sort()

                    for key1, key2 in zip(state1, state2):
                        if key1 != key2:
                            print(key1 == key2, key1, key2)

                    if hasattr(state_dict, "_ema"):
                        net.load_state_dict(
                            state_dict["_ema"], strict=self.opt.model_load_no_strictness
                        )
                    else:
                        if (
                            name == "G_A"
                            and hasattr(net, "unet")
                            and hasattr(net, "vae")
                            and any("lora" in n for n, _ in net.unet.named_parameters())
                        ):
                            net.load_lora_config(load_path)
                            print("loading the lora")
                        else:
                            net.load_state_dict(
                                state_dict, strict=self.opt.model_load_no_strictness
                            )

    def get_nets(self):
        return_nets = {}
        for name in self.model_names:
            if isinstance(name, str):
                return_nets[name] = getattr(self, "net" + name)
        return return_nets

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for name, param in net.named_parameters():
                    if self.opt.G_netG == "img2img_turbo":
                        if "lora" or "skip_conv" or "conv_in" in name:
                            param.requires_grad = requires_grad
                    if (
                        not "freeze" in name
                        and not "cv_ensemble" in name
                        and not ("f_s" in name and self.opt.f_s_net == "sam")
                    ):  # cv_ensemble is for vision-aided
                        param.requires_grad = requires_grad
                    else:
                        param.requires_grad = False

    def save_networks_img(self, data):
        self.set_input(data)
        paths = []
        for name in self.model_names:
            net = getattr(self, "net" + name)
            path = self.opt.checkpoints_dir + self.opt.name + "/networks/" + name
            if not "Decoder" in name:
                y_0 = self.gt_image
                y_cond = self.cond_image
                mask = self.mask
                noise = None

                noise, noise_hat = net(y_0, y_cond, mask, noise)
                # temp = net(self.real_A)
                temp = noise_hat
            else:
                temp = net(self.netG_A(self.real_A).detach())
            make_dot(temp, params=dict(net.named_parameters())).render(
                path, format="png"
            )
            # paths.append(path)

        return paths

    def set_display_param(self, params=None):
        if params is None:
            params = vars(self.opt).keys()
        for param in params:
            self.display_param.append(param)
        self.display_param.sort()

    def compute_step(
        self, optimizers_names, loss_names
    ):  # loss_names are only use to compute average values over iter_size
        optimizers = []
        for optimizer_name in optimizers_names:
            optimizers.append(getattr(self, optimizer_name))

        if self.opt.train_iter_size > 1:
            for loss_name in loss_names:
                value = (
                    getattr(self, "loss_" + loss_name).clone()
                    / self.opt.train_iter_size
                )
                if torch.is_tensor(value):
                    value = value.detach()
                self.iter_calculator.compute_step(loss_name, value)

        if self.niter % self.opt.train_iter_size == 0:
            for optimizer in optimizers:
                if self.use_cuda:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            if self.opt.train_iter_size > 1:
                self.iter_calculator.compute_last_step(loss_names)
                for loss_name in loss_names:
                    setattr(
                        self,
                        "loss_" + loss_name + "_avg",
                        getattr(self.iter_calculator, "loss_" + loss_name),
                    )

    def ema_step(self, network_name):
        ema_beta = self.opt.train_G_ema_beta
        network = getattr(self, "net" + network_name)
        network_ema = getattr(self, "net" + network_name + "_ema", None)
        # - first iteration create the EMA + add to self.model_names + new X_ema to self.visual_names
        if network_ema is None:
            setattr(self, "net" + network_name + "_ema", copy.deepcopy(network).eval())
            network_ema = getattr(self, "net" + network_name + "_ema")
        # - update EMAs
        with torch.no_grad():
            for p_ema, p in zip(network_ema.parameters(), network.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))  # p is updated as well
            for b_ema, b in zip(network_ema.buffers(), network.buffers()):
                b_ema.copy_(b)

    def get_current_batch_size(self):
        return self.real_A.shape[0]

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        self.niter = self.niter + 1

        with ExitStack() as stack:
            if torch.__version__[0] == "2" and self.opt.with_torch_compile:
                torch._dynamo.config.suppress_errors = (
                    True  # automatic fall back to eager mode
                )

            if len(self.opt.gpu_ids) > 1 and self.niter % self.opt.train_iter_size != 0:
                for network in self.model_names:
                    stack.enter_context(getattr(self, "net" + network).no_sync())

            for group in self.networks_groups:
                for network in self.model_names:
                    if network in group.networks_to_optimize:
                        self.set_requires_grad(getattr(self, "net" + network), True)
                    else:
                        self.set_requires_grad(getattr(self, "net" + network), False)

                if not group.forward_functions is None:
                    with torch.cuda.amp.autocast(enabled=self.with_amp):
                        for forward in group.forward_functions:
                            if (
                                torch.__version__[0] == "2"
                                and self.opt.with_torch_compile
                                and self.niter == 1
                            ):
                                print("Torch compile forward function=", forward)
                                setattr(
                                    self, forward, torch.compile(getattr(self, forward))
                                )

                            getattr(self, forward)()

                for backward in group.backward_functions:
                    if (
                        torch.__version__[0] == "2"
                        and self.opt.with_torch_compile
                        and self.niter == 1
                    ):
                        print("Torch compile backward function=", backward)
                        setattr(self, backward, torch.compile(getattr(self, backward)))

                    getattr(self, backward)()

                for loss in group.loss_backward:
                    if self.use_cuda:
                        ll = (
                            self.scaler.scale(getattr(self, loss))
                            / self.opt.train_iter_size
                        )
                    else:
                        ll = getattr(self, loss) / self.opt.train_iter_size
                    if self.opt.model_multimodal:
                        retain_graph = True
                    else:
                        retain_graph = False
                    ll.backward(retain_graph=retain_graph)

                loss_names = []

                for temp in group.loss_names_list:
                    loss_names += getattr(self, temp)

                self.compute_step(group.optimizer, loss_names)

                if self.opt.train_G_ema:
                    for network in self.model_names:
                        if network in group.networks_to_ema:
                            self.ema_step(network)

            for cur_object in self.objects_to_update:
                cur_object.update(self.niter)

    def compute_miou_f_s_generic(self, pred, target):
        target = self.one_hot(target)
        intersection = (pred * target).sum()
        total = (pred + target).sum()
        union = total - intersection

        IoU = (intersection) / (union)
        return IoU

    def compute_miou(self):
        self.miou_real_A = self.compute_miou_f_s_generic(
            self.gt_pred_f_s_real_A, self.input_A_label_mask
        )
        self.miou_real_B = self.compute_miou_f_s_generic(
            self.gt_pred_f_s_real_B, self.input_B_label_mask
        )

        self.miou_fake_B = self.compute_miou_f_s_generic(
            self.pfB, self.input_A_label_mask
        )
        if hasattr(self, "fake_A"):
            self.miou_fake_A = self.compute_miou_f_s_generic(
                self.pfA, self.input_B_label_mask
            )

    def get_current_miou(self):
        miou = OrderedDict()
        miou_names = ["miou_real_A", "miou_real_B", "miou_fake_B"]
        if hasattr(self, "fake_A"):
            miou_names.append("miou_fake_A")

        for name in miou_names:
            if isinstance(name, str):
                miou[name] = float(
                    getattr(self, name)
                )  # float(...) works for both scalar tensor and float number
        return miou

    def one_hot(self, tensor):
        batch_size, height, width = tensor.shape
        one_hot = torch.zeros(
            batch_size,
            self.opt.f_s_semantic_nclasses,
            height,
            width,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return one_hot.scatter_(1, tensor.unsqueeze(1), 1.0)

    def compute_fake_real_masks(self):
        fake_mask = self.netf_s(self.real_A)
        fake_mask = F.gumbel_softmax(fake_mask, tau=1.0, hard=True, dim=1)
        real_mask = self.netf_s(
            self.real_B
        )  # f_s(B) is a good approximation of the real mask when task is easy
        real_mask = F.gumbel_softmax(real_mask, tau=1.0, hard=True, dim=1)

        setattr(self, "fake_mask_B_inv", fake_mask.argmax(dim=1))
        setattr(self, "real_mask_B_inv", real_mask.argmax(dim=1))
        setattr(self, "fake_mask_B", fake_mask)
        setattr(self, "real_mask_B", real_mask)

    def compute_f_s_loss(self):
        """Calculate segmentation loss for f_s"""
        self.loss_f_s = 0

        if "mask" in self.opt.D_netDs:
            for discriminator in self.discriminators:
                if "mask" in discriminator.name:
                    disc = discriminator
            domain = "B"
            netD = getattr(self, disc.name)
            loss = getattr(self, disc.loss_type)
            fake_name = disc.fake_name + "_" + domain
            real_name = disc.real_name + "_" + domain
            self.compute_fake_real_masks()
            self.loss_D_mask_value = (
                self.opt.alg_gan_lambda
                * self.compute_G_loss_GAN_generic(
                    netD,
                    domain,
                    loss,
                    fake_name=fake_name,
                    real_name=real_name,
                )
            )
            self.loss_f_s = self.loss_D_mask_value

        if not self.opt.train_mask_no_train_f_s_A:
            label_A = self.input_A_label_mask
            # forward only real source image through semantic classifier
            if self.opt.train_mask_disjoint_f_s:
                f_s = self.netf_s_A
            else:
                f_s = self.netf_s

            pred_A = f_s(self.real_A)
            self.loss_f_s += self.criterionf_s(pred_A, label_A)  # .squeeze(1))

        if self.opt.train_mask_f_s_B:
            if self.opt.train_mask_disjoint_f_s:
                f_s = self.netf_s_B
            else:
                f_s = self.netf_s

            if self.opt.data_refined_mask:
                # get mask with sam instead of label from self.real_B and self.input_B_ref_bbox
                self.label_sam_B = (
                    predict_sam(self.real_B, self.predictor_sam, self.input_B_ref_bbox)
                    > 0.0
                )
                label_B = self.label_sam_B.long()
            else:
                label_B = self.input_B_label_mask
            pred_B = f_s(self.real_B)
            self.loss_f_s += self.criterionf_s(pred_B, label_B)  # .squeeze(1))

    def compute_CLS_loss(self):
        """Calculate classif loss for cls"""
        label_A = self.input_A_label_cls
        # forward only real source image through semantic classifier
        pred_A = self.netCLS(self.real_A)
        if not self.opt.train_cls_regression:
            self.loss_CLS = self.opt.train_sem_cls_lambda * self.criterionCLS(
                pred_A, label_A
            )
        else:
            self.loss_CLS = self.opt.train_sem_cls_lambda * self.criterionCLS(
                pred_A.squeeze(1), label_A
            )
        if self.opt.train_sem_cls_B:
            label_B = self.input_B_label_cls
            pred_B = self.netCLS(self.real_B)
            if not self.opt.train_cls_regression:
                self.loss_CLS += self.opt.train_sem_cls_lambda * self.criterionCLS(
                    pred_B, label_B
                )
            else:
                self.loss_CLS += self.opt.train_sem_cls_lambda * self.criterionCLS(
                    pred_B.squeeze(1), label_B
                )

    def forward_semantic_mask(self):
        d = 1

        if self.opt.f_s_net == "sam":
            self.pred_f_s_real_A = predict_sam(
                self.real_A, self.f_s_mg, self.input_A_ref_bbox
            )
            self.input_A_label_mask = self.pred_f_s_real_A
            # self.input_A_label_mask = self.input_A_label_mask.float()
            self.gt_pred_f_s_real_A_max = (
                self.input_A_label_mask > 0.0
            ).float()  # Note: this is different than clamping
        else:
            if self.opt.train_mask_disjoint_f_s:
                f_s = self.netf_s_A
            else:
                f_s = self.netf_s
            self.pred_f_s_real_A = f_s(self.real_A)

            self.gt_pred_f_s_real_A = F.log_softmax(self.pred_f_s_real_A, dim=d)
            self.gt_pred_f_s_real_A_max = self.gt_pred_f_s_real_A.argmax(dim=d)

        if self.opt.train_mask_disjoint_f_s:
            f_s = self.netf_s_B
        else:
            f_s = self.netf_s

        if self.opt.f_s_net == "sam":
            self.pred_f_s_real_B = predict_sam(
                self.real_B, self.f_s_mg, self.input_B_ref_bbox
            )
            self.input_B_label_mask = self.pred_f_s_real_B
            self.gt_pred_f_s_real_B_max = (self.input_B_label_mask > 0.0).float()
        else:
            self.pred_f_s_real_B = f_s(self.real_B)
            self.gt_pred_f_s_real_B = F.log_softmax(self.pred_f_s_real_B, dim=d)
            self.gt_pred_f_s_real_B_max = self.gt_pred_f_s_real_B.argmax(dim=d)

        if self.opt.f_s_net == "sam":
            self.pred_f_s_fake_B = predict_sam(
                self.fake_B, self.f_s_mg, self.input_A_ref_bbox
            )
            self.pfB_max = (self.pred_f_s_fake_B > 0.0).float()
        else:
            self.pred_f_s_fake_B = f_s(self.fake_B)
            self.pfB = F.log_softmax(self.pred_f_s_fake_B, dim=d)
            self.pfB_max = self.pfB.argmax(dim=d)

        # fake A
        if hasattr(self, "fake_A"):
            self.pred_f_s_fake_A = f_s(self.fake_A)
            self.pfA = F.log_softmax(self.pred_f_s_fake_A, dim=d)
            self.pfA_max = self.pfA.argmax(dim=d)

        if self.opt.train_sem_idt:
            if self.opt.f_s_net == "sam":
                self.pred_f_s_idt_B = predict_sam(
                    self.idt_B, self.f_s_mg, self.input_B_ref_bbox
                )
                self.pfB_idt_max = (self.pred_f_s_idt_B > 0.0).float()
            else:
                self.pred_f_s_idt_B = f_s(self.idt_B)
                self.pred_f_s_idt_B = F.log_softmax(self.pred_f_s_idt_B, dim=d)
                self.pfB_idt_max = self.pred_f_s_idt_B.argmax(dim=d)

        if hasattr(self, "criterionMask"):
            label_A = self.input_A_label_mask
            label_A_inv = (
                torch.tensor(np.ones(label_A.size())).to(self.device) - label_A > 0.5
            )
            label_A_inv = label_A_inv.unsqueeze(1)
            if "mask" in self.opt.D_netDs:
                label_A_pred = self.gt_pred_f_s_real_A_max
                label_A_inv_pred = (
                    torch.tensor(np.ones(label_A_pred.size())).to(self.device)
                    - label_A_pred
                    > 0.5
                )
                self.fake_B_out_mask = self.fake_B * label_A_inv_pred
            else:
                self.fake_B_out_mask = self.fake_B * label_A_inv
            self.real_A_out_mask = self.real_A * label_A_inv
            if (
                hasattr(self, "fake_A")
                and hasattr(self, "input_B_label_mask")
                and len(self.input_B_label_mask) > 0
            ):
                label_B = self.input_B_label_mask
                label_B_inv = (
                    torch.tensor(np.ones(label_B.size())).to(self.device) - label_B > 0
                )
                label_B_inv = label_B_inv.unsqueeze(1)

                self.real_B_out_mask = self.real_B * label_B_inv
                self.fake_A_out_mask = self.fake_A * label_B_inv

    def forward_semantic_cls(self):
        d = 1
        self.pred_cls_real_A = self.netCLS(self.real_A)
        if not self.opt.train_cls_regression:
            _, self.gt_pred_cls_A = self.pred_cls_real_A.max(1)

        self.pred_cls_fake_B = self.netCLS(self.fake_B)
        if not self.opt.train_cls_regression:
            _, self.pfB = self.pred_cls_fake_B.max(1)

        if hasattr(self, "fake_A"):
            self.pred_cls_real_B = self.netCLS(self.real_B)
            if not self.opt.train_cls_regression:
                _, self.gt_pred_cls_B = self.pred_cls_real_B.max(1)

            self.pred_cls_fake_A = self.netCLS(self.fake_A)
            if not self.opt.train_cls_regression:
                _, self.pfB = self.pred_cls_fake_A.max(1)

    def get_current_metrics(self, test_names):
        metrics = OrderedDict()

        metrics_names = []

        if self.opt.train_compute_metrics_test:
            for name in test_names:
                if "FID" in self.opt.train_metrics_list:
                    metrics_names += [
                        "fidB_test_" + name,
                    ]

                if "MSID" in self.opt.train_metrics_list:
                    metrics_names += [
                        "msidB_test_" + name,
                    ]

                if "KID" in self.opt.train_metrics_list:
                    metrics_names += [
                        "kidB_test_" + name,
                    ]

                if "PSNR" in self.opt.train_metrics_list:
                    metrics_names += [
                        "psnr_test_" + name,
                    ]
                if "FVD" in self.opt.train_metrics_list:
                    metrics_names += [
                        "fvd_test_" + name,
                    ]

                if "SSIM" in self.opt.train_metrics_list:
                    metrics_names += [
                        "ssim_test_" + name,
                    ]

                if "LPIPS" in self.opt.train_metrics_list:
                    metrics_names += [
                        "lpips_test_" + name,
                    ]

            for name in metrics_names:
                if isinstance(name, str):
                    metrics[name] = float(
                        getattr(self, name)
                    )  # float(...) works for both scalar tensor and float number

            if (
                hasattr(self, "psnr_step_results")
                and "PSNR" in self.opt.train_metrics_list
            ):
                for test_name in test_names:
                    per_step_psnr = self.psnr_step_results.get(test_name, {})
                    for step, value in sorted(per_step_psnr.items()):
                        metrics[f"psnr_step_{step}_{test_name}"] = float(value)

            if (
                hasattr(self, "ssim_step_results")
                and "SSIM" in self.opt.train_metrics_list
            ):
                for test_name in test_names:
                    per_step_ssim = self.ssim_step_results.get(test_name, {})
                    for step, value in sorted(per_step_ssim.items()):
                        metrics[f"ssim_step_{step}_{test_name}"] = float(value)
            if (
                hasattr(self, "lpips_step_results")
                and "LPIPS" in self.opt.train_metrics_list
            ):
                for test_name in test_names:
                    per_step_lpips = self.lpips_step_results.get(test_name, {})
                    for step, value in sorted(per_step_lpips.items()):
                        metrics[f"lpips_step_{step}_{test_name}"] = float(value)

        return metrics

    def _compute_metrics(self, fake_images, gt_images):
        psnr_sum, ssim_sum, lpips_sum, n = 0.0, 0.0, 0.0, 0
        for fake, gt in zip(fake_images, gt_images):
            if fake.shape != gt.shape:
                print(f"Skip mismatched shapes: {fake.shape} vs {gt.shape}")
                continue
            fake = (fake.clamp(-1, 1).unsqueeze(0) + 1) / 2
            gt = (gt.clamp(-1, 1).unsqueeze(0) + 1) / 2
            min_lpips_size = 64
            h, w = fake.shape[-2:]

            if h < min_lpips_size or w < min_lpips_size:
                fake_lpips = pad_to_lpips_safe(fake, min_lpips_size)
                gt_lpips = pad_to_lpips_safe(gt, min_lpips_size)
            else:
                fake_lpips = fake
                gt_lpips = gt

            psnr_sum += psnr(fake, gt, data_range=1.0).item()
            ssim_sum += ssim(fake, gt).item()
            lpips_sum += self.lpips_metric(fake_lpips, gt_lpips).item()
            n += 1

        psnr_val = psnr_sum / n if n else 0.0
        ssim_val = ssim_sum / n if n else 0.0
        lpips_val = lpips_sum / n if n else 0.0
        return psnr_val, ssim_val, lpips_val

    def compute_metrics_test(
        self, dataloaders_test, n_epoch, n_iter, save_images=False, test_name=""
    ):
        dims = 2048
        batch = 1

        if hasattr(self, "netG_B"):
            netG = self.netG_B
        elif hasattr(self, "netG"):
            netG = self.netG

        fake_list = []
        real_list = []

        if self.opt.train_nb_img_max_fid != MAX_INT:
            progress = tqdm(
                desc="compute metrics test",
                position=1,
                total=self.opt.train_nb_img_max_fid,
            )
        else:
            progress = None

        for i, data_test_list in enumerate(
            dataloaders_test
        ):  # inner loop (minibatch) within one epoch
            data_test = data_test_list[0]
            if self.use_temporal:
                temporal_data_test = data_test_list[1]
                self.set_input_temporal(temporal_data_test)
            else:
                self.set_input(
                    data_test
                )  # unpack data from dataloader and apply preprocessing

            offset = i * self.opt.test_batch_size
            istrain = self.opt.isTrain
            self.opt.isTrain = False
            self.inference(self.opt.test_batch_size, offset=offset)
            self.opt.isTrain = istrain

            if save_images:
                pathB = self.save_dir + "/fakeB/%s_epochs_%s_iters_imgs" % (
                    n_epoch,
                    n_iter,
                )
                if not os.path.exists(pathB):
                    os.mkdir(pathB)

            for j, cur_fake_B in enumerate(self.fake_B):
                if save_images:
                    save_image(
                        tensor2im(cur_fake_B.unsqueeze(0)),
                        pathB + "/" + str(offset + j) + ".png",
                        aspect_ratio=1.0,
                    )

                fake_list.append(cur_fake_B.unsqueeze(0).clone())

            if hasattr(self, "gt_image"):
                batch_real_img = self.gt_image
            else:
                if self.opt.data_direction == "AtoB":
                    batch_real_img = self.real_B
                else:
                    batch_real_img = self.real_A

            for i, cur_real in enumerate(batch_real_img):
                real_list.append(cur_real.unsqueeze(0).clone())

            i = 0
            for sub_list in self.visual_names:
                if self.opt.G_netG == "unet_vid":
                    for name in sub_list:
                        if hasattr(self, name):
                            setattr(
                                self, name + "_test_" + test_name, getattr(self, name)
                            )

                else:
                    if i < offset:
                        i += 1
                        continue
                    for name in sub_list:
                        if hasattr(self, name):
                            setattr(
                                self, name + "_test_" + test_name, getattr(self, name)
                            )
                    i += 1
                    if i - offset == self.opt.test_batch_size:
                        break

            if progress:
                progress.n = min(len(fake_list), progress.total)
                progress.refresh()

            if len(fake_list) >= self.opt.train_nb_img_max_fid:
                break
            if self.opt.G_netG == "unet_vid" and i < self.opt.test_batch_size:
                break

        fake_list = fake_list[: self.opt.train_nb_img_max_fid]
        real_list = real_list[: self.opt.train_nb_img_max_fid]

        if progress:
            progress.close()

        if self.use_inception:
            if self.use_cuda:
                test_device = self.gpu_ids[0]
            else:
                test_device = self.device  # cpu
            domain = "B"
            if self.opt.data_direction == "BtoA":
                domain = "A"

            fakeactB_test = _compute_statistics_of_dataloader(
                path_sv=None,
                model=self.netFid,
                domain=domain,
                batch_size=1,
                dims=dims,
                device=test_device,
                dataloader=fake_list,
                nb_max_img=self.opt.train_nb_img_max_fid,
                root=self.root,
                data_image_bits=self.opt.data_image_bits,
            )

            realactB_test = getattr(self, "realactB_test" + test_name)
            (
                fidB_test,
                msidB_test,
                kidB_test,
            ) = self.compute_metrics_generic(realactB_test, fakeactB_test)

            setattr(self, "fidB_test_" + test_name, fidB_test)
            setattr(self, "msidB_test_" + test_name, msidB_test)
            setattr(self, "kidB_test_" + test_name, kidB_test)
        real_tensor = (torch.clamp(torch.cat(real_list), min=-1.0, max=1.0) + 1.0) / 2.0
        fake_tensor = (torch.clamp(torch.cat(fake_list), min=-1.0, max=1.0) + 1.0) / 2.0
        if self.opt.G_netG == "unet_vid":  # temporal
            real_tensor, fake_tensor = rearrange_5dto4d_bf(real_tensor, fake_tensor)
            ssim_test = ssim(real_tensor, fake_tensor)
            psnr_test = psnr(real_tensor, fake_tensor)
            if getattr(self.opt, "alg_palette_metric_mask", False) or getattr(
                self.opt, "alg_cm_metric_mask", False
            ):
                fake_images = [img for seq in self.fake_B_dilated for img in seq]
                gt_images = [img for seq in self.gt_image_dilated for img in seq]
                psnr_test, ssim_test, lpips_test = self._compute_metrics(
                    fake_images, gt_images
                )
                setattr(self, "psnr_test_" + test_name, psnr_test)
                setattr(self, "ssim_test_" + test_name, ssim_test)
                setattr(self, "lpips_test_" + test_name, lpips_test)

            elif getattr(self.opt, "alg_sc_metric_mask", False):

                self.psnr_step = {}
                self.ssim_step = {}
                self.lpips_step = {}

                steps = sorted(self.fake_B_dilated_per_step.keys())
                for step in steps:
                    fake_crops = [
                        img for seq in self.fake_B_dilated_per_step[step] for img in seq
                    ]
                    gt_crops = [
                        img
                        for seq in self.gt_image_dilated_per_step[step]
                        for img in seq
                    ]

                    psnr_val, ssim_val, lpips_val = self._compute_metrics(
                        fake_crops, gt_crops
                    )
                    self.psnr_step[step] = psnr_val
                    self.ssim_step[step] = ssim_val
                    self.lpips_step[step] = lpips_val

                if not hasattr(self, "psnr_step_results"):
                    self.psnr_step_results = {}
                if not hasattr(self, "ssim_step_results"):
                    self.ssim_step_results = {}
                if not hasattr(self, "lpips_step_results"):
                    self.lpips_step_results = {}

                self.psnr_step_results[test_name] = {
                    step: float(val) for step, val in self.psnr_step.items()
                }
                self.ssim_step_results[test_name] = {
                    step: float(val) for step, val in self.ssim_step.items()
                }
                self.lpips_step_results[test_name] = {
                    step: float(val) for step, val in self.lpips_step.items()
                }
                psnr_test = sum(self.psnr_step.values()) / len(self.psnr_step)
                ssim_test = sum(self.ssim_step.values()) / len(self.ssim_step)
                lpips_test = sum(self.lpips_step.values()) / len(self.lpips_step)

                setattr(self, "psnr_test_" + test_name, psnr_test)
                setattr(self, "ssim_test_" + test_name, ssim_test)
                setattr(self, "lpips_test_" + test_name, lpips_test)

            else:
                setattr(self, "psnr_test_" + test_name, psnr_test)
                setattr(self, "ssim_test_" + test_name, ssim_test)

        else:  # image
            ssim_test = ssim(real_tensor, fake_tensor)
            psnr_test = psnr(real_tensor, fake_tensor)

            if getattr(self.opt, "alg_palette_metric_mask", False) or getattr(
                self.opt, "alg_cm_metric_mask", False
            ):
                psnr_test, ssim_test, lpips_test = self._compute_metrics(
                    self.fake_B_dilated, self.gt_image_dilated
                )
                setattr(self, "psnr_test_" + test_name, psnr_test)
                setattr(self, "ssim_test_" + test_name, ssim_test)
                setattr(self, "lpips_test_" + test_name, lpips_test)

            elif getattr(self.opt, "alg_sc_metric_mask", False):

                self.psnr_step = {}
                self.ssim_step = {}
                self.lpips_step = {}
                steps = sorted(self.fake_B_dilated_per_step.keys())
                for step in steps:
                    fake_crops = self.fake_B_dilated_per_step[step]
                    gt_crops = self.gt_image_dilated_per_step[step]

                    psnr_val, ssim_val, lpips_val = self._compute_metrics(
                        fake_crops, gt_crops
                    )
                    self.psnr_step[step] = psnr_val
                    self.ssim_step[step] = ssim_val
                    self.lpips_step[step] = lpips_val

                if not hasattr(self, "psnr_step_results"):
                    self.psnr_step_results = {}
                if not hasattr(self, "ssim_step_results"):
                    self.ssim_step_results = {}
                if not hasattr(self, "lpips_step_results"):
                    self.lpips_step_results = {}

                self.psnr_step_results[test_name] = {
                    step: float(val) for step, val in self.psnr_step.items()
                }
                self.ssim_step_results[test_name] = {
                    step: float(val) for step, val in self.ssim_step.items()
                }
                self.lpips_step_results[test_name] = {
                    step: float(val) for step, val in self.lpips_step.items()
                }

                # Global average across steps (same logic as palette/cm)
                psnr_test = sum(self.psnr_step.values()) / len(self.psnr_step)
                ssim_test = sum(self.ssim_step.values()) / len(self.ssim_step)
                lpips_test = sum(self.lpips_step.values()) / len(self.lpips_step)
                setattr(self, "psnr_test_" + test_name, psnr_test)
                setattr(self, "ssim_test_" + test_name, ssim_test)
                setattr(self, "lpips_test_" + test_name, lpips_test)

            else:
                setattr(self, "psnr_test_" + test_name, psnr_test)
                setattr(self, "ssim_test_" + test_name, ssim_test)

        if "LPIPS" in self.opt.train_metrics_list:
            real_tensor = torch.cat(real_list)
            fake_tensor = torch.clamp(torch.cat(fake_list), min=-1, max=1)
            if len(real_tensor.shape) == 5:  # temporal
                real_tensor, fake_tensor = rearrange_5dto4d_bf(real_tensor, fake_tensor)
                lpips_test = self.lpips_metric(real_tensor, fake_tensor).mean()
            elif real_tensor.shape[1] > 3:  # 3+ channels
                real_tensor_3c = real_tensor[:, :-1, :, :]
                fake_tensor_3c = fake_tensor[:, :-1, :, :]
                lpips_test = self.lpips_metric(
                    real_tensor_3c, fake_tensor_3c
                ).mean()  ##TODO: per channel and sum
            else:
                lpips_test = self.lpips_metric(real_tensor, fake_tensor).mean()
            setattr(self, "lpips_test_" + test_name, lpips_test)

        if "FVD" in self.opt.train_metrics_list:
            real_tensor = torch.cat(real_list)
            fake_tensor = torch.clamp(torch.cat(fake_list), min=-1, max=1)
            combined_tensor = torch.cat((real_tensor, fake_tensor), dim=0)
            resized_frames = []
            for b in range(combined_tensor.shape[1]):
                resized_video = []
                for t in range(combined_tensor.shape[0]):
                    frame = combined_tensor[t, b].permute(1, 2, 0).cpu().numpy()
                    resized_frame = cv2.resize(frame, (224, 224))

                    resized_frame_tensor = torch.tensor(resized_frame).permute(2, 0, 1)
                    resized_video.append(resized_frame_tensor)
                resized_frames.append(torch.stack(resized_video))
            resized_tensor = torch.stack(resized_frames)
            real_tensor_fvd, fake_tensor_fvd = torch.split(resized_tensor, 2, dim=1)
            with torch.no_grad():
                fvd_test = self.fvd_metric.compute(real_tensor_fvd, fake_tensor_fvd)
            setattr(self, "fvd_test_" + test_name, fvd_test)

    def compute_metrics_generic(self, real_act, fake_act):
        # FID
        if "FID" in self.opt.train_metrics_list:
            fid = self.fid_metric(real_act, fake_act)
        else:
            fid = None

        # MSID
        if "MSID" in self.opt.train_metrics_list:
            msid = self.msid_metric(real_act, fake_act)
        else:
            msid = None

        # KID needs to have the same number of examples
        if "KID" in self.opt.train_metrics_list:
            if fake_act.shape == real_act.shape:
                kid = self.kid_metric(real_act, fake_act)
            else:
                print(
                    "KID needs to have the same number of examples in both domains. Here, there %d examples in real domain and %d in fake domain,we will use a subsample from each"
                    % (real_act.shape[0], fake_act.shape[0])
                )
                nb_sub_sample = min(real_act.shape[0], fake_act.shape[0])
                kid = self.kid_metric(
                    real_act[:nb_sub_sample], fake_act[:nb_sub_sample]
                )
        else:
            kid = None

        return fid, msid, kid

    def set_input_first_gpu(self, data):
        self.set_input(data)
        self.bs_per_gpu = self.real_A.size(0)
        self.real_A = self.real_A[: self.bs_per_gpu]
        self.real_B = self.real_B[: self.bs_per_gpu]

        if self.opt.train_semantic_mask:
            self.set_input_first_gpu_semantic_mask()

        if self.opt.train_semantic_cls:
            self.set_input_first_gpu_semantic_cls()

    def set_input_first_gpu_semantic_mask(self):
        if self.opt.f_s_net == "sam" and not hasattr(self, "input_A_label_mask"):
            self.input_A_label_mask = torch.randn(
                1,
                1,
                self.opt.data_crop_size,
                self.opt.data_crop_size,
                device=self.device,
            )
        self.input_A_label_mask = self.input_A_label_mask[: self.bs_per_gpu]
        if hasattr(self, "input_B_label_mask"):
            self.input_B_label_mask = self.input_B_label_mask[: self.bs_per_gpu]

    def set_input_first_gpu_semantic_cls(self):
        self.input_A_label_cls = self.input_A_label_cls[: self.bs_per_gpu]
        if hasattr(self, "input_B_label_cls"):
            self.input_B_label_cls = self.input_B_label_cls[: self.bs_per_gpu]

    def print_flop(self):
        model_name = "netG_A"
        model = getattr(self, model_name)
        input = self.get_dummy_input()

        if torch.is_tensor(input):
            input = (input,)

        macs, params = profile(model, inputs=(input))

        print(
            "Network %s has %d M macs, %d Gflops and %d M params."
            % (model_name, macs / 1e6, macs * 2 / 1e9, params / 1e6)
        )

        delete_flop_param(model)

    def iter_calculator_init(self):
        if self.opt.train_iter_size > 1:
            self.iter_calculator = IterCalculator(self.loss_names)
            for i, cur_loss in enumerate(self.loss_names):
                self.loss_names[i] = cur_loss + "_avg"
                setattr(
                    self,
                    "loss_" + self.loss_names[i],
                    torch.zeros(size=(), device=self.device),
                )
