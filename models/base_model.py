import os
import copy
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import gan_networks, semantic_networks
from .modules.utils import get_scheduler
from torchviz import make_dot


from util.network_group import NetworkGroup

# for FID
from data.base_dataset import get_transform
from .modules.fid.pytorch_fid.fid_score import (
    _compute_statistics_of_path,
    calculate_frechet_distance,
)
from util.util import save_image, tensor2im
import numpy as np
from util.diff_aug import DiffAugment
from . import base_networks

# for D accuracy
from util.image_pool import ImagePool
import torch.nn.functional as F

# For D loss computing
from .modules import loss
from util.discriminator import DiscriminatorInfo

# For export
from util.export.onnx import export_onnx


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

        if rank == 0 and (opt.train_compute_fid or opt.train_compute_fid_val):
            self.transform = get_transform(opt, grayscale=(opt.model_input_nc == 1))
            dims = 2048
            batch = 1
            self.netFid = base_networks.define_inception(self.gpu_ids[0], dims)

            pathA = opt.dataroot + "/trainA"
            path_sv_A = os.path.join(
                opt.checkpoints_dir, opt.name, "fid_mu_sigma_A.npz"
            )
            if self.opt.data_relative_paths:
                self.root = opt.dataroot
            else:
                self.root = None

            if not os.path.isfile(path_sv_A):
                self.realmA, self.realsA = _compute_statistics_of_path(
                    pathA,
                    self.netFid,
                    batch,
                    dims,
                    self.gpu_ids[0],
                    self.transform,
                    nb_max_img=opt.train_nb_img_max_fid,
                    root=self.root,
                )
                np.savez(path_sv_A, mu=self.realmA, sigma=self.realsA)
            else:
                print("Mu and sigma loaded for domain A")
                self.realmA, self.realsA = _compute_statistics_of_path(
                    path_sv_A,
                    self.netFid,
                    batch,
                    dims,
                    self.gpu_ids[0],
                    self.transform,
                    nb_max_img=opt.train_nb_img_max_fid,
                    root=self.root,
                )

            pathB = opt.dataroot + "/trainB"
            path_sv_B = os.path.join(
                opt.checkpoints_dir, opt.name, "fid_mu_sigma_B.npz"
            )
            if not os.path.isfile(path_sv_B):
                self.realmB, self.realsB = _compute_statistics_of_path(
                    pathB,
                    self.netFid,
                    batch,
                    dims,
                    self.gpu_ids[0],
                    self.transform,
                    nb_max_img=opt.train_nb_img_max_fid,
                    root=self.root,
                )
                np.savez(path_sv_B, mu=self.realmB, sigma=self.realsB)
            else:

                print("Mu and sigma loaded for domain B")
                self.realmB, self.realsB = _compute_statistics_of_path(
                    path_sv_B,
                    self.netFid,
                    batch,
                    dims,
                    self.gpu_ids[0],
                    self.transform,
                    nb_max_img=opt.train_nb_img_max_fid,
                    root=self.root,
                )
            pathA = self.save_dir + "/fakeA/"
            if not os.path.exists(pathA):
                os.mkdir(pathA)

            pathB = self.save_dir + "/fakeB/"
            if not os.path.exists(pathB):
                os.mkdir(pathB)

            if hasattr(self, "netG_B"):
                self.fidA = 0
            self.fidB = 0

        if rank == 0 and opt.train_compute_fid_val:
            ### For validation
            pathB = self.save_dir + "/fakeB/"
            if not os.path.exists(pathB):
                os.mkdir(pathB)

            pathB = opt.dataroot + "/validationB"
            path_sv = os.path.join(
                opt.checkpoints_dir, opt.name, "fid_mu_sigma_B_val.npz"
            )
            if not os.path.isfile(path_sv):
                self.realmB_val, self.realsB_val = _compute_statistics_of_path(
                    pathB,
                    self.netFid,
                    batch,
                    dims,
                    self.gpu_ids[0],
                    self.transform,
                    nb_max_img=opt.train_nb_img_max_fid,
                    root=self.root,
                )
                np.savez(path_sv, mu=self.realmB_val, sigma=self.realsB_val)
            else:
                print("Mu and sigma loaded for domain B (validation)")
                self.realmB_val, self.realsB_val = _compute_statistics_of_path(
                    path_sv,
                    self.netFid,
                    batch,
                    dims,
                    self.gpu_ids[0],
                    self.transform,
                    nb_max_img=opt.train_nb_img_max_fid,
                    root=self.root,
                )

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
        elif "ittr" in self.opt.G_netG or "unet_mha" in self.opt.G_netG:
            self.onnx_opset_version = 12
        else:
            self.onnx_opset_version = 9

        if self.opt.output_display_env == "":
            self.opt.output_display_env = self.opt.name

    def init_semantic_cls(self, opt):

        # specify the semantic training networks and losses.
        # The training/test scripts will call <BaseModel.get_current_losses>

        losses_G = ["G_sem_cls_AB"]

        if hasattr(self, "fake_A"):
            losses_G.append("G_sem_cls_BA")

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

        losses_f_s = ["f_s"]

        self.loss_names_G += losses_G
        self.loss_names_f_s = losses_f_s

        self.loss_names += losses_G + losses_f_s

        # define networks (both generator and discriminator)
        if self.isTrain:
            networks_f_s = []
            if self.opt.train_mask_disjoint_f_s:
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
            self.criterionf_s = torch.nn.modules.CrossEntropyLoss(weight=tweights)

            if opt.train_mask_out_mask:
                if opt.train_mask_loss_out_mask == "L1":
                    self.criterionMask = torch.nn.L1Loss()
                elif opt.train_mask_loss_out_mask == "MSE":
                    self.criterionMask = torch.nn.MSELoss()
                elif opt.train_mask_loss_out_mask == "Charbonnier":
                    self.criterionMask = L1_Charbonnier_loss(
                        opt.train_mask_charbonnier_eps
                    )

            if self.opt.train_mask_disjoint_f_s:
                self.optimizer_f_s = opt.optim(
                    opt,
                    itertools.chain(
                        self.netf_s_A.parameters(), self.netf_s_B.parameters()
                    ),
                    lr=opt.train_sem_lr_f_s,
                    betas=(opt.train_beta1, opt.train_beta2),
                )
            else:
                self.optimizer_f_s = opt.optim(
                    opt,
                    self.netf_s.parameters(),
                    lr=opt.train_sem_lr_f_s,
                    betas=(opt.train_beta1, opt.train_beta2),
                )

            self.optimizers.append(self.optimizer_f_s)

            ###Making groups
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

    def set_input(self, data):

        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A_with_context = data["A"].to(self.device)
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
            self.real_B_with_context, size=self.real_A.shape[2:]
        )

        self.image_paths = data["A_img_paths"]

        if self.opt.train_semantic_mask:
            self.set_input_semantic_mask(data)
        if self.opt.train_semantic_cls:
            self.set_input_semantic_cls(data)

    def set_input_semantic_mask(self, data):
        if "A_label_mask" in data:
            self.input_A_label_mask = data["A_label_mask"].to(self.device).squeeze(1)

            if self.opt.data_online_context_pixels > 0:
                self.input_A_label_mask = self.input_A_label_mask[
                    :,
                    self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                    self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                ]

        if "B_label_mask" in data:
            self.input_B_label_mask = data["B_label_mask"].to(self.device).squeeze(1)

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

        for i in range(self.opt.D_temporal_number_frames):
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
                        size=self.real_A.shape[2:],
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
                        size=self.real_B.shape[2:],
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

    def forward(self):
        for forward_function in self.forward_functions:
            getattr(self, forward_function)()

    def compute_temporal_fake(self, objective_domain):
        origin_domain = "B" if objective_domain == "A" else "A"
        netG = getattr(self, "netG_" + origin_domain)
        temporal_fake = []

        for i in range(self.opt.D_temporal_number_frames):
            temporal_fake.append(
                netG(getattr(self, "temporal_real_" + origin_domain)[:, i])
            )

        temporal_fake = torch.stack(temporal_fake, dim=1)

        for i in range(self.opt.D_temporal_number_frames):
            setattr(
                self,
                "temporal_fake_" + objective_domain + "_" + str(i),
                temporal_fake[:, i],
            )
            if self.opt.data_online_context_pixels > 0:
                self.compute_fake_with_context(
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
            self.compute_visuals()

    def compute_visuals(self):
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

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = []
        for i, group in enumerate(self.visual_names):
            cur_visual = OrderedDict()
            for name in group:
                if isinstance(name, str):
                    cur_visual[name] = getattr(self, name)
            visual_ret.append(cur_visual)
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
                errors_ret[name] = float(
                    getattr(self, "loss_" + name)
                )  # float(...) works for both scalar tensor and float number
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
                    torch.save(net.module.state_dict(), save_path)
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

                if (
                    not "ittr" in self.opt.G_netG
                    and not "palette" in self.opt.model_type
                ):
                    input_nc = self.opt.model_input_nc
                    if self.opt.model_multimodal:
                        input_nc += self.opt.train_mm_nz

                    # onnx
                    if not "ittr" in self.opt.G_netG:
                        export_path_onnx = save_path.replace(".pth", ".onnx")

                        export_onnx(
                            self.opt,
                            cuda=False,  # onnx export is made on cpu
                            model_in_file=save_path,
                            model_out_file=export_path_onnx,
                            opset_version=self.onnx_opset_version,
                        )

                    # jit
                    if self.opt.train_export_jit and not "segformer" in self.opt.G_netG:
                        export_path_jit = save_path.replace(".pth", ".pt")
                        jit_model = torch.jit.trace(net, self.get_dummy_input())
                        jit_model.save(export_path_jit)

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

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "running_mean" or key == "running_var"
            ):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "num_batches_tracked"
            ):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1
            )

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
                for key in list(
                    state_dict.keys()
                ):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(
                        state_dict, net, key.split(".")
                    )

                if hasattr(state_dict, "g_ema"):
                    net.load_state_dict(state_dict["g_ema"])
                else:
                    net.load_state_dict(state_dict)

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
                    if (
                        not "freeze" in name and not "cv_ensemble" in name
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
                temp = net(self.real_A)
            else:
                temp = net(self.netG_A(self.real_A).detach())
            make_dot(temp, params=dict(net.named_parameters())).render(
                path, format="png"
            )
            paths.append(path)

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
                if len(self.opt.gpu_ids) > 1:
                    torch.distributed.all_reduce(
                        value, op=torch.distributed.ReduceOp.SUM
                    )  # loss value is summed accross gpus
                    value = value / len(self.opt.gpu_ids)
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

        for group in self.networks_groups:
            for network in self.model_names:
                if network in group.networks_to_optimize:
                    self.set_requires_grad(getattr(self, "net" + network), True)
                else:
                    self.set_requires_grad(getattr(self, "net" + network), False)

            if not group.forward_functions is None:
                with torch.cuda.amp.autocast(enabled=self.with_amp):
                    for forward in group.forward_functions:
                        getattr(self, forward)()

            for backward in group.backward_functions:
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

    def compute_f_s_loss(self):
        """Calculate segmentation loss for f_s"""
        self.loss_f_s = 0
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

        self.pred_f_s_real_B = f_s(self.real_B)
        self.gt_pred_f_s_real_B = F.log_softmax(self.pred_f_s_real_B, dim=d)
        self.gt_pred_f_s_real_B_max = self.gt_pred_f_s_real_B.argmax(dim=d)

        self.pred_f_s_fake_B = f_s(self.fake_B)
        self.pfB = F.log_softmax(self.pred_f_s_fake_B, dim=d)  # .argmax(dim=d)
        self.pfB_max = self.pfB.argmax(dim=d)

        # fake A
        if hasattr(self, "fake_A"):
            self.pred_f_s_fake_A = f_s(self.fake_A)
            self.pfA = F.log_softmax(self.pred_f_s_fake_A, dim=d)
            self.pfA_max = self.pfA.argmax(dim=d)

        if self.opt.train_sem_idt:
            self.pred_f_s_idt_B = f_s(self.idt_B)
            self.pred_f_s_idt_B = F.log_softmax(self.pred_f_s_idt_B, dim=d)

        if hasattr(self, "criterionMask"):
            label_A = self.input_A_label_mask
            label_A_in = label_A.unsqueeze(1)
            label_A_inv = (
                torch.tensor(np.ones(label_A.size())).to(self.device) - label_A > 0.5
            )
            label_A_inv = label_A_inv.unsqueeze(1)
            self.real_A_out_mask = self.real_A * label_A_inv
            self.fake_B_out_mask = self.fake_B * label_A_inv

            if (
                hasattr(self, "fake_A")
                and hasattr(self, "input_B_label_mask")
                and len(self.input_B_label_mask) > 0
            ):

                label_B = self.input_B_label_mask
                label_B_in = label_B.unsqueeze(1)
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

    def compute_fid(self, n_epoch, n_iter):
        dims = 2048
        batch = 1

        # A->B
        if hasattr(self, "netG_B"):
            pathA = self.save_dir + "/fakeA/" + str(n_iter) + "_" + str(n_epoch)
            if not os.path.exists(pathA):
                os.mkdir(pathA)
            if len(self.fake_A_pool.get_all()) > 0:
                for i, temp_fake_A in enumerate(self.fake_A_pool.get_all()):
                    save_image(
                        tensor2im(temp_fake_A),
                        pathA + "/" + str(i) + ".png",
                        aspect_ratio=1.0,
                    )
                self.fakemA, self.fakesA = _compute_statistics_of_path(
                    pathA,
                    self.netFid,
                    batch,
                    dims,
                    self.gpu_ids[0],
                    nb_max_img=self.opt.train_nb_img_max_fid,
                    root=self.root,
                )

        # B->A
        pathB = self.save_dir + "/fakeB/" + str(n_iter) + "_" + str(n_epoch)
        if not os.path.exists(pathB):
            os.mkdir(pathB)

        for j, temp_fake_B in enumerate(self.fake_B_pool.get_all()):
            save_image(
                tensor2im(temp_fake_B), pathB + "/" + str(j) + ".png", aspect_ratio=1.0
            )
        self.fakemB, self.fakesB = _compute_statistics_of_path(
            pathB,
            self.netFid,
            batch,
            dims,
            self.gpu_ids[0],
            nb_max_img=self.opt.train_nb_img_max_fid,
            root=self.root,
        )

        if len(self.fake_A_pool.get_all()) > 0:
            self.fidA = calculate_frechet_distance(
                self.realmA, self.realsA, self.fakemA, self.fakesA
            )
        self.fidB = calculate_frechet_distance(
            self.realmB, self.realsB, self.fakemB, self.fakesB
        )

    def get_current_fids(self):

        fids = OrderedDict()

        if hasattr(self, "netG_B"):
            fid_names = ["fidA", "fidB"]
        else:
            fid_names = ["fidB"]

        for name in fid_names:
            if isinstance(name, str):
                fids[name] = float(
                    getattr(self, name)
                )  # float(...) works for both scalar tensor and float number

        return fids

    def compute_fid_val(self):
        dims = 2048
        batch = 1

        pathB = self.save_dir + "/fakeB/%s_imgs" % (self.opt.data_max_dataset_size)
        if not os.path.exists(pathB):
            os.mkdir(pathB)

        if hasattr(self, "netG_B"):
            netG = self.netG_B
        elif hasattr(self, "netG"):
            netG = self.netG

        self.fake_B_val = self.compute_fake_val(self.real_A_val, netG)

        for j, temp_fake_B in enumerate(self.fake_B_val):
            save_image(
                tensor2im(temp_fake_B.unsqueeze(0)),
                pathB + "/" + str(j) + ".png",
                aspect_ratio=1.0,
            )

        self.fakemB_val, self.fakesB_val = _compute_statistics_of_path(
            pathB,
            self.netFid,
            batch,
            dims,
            self.gpu_ids[0],
            nb_max_img=self.opt.train_nb_img_max_fid,
            root=self.root,
        )

        self.fidB_val = calculate_frechet_distance(
            self.realmB_val, self.realsB_val, self.fakemB_val, self.fakesB_val
        )
        return self.fidB_val

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
        self.input_A_label_mask = self.input_A_label_mask[: self.bs_per_gpu]
        if hasattr(self, "input_B_label_mask"):
            self.input_B_label_mask = self.input_B_label_mask[: self.bs_per_gpu]

    def set_input_first_gpu_semantic_cls(self):
        self.input_A_label_cls = self.input_A_label_cls[: self.bs_per_gpu]
        if hasattr(self, "input_B_label_cls"):
            self.input_B_label_cls = self.input_B_label_cls[: self.bs_per_gpu]
