import os
import copy
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
from .modules.utils import get_scheduler
from torchviz import make_dot

# for FID
from data.base_dataset import get_transform
from .modules.fid.pytorch_fid.fid_score import (
    _compute_statistics_of_path,
    calculate_frechet_distance,
)
from util.util import save_image, tensor2im
import numpy as np
from util.diff_aug import DiffAugment

# for D accuracy
from util.image_pool import ImagePool
import torch.nn.functional as F

# For D loss computing
from .modules import loss
from util.discriminator import DiscriminatorInfo


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
        if hasattr(opt, "fs_light"):
            self.fs_light = opt.fs_light
        self.device = torch.device(
            "cuda:{}".format(self.gpu_ids[rank])
            if self.gpu_ids
            else torch.device("cpu")
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

        self.fake_A_pool = ImagePool(
            opt.train_pool_size
        )  # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(
            opt.train_pool_size
        )  # create image buffer to store previously generated images
        self.real_A_pool = ImagePool(
            opt.train_pool_size
        )  # create image buffer to store previously generated images
        self.real_B_pool = ImagePool(
            opt.train_pool_size
        )  # create image buffer to store previously generated images

        if rank == 0 and (opt.train_compute_fid or opt.train_compute_fid_val):
            self.transform = get_transform(opt, grayscale=(opt.model_input_nc == 1))
            dims = 2048
            batch = 1
            self.netFid = networks.define_inception(self.gpu_ids[0], dims)

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

        if opt.dataaug_diff_aug_policy != "":
            self.diff_augment = DiffAugment(
                opt.dataaug_diff_aug_policy, opt.dataaug_diff_aug_proba
            )

        self.niter = 0

        self.objects_to_update = []

        if self.opt.dataaug_APA:
            self.visual_names.append(["APA_img"])

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
        else:
            self.onnx_opset_version = 9

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

    def set_input(self, input):

        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A_with_context = input["A"].to(self.device)
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

        self.real_B_with_context = input["B"].to(self.device)

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

        self.image_paths = input["A_img_paths"]

    def set_input_temporal(self, input_temporal):

        self.temporal_real_A_with_context = input_temporal["A"]
        self.temporal_real_B_with_context = input_temporal["B"]

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
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real_A_pool.query(self.real_A)
        self.real_B_pool.query(self.real_B)

        if self.opt.output_display_G_attention_masks:
            images, attentions, outputs = self.netG_A.get_attention_masks(self.real_A)
            for i, cur_mask in enumerate(attentions):
                setattr(self, "attention_" + str(i), cur_mask)

            for i, cur_output in enumerate(outputs):
                setattr(self, "output_" + str(i), cur_output)

            for i, cur_image in enumerate(images):
                setattr(self, "image_" + str(i), cur_image)

        if self.opt.data_online_context_pixels > 0:

            bs = self.get_current_batch_size()
            self.mask_context = torch.ones(
                [
                    bs,
                    self.opt.model_input_nc,
                    self.opt.data_crop_size + self.margin,
                    self.opt.data_crop_size + self.margin,
                ],
                device=self.device,
            )

            self.mask_context[
                :,
                :,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
            ] = torch.zeros(
                [
                    bs,
                    self.opt.model_input_nc,
                    self.opt.data_crop_size,
                    self.opt.data_crop_size,
                ],
                device=self.device,
            )

            self.mask_context_vis = torch.nn.functional.interpolate(
                self.mask_context, size=self.real_A.shape[2:]
            )[:, 0]

        if any("temporal" in D_name for D_name in self.opt.D_netDs):
            self.compute_temporal_fake(objective_domain="B")

            if hasattr(self, "netG_B"):
                self.compute_temporal_fake(objective_domain="A")

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
        if self.rank == 0:
            self.print_networks(opt.output_verbose)

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

                if len(self.gpu_ids) > 1 and torch.cuda.is_available():
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

                input_nc = self.opt.model_input_nc
                if self.opt.model_multimodal:
                    input_nc += self.opt.train_mm_nz

                dummy_input = torch.randn(
                    1,
                    input_nc,
                    self.opt.data_crop_size,
                    self.opt.data_crop_size,
                    device=self.device,
                )

                # onnx
                export_path_onnx = save_path.replace(".pth", ".onnx")

                torch.onnx.export(
                    net,
                    dummy_input,
                    export_path_onnx,
                    verbose=False,
                    opset_version=self.onnx_opset_version,
                )

                # jit
                if not "segformer" in self.opt.G_netG:
                    export_path_jit = save_path.replace(".pth", ".pt")
                    jit_model = torch.jit.trace(net, dummy_input)
                    jit_model.save(export_path_jit)

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

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(
                    "[Network %s] Total number of parameters : %.3f M"
                    % (name, num_params / 1e6)
                )
        print("-----------------------------------------------")

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
                temp = net(
                    self.netG_A(self.real_A).detach()
                )  # decoders take w+ in input
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

    def compute_fid(self, n_epoch, n_iter):
        dims = 2048
        batch = 1
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

    def compute_D_accuracy_pred(self, real, fake, netD):
        pred_real = (netD(real).flatten() > 0.5) * 1
        pred_fake = (netD(fake).flatten() > 0.5) * 1

        FP = F.l1_loss(
            pred_fake, torch.zeros(pred_real.shape).to(self.device), reduction="sum"
        )
        TP = F.l1_loss(
            pred_real, torch.zeros(pred_real.shape).to(self.device), reduction="sum"
        )
        TN = F.l1_loss(
            pred_fake, torch.ones(pred_real.shape).to(self.device), reduction="sum"
        )
        FN = F.l1_loss(
            pred_real, torch.ones(pred_real.shape).to(self.device), reduction="sum"
        )

        prec_real = TP / (TP + FP)
        prec_fake = TN / (TN + FN)
        rec_real = TP / (TP + FN)
        rec_fake = TN / (TN + FP)
        acc = (TP + TN) / (TP + TN + FN + FP)

        return prec_real, prec_fake, rec_real, rec_fake, acc

    def compute_fake_val(self, imgs, netG):
        return_imgs = []
        for img in imgs:
            return_imgs.append(netG(img.unsqueeze(0)))
        return torch.cat(return_imgs)

    def compute_D_accuracy(self):
        real_A = torch.cat(self.real_A_pool.get_all())
        real_B = torch.cat(self.real_B_pool.get_all())
        if hasattr(self, "netD_A"):
            fake_A = self.compute_fake_val(real_B, self.netG_B)
            (
                self.prec_real_A,
                self.prec_fake_A,
                self.rec_real_A,
                self.rec_fake_A,
                self.acc_A,
            ) = self.compute_D_accuracy_pred(real_A, fake_A, self.netD_A)

            fake_A_val = self.compute_fake_val(self.real_B_val, self.netG_B)
            (
                self.prec_real_A_val,
                self.prec_fake_A_val,
                self.rec_real_A_val,
                self.rec_fake_A_val,
                self.acc_A_val,
            ) = self.compute_D_accuracy_pred(self.real_A_val, fake_A_val, self.netD_A)

        if hasattr(self, "netD_B") or hasattr(self, "netD"):
            if hasattr(self, "netD_B"):
                netD = self.netD_B
                netG = self.netG_B
            elif hasattr(self, "netD"):
                netD = self.netD
                netG = self.netG

            fake_B = self.compute_fake_val(real_A, netG)
            (
                self.prec_real_B,
                self.prec_fake_B,
                self.rec_real_B,
                self.rec_fake_B,
                self.acc_B,
            ) = self.compute_D_accuracy_pred(real_B, fake_B, netD)

            fake_B_val = self.compute_fake_val(self.real_B_val, netG)
            (
                self.prec_real_B_val,
                self.prec_fake_B_val,
                self.rec_real_B_val,
                self.rec_fake_B_val,
                self.acc_B_val,
            ) = self.compute_D_accuracy_pred(self.real_A_val, fake_B_val, netD)

    def get_current_D_accuracies(self):
        accuracies = OrderedDict()
        names = []
        if hasattr(self, "netD_A"):
            names += ["acc_A", "prec_real_A", "prec_fake_A", "rec_real_A", "rec_fake_A"]
            names += [
                "acc_A_val",
                "prec_real_A_val",
                "prec_fake_A_val",
                "rec_real_A_val",
                "rec_fake_A_val",
            ]
        if hasattr(self, "netD_B") or hasattr(self, "netD"):
            names += ["acc_B", "prec_real_B", "prec_fake_B", "rec_real_B", "rec_fake_B"]
            names += [
                "acc_B_val",
                "prec_real_B_val",
                "prec_fake_B_val",
                "rec_real_B_val",
                "rec_fake_B_val",
            ]
        for name in names:
            if isinstance(name, str):
                accuracies[name] = float(
                    getattr(self, name)
                )  # float(...) works for both scalar tensor and float number
        return accuracies

    def get_current_APA_prob(self):
        current_APA_prob = OrderedDict()
        current_APA_prob["APA_p"] = float(self.D_loss.adaptive_pseudo_augmentation_p)
        current_APA_prob["APA_adjust"] = float(self.D_loss.adjust)

        return current_APA_prob

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
                for forward in group.forward_functions:
                    getattr(self, forward)()

            for backward in group.backward_functions:
                getattr(self, backward)()

            for loss in group.loss_backward:
                (getattr(self, loss) / self.opt.train_iter_size).backward(
                    retain_graph=True
                )

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

    def compute_D_loss_generic(
        self, netD, domain_img, loss, real_name=None, fake_name=None
    ):
        noisy = ""
        if self.opt.dataaug_D_noise > 0.0:
            noisy = "_noisy"

        context = ""
        if self.opt.data_online_context_pixels > 0:
            context = "_with_context"

        if fake_name is None:
            fake = getattr(self, "fake_" + domain_img + "_pool").query(
                getattr(self, "fake_" + domain_img + context + noisy)
            )
        else:
            fake = getattr(self, fake_name)

        if self.opt.dataaug_APA:
            fake_2 = getattr(self, "fake_" + domain_img + "_pool").get_random(
                fake.shape[0]
            )
            self.APA_img = fake_2
        else:
            fake_2 = None

        if real_name is None:
            real = getattr(self, "real_" + domain_img + context + noisy)
        else:
            real = getattr(self, real_name)

        loss = loss.compute_loss_D(netD, real, fake, fake_2)
        return loss

    def compute_D_loss(self):
        """Calculate GAN loss for discriminators"""

        self.loss_D_tot = 0

        for discriminator in self.discriminators:
            if self.niter % discriminator.compute_every == 0:
                domain = discriminator.name.split("_")[1]
                netD = getattr(self, discriminator.name)
                loss = getattr(self, discriminator.loss_type)
                if discriminator.fake_name is not None:
                    fake_name = discriminator.fake_name + "_" + domain
                if discriminator.real_name is not None:
                    real_name = discriminator.real_name + "_" + domain
                else:
                    fake_name = None
                    real_name = None

                loss_value = self.compute_D_loss_generic(
                    netD,
                    domain,
                    loss,
                    fake_name=fake_name,
                    real_name=real_name,
                )
                loss_name = "loss_" + discriminator.loss_name_D

                setattr(
                    self,
                    loss_name,
                    loss_value,
                )

                self.loss_D_tot += loss_value

    def compute_G_loss_GAN_generic(
        self, netD, domain_img, loss, real_name=None, fake_name=None
    ):
        context = ""
        if self.opt.data_online_context_pixels > 0:
            context = "_with_context"

        if fake_name is None:
            fake = getattr(self, "fake_" + domain_img + context)
        else:
            fake = getattr(self, fake_name)
        if real_name is None:
            real = getattr(self, "real_" + domain_img + context)
        else:
            real = getattr(self, real_name)

        if hasattr(self, "diff_augment"):

            real = self.diff_augment(real)
            fake = self.diff_augment(fake)

            if fake_name is None:
                setattr(self, "fake_" + domain_img + "_aug", fake)
            else:
                setattr(self, fake_name + "_aug", fake)

            if real_name is None:
                setattr(self, "real_" + domain_img + "_aug", real)
            else:
                setattr(self, real_name + "_aug", real)

        loss = loss.compute_loss_G(netD, real, fake)
        return loss

    def compute_G_loss(self):
        """Calculate GAN losses for generator(s)"""

        self.loss_G_tot = 0

        # GAN losses
        for discriminator in self.discriminators:
            if self.niter % discriminator.compute_every == 0:
                domain = discriminator.name.split("_")[1]
                netD = getattr(self, discriminator.name)
                loss = getattr(self, discriminator.loss_type)
                if discriminator.fake_name is not None:
                    fake_name = discriminator.fake_name + "_" + domain
                if discriminator.real_name is not None:
                    real_name = discriminator.real_name + "_" + domain
                else:
                    fake_name = None
                    real_name = None

                loss_value = self.compute_G_loss_GAN_generic(
                    netD,
                    domain,
                    loss,
                    fake_name=fake_name,
                    real_name=real_name,
                )

                loss_name = "loss_" + discriminator.loss_name_G

                setattr(
                    self,
                    loss_name,
                    loss_value,
                )

                self.loss_G_tot += loss_value

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

    def set_discriminators_info(self):
        self.discriminators = []

        for discriminator_name in self.discriminators_names:

            loss_calculator_name = "D_" + discriminator_name + "_loss_calculator"

            if "temporal" in discriminator_name or "projected" in discriminator_name:
                train_gan_mode = "projected"
            elif "vision_aided" in discriminator_name:
                train_gan_mode = "vanilla"
            else:
                train_gan_mode = self.opt.train_gan_mode

            if "temporal" in discriminator_name:
                setattr(
                    self,
                    loss_calculator_name,
                    loss.DiscriminatorGANLoss(
                        netD=getattr(self, "net" + discriminator_name),
                        device=self.device,
                        dataaug_APA_p=self.opt.dataaug_APA_p,
                        dataaug_APA_target=self.opt.dataaug_APA_target,
                        train_batch_size=self.opt.train_batch_size,
                        dataaug_APA_nimg=self.opt.dataaug_APA_nimg,
                        dataaug_APA_every=self.opt.dataaug_APA_every,
                        dataaug_D_label_smooth=self.opt.dataaug_D_label_smooth,
                        train_gan_mode=train_gan_mode,
                        dataaug_APA=self.opt.dataaug_APA,
                    ),
                )

                fake_name = "temporal_fake"
                real_name = "temporal_real"
                compute_every = self.opt.D_temporal_every

            else:
                fake_name = None
                real_name = None
                compute_every = 1

                if self.opt.train_use_contrastive_loss_D:
                    loss_calculator = (
                        loss.DiscriminatorContrastiveLoss(
                            netD=getattr(self, "net" + discriminator_name),
                            device=self.device,
                            dataaug_APA_p=self.opt.dataaug_APA_p,
                            dataaug_APA_target=self.opt.dataaug_APA_target,
                            train_batch_size=self.opt.train_batch_size,
                            dataaug_APA_nimg=self.opt.dataaug_APA_nimg,
                            dataaug_APA_every=self.opt.dataaug_APA_every,
                            model_input_nc=self.opt.model_input_nc,
                            train_crop_size=train_crop_size,
                            dataaug_APA=self.opt.dataaug_APA,
                        ),
                    )
                else:
                    loss_calculator = loss.DiscriminatorGANLoss(
                        netD=getattr(self, "net" + discriminator_name),
                        device=self.device,
                        dataaug_APA_p=self.opt.dataaug_APA_p,
                        dataaug_APA_target=self.opt.dataaug_APA_target,
                        train_batch_size=self.opt.train_batch_size,
                        dataaug_APA_nimg=self.opt.dataaug_APA_nimg,
                        dataaug_APA_every=self.opt.dataaug_APA_every,
                        dataaug_D_label_smooth=self.opt.dataaug_D_label_smooth,
                        train_gan_mode=train_gan_mode,
                        dataaug_APA=self.opt.dataaug_APA,
                    )

            setattr(
                self,
                loss_calculator_name,
                loss_calculator,
            )

            self.objects_to_update.append(getattr(self, loss_calculator_name))

            self.discriminators.append(
                DiscriminatorInfo(
                    name="net" + discriminator_name,
                    loss_name_D="D_GAN_" + discriminator_name,
                    loss_name_G="G_GAN_" + discriminator_name,
                    loss_type=loss_calculator_name,
                    fake_name=fake_name,
                    real_name=real_name,
                    compute_every=compute_every,
                )
            )

    # multimodal input latent vector
    def get_z_random(self, batch_size, nz, random_type="gauss"):
        if random_type == "uni":
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == "gauss":
            z = torch.randn(batch_size, nz)
        return z.detach().to(self.device)
