import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from .modules import loss
from util.iter_calculator import IterCalculator
from util.network_group import NetworkGroup
import itertools
from util.image_pool import ImagePool

class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--use_label_B', action='store_true', help='if true domain B has labels too')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt,rank):
        BaseModel.__init__(self, opt,rank)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        losses_G = ['G_GAN', 'G', 'NCE']
        losses_D= ['D_tot','D']
        if opt.nce_idt and self.isTrain:
            losses_G += ['NCE_Y']
        if opt.netD_global != "none":
            losses_D += ['D_global']
            losses_G += ['G_GAN_global']

        self.loss_names_G = losses_G
        self.loss_names_D = losses_D
        
        self.loss_names = self.loss_names_G + self.loss_names_D
        
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B']

        
        
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            visual_names_B += ['idt_B']
        self.visual_names = [visual_names_A,visual_names_B]

        if self.opt.diff_aug_policy != '':
            self.visual_names.append(['fake_B_aug'])
            self.visual_names.append(['real_B_aug'])
        
        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
            if opt.netD_global != "none":
                self.model_names += ['D_global']

            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
        
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG =networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.G_spectral, opt.init_type, opt.init_gain, self.gpu_ids,opt=self.opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netF.set_device(self.device)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.D_dropout, opt.D_spectral, opt.init_type, opt.init_gain,opt.no_antialias, self.gpu_ids,opt)
            if opt.netD_global != "none":
                self.netD_global = networks.define_D(opt.output_nc, opt.ndf, opt.netD_global,
                                            opt.n_layers_D, opt.norm, opt.D_dropout, opt.D_spectral, opt.init_type, opt.init_gain,opt.no_antialias, self.gpu_ids,opt)


            # define loss functions
            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            if opt.netD_global== "none":
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            else:
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters(),self.netD_global.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if self.opt.iter_size > 1 :
                self.iter_calculator = IterCalculator(self.loss_names)
                for i,cur_loss in enumerate(self.loss_names):
                    self.loss_names[i] = cur_loss + '_avg'
                    setattr(self, "loss_" + self.loss_names[i], 0)

            self.niter=0

            if opt.netD_global == "none":
                self.loss_D_global=0
                self.loss_G_GAN_global=0
 
            ###Making groups
            discriminators=["netD"]
            if opt.netD_global != "none":
                discriminators += ["netD_global"]
                self.D_global_loss=loss.DiscriminatorGANLoss(opt,self.netD_global,self.device)
            self.networks_groups = []

            self.group_G = NetworkGroup(networks_to_optimize=["G","F"],forward_functions=["forward"],backward_functions=["compute_G_loss"],loss_names_list=["loss_names_G"],optimizer=["optimizer_G"],loss_backward=["loss_G"])
            self.networks_groups.append(self.group_G)

            self.group_D = NetworkGroup(networks_to_optimize=["D"],forward_functions=None,backward_functions=["compute_D_loss"],loss_names_list=["loss_names_D"],optimizer=["optimizer_D"],loss_backward=["loss_D_tot"])
            self.networks_groups.append(self.group_D) 


        if self.opt.use_contrastive_loss_D:
            self.D_loss=loss.DiscriminatorContrastiveLoss(opt,self.netD,self.device)
        else:
            self.D_loss=loss.DiscriminatorGANLoss(opt,self.netD,self.device)

    def set_input_first_gpu(self,data):
        self.set_input(data)
        self.bs_per_gpu = self.real_A.size(0) #// max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:self.bs_per_gpu]
        self.real_B = self.real_B[:self.bs_per_gpu]
            
    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        for net in self.model_names:
            setattr(self,"net" + net,getattr(self,"net" + net).to(self.device))

        self.set_input_first_gpu(data)
        if self.opt.isTrain:
            self.optimize_parameters()
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

        for optimizer in self.optimizers:
            optimizer.zero_grad()        

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for both discriminators"""
        self.loss_D = self.compute_D_loss_generic(self.netD,"B",self.D_loss)

        if self.opt.netD_global != "none":
            self.loss_D_global = self.compute_D_loss_generic(self.netD_global,"B",self.D_global_loss)
        
        self.loss_D_tot = self.loss_D + self.loss_D_global

                                        
    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            self.loss_G_GAN = self.compute_G_loss_GAN_generic(self.netD,"B",self.D_loss)
            if self.opt.netD_global != "none":
                self.loss_G_GAN_global = self.compute_G_loss_GAN_generic(self.netD_global,"B",self.D_global_loss)

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN_global + loss_NCE_both

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, True)

        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k,current_batch=src.shape[0]) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
