import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from .modules import loss
import torch.nn.functional as F
from util.util import gaussian
from util.iter_calculator import IterCalculator

class CUTSemanticModel(BaseModel):
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
        parser.add_argument('--train_cls_B', action='store_true', help='if true cls will be trained not only on domain A but also on domain B')
        parser.add_argument('--cls_template', help='classifier/regressor model type, from torchvision (resnet18, ...), default is custom simple model', default='basic')
        parser.add_argument('--cls_pretrained', action='store_true', help='whether to use a pretrained model, available for non "basic" model only')    
        parser.add_argument('--lr_f_s', type=float, default=0.0002, help='f_s learning rate')
        parser.add_argument('--D_noise', type=float, default=0.0, help='whether to add instance noise to discriminator inputs')
        parser.add_argument('--contrastive_noise', type=float, default=0.0, help='noise on constrastive classifier')
        parser.add_argument('--regression', action='store_true', help='if true cls will be a regressor and not a classifier')
        parser.add_argument('--lambda_sem', type=float, default=1.0, help='weight for semantic loss')
        parser.add_argument('--l1_regression', action='store_true', help='if true l1 loss will be used to compute regressor loss')
        
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

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>

        if self.opt.iter_size == 1:
            losses_G = ['G_GAN', 'NCE', 'G','sem']

            if opt.nce_idt and self.isTrain:
                losses_G += ['NCE_Y']

            losses_CLS = ['CLS']            
            
            losses_D = ['D_real', 'D_fake']
            
        else:
            losses_G = ['G_GAN_avg', 'NCE_avg', 'G_avg','sem_avg']

            if opt.nce_idt and self.isTrain:
                losses_G += ['NCE_Y_avg']

            losses_CLS = ['CLS_avg']            
            
            losses_D = ['D_real_avg', 'D_fake_avg']

        self.loss_names_G = losses_G
        self.loss_names_CLS = losses_CLS
        self.loss_names_D = losses_D
        
        self.loss_names = self.loss_names_G + self.loss_names_CLS + self.loss_names_D

        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.visual_names += ['idt_B']
            
        if self.isTrain:
            self.model_names = ['G', 'F', 'D','CLS']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG =networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.G_spectral, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.D_dropout, opt.D_spectral, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netCLS = networks.define_C(opt.output_nc, opt.ndf,opt.crop_size,
                                            init_type=opt.init_type, init_gain=opt.init_gain,
                                            gpu_ids=self.gpu_ids, nclasses=opt.semantic_nclasses)
 
            # define loss functions
            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            self.criterionCLS = torch.nn.modules.CrossEntropyLoss()

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_CLS = torch.optim.Adam(self.netCLS.parameters(), lr=opt.lr_f_s, betas=(opt.beta1, 0.999))
            
            if opt.regression:
                if opt.l1_regression:
                    self.criterionCLS = torch.nn.L1Loss()
                else:
                    self.criterionCLS = torch.nn.modules.MSELoss()
            else:
                self.criterionCLS = torch.nn.modules.CrossEntropyLoss()
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if self.opt.iter_size > 1 :
                self.iter_calculator = IterCalculator(self.loss_names)
                for loss_name in self.loss_names:
                    setattr(self, "loss_" + loss_name, 0)
            
            self.niter=0


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.input_A_label=self.input_A_label[:bs_per_gpu]
        if hasattr(self,'input_B_label'):
            self.input_B_label=self.input_B_label[:bs_per_gpu]
        
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_CLS_loss().backward()                  # calculate gradients for CLS
            self.compute_G_loss().backward()                   # calculate gradients for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

        for optimizer in self.optimizers:
            optimizer.zero_grad()
        

    def optimize_parameters(self):

        self.niter = self.niter +1

        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netG, True)
        self.set_requires_grad(self.netF, True)
        self.set_requires_grad(self.netCLS, False)

        # forward
        self.forward()
        
        self.loss_G = self.compute_G_loss()
        (self.loss_G/self.opt.iter_size).backward()
        self.compute_step(self.optimizer_G,self.loss_names_G)
        if self.opt.netF == 'mlp_sample' and self.niter % self.opt.iter_size == 0:
            self.optimizer_F.step()
            self.optimizer_F.zero_grad()
 
        
        # update D
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netF, False)
        self.set_requires_grad(self.netCLS, False)
        self.loss_D = self.compute_D_loss()
        (self.loss_D/self.opt.iter_size).backward()
        self.compute_step(self.optimizer_D,self.loss_names_D)           

        # update CLS
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netF, False)
        self.set_requires_grad(self.netCLS, True)
        self.loss_CLS = self.compute_CLS_loss()
        (self.loss_CLS/self.opt.iter_size).backward()
        self.compute_step(self.optimizer_CLS,self.loss_names_CLS)


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
        if 'A_label' in input :
            if not self.opt.regression:
                self.input_A_label = input['A_label'].to(self.device)
            else:
                self.input_A_label = input['A_label'].to(torch.float).to(device=self.device)
        if self.opt.train_cls_B and 'B_label' in input:
            if not self.opt.regression:
                self.input_B_label = input['B_label'].to(self.device)
            else:
                self.input_B_label = input['B_label'].to(torch.float).to(device=self.device)
            

        
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

        d = 1
        self.pred_real_A = self.netCLS(self.real_A)
        if not self.opt.regression:
               _,self.gt_pred_A = self.pred_real_A.max(1)
        
        self.pred_fake_B = self.netCLS(self.fake_B)
        if not self.opt.regression:
            _,self.pfB = self.pred_fake_B.max(1)
            
        
    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        if self.opt.D_noise:
            fake = self.fake_B_noisy.detach()
        else:
            fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        if self.opt.D_noise:
            real_B = self.real_B_noisy
        else:
            real_B = self.real_B
        self.pred_real = self.netD(real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
        if not self.opt.regression:
            self.loss_sem = self.criterionCLS(self.pred_fake_B, self.input_A_label)
        else:
            self.loss_sem = self.criterionCLS(self.pred_fake_B.squeeze(1), self.input_A_label)
        if self.loss_CLS.detach().item() > self.opt.semantic_threshold:
            self.loss_sem = 0 * self.loss_sem
        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_sem
        
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            if self.opt.contrastive_noise>0.0:
                f_q=gaussian(f_q,self.opt.contrastive_noise)
                f_k=gaussian(f_k,self.opt.contrastive_noise)
            loss = crit(f_q, f_k,current_batch=src.shape[0]) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    
    def compute_CLS_loss(self):
        label_A = self.input_A_label
        # forward only real source image through semantic classifier
        pred_A = self.netCLS(self.real_A)
        if not self.opt.regression:
            self.loss_CLS = self.opt.lambda_sem * self.criterionCLS(pred_A, label_A)
        else:
            self.loss_CLS = self.opt.lambda_sem * self.criterionCLS(pred_A.squeeze(1), label_A)
        if self.opt.train_cls_B:
            label_B = self.input_B_label
            pred_B = self.netCLS(self.real_B)
            if not self.opt.regression:
                self.loss_CLS += self.opt.lambda_sem * self.criterionCLS(pred_B, label_B)
            else:
                self.loss_CLS += self.opt.lambda_sem * self.criterionCLS(pred_B.squeeze(1), label_B)
        
        return self.loss_CLS
