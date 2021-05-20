import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .modules import loss
from util.iter_calculator import IterCalculator
from util.util import gaussian
from util.network_group import NetworkGroup

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--use_label_B', action='store_true', help='if true domain B has labels too')
            parser.add_argument('--rec_noise', type=float, default=0.0, help='whether to add noise to reconstruction')
            parser.add_argument('--D_noise', type=float, default=0.0, help='whether to add instance noise to discriminator inputs')
            parser.add_argument('--D_label_smooth', action='store_true', help='whether to use one-sided label smoothing with discriminator')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.opt.iter_size == 1:
            losses_G = ['G_A','G_B']

            losses_G += ['cycle_A', 'idt_A', 
                       'cycle_B', 'idt_B']            
            
            losses_D = ['D_A', 'D_B']
            
        else:
            losses_G = ['G_A_avg','G_B_avg']
            
            losses_D = ['D_A_avg', 'D_B_avg']    

            losses_G += ['cycle_A_avg', 'idt_A_avg', 
                       'cycle_B_avg', 'idt_B_avg',]

        self.loss_names_G = losses_G
        self.loss_names_D = losses_D
        
        self.loss_names = self.loss_names_G + self.loss_names_D

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.G_spectral, opt.init_type, opt.init_gain, self.gpu_ids,opt=self.opt)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.G_spectral, opt.init_type, opt.init_gain, self.gpu_ids,opt=self.opt)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.D_dropout, opt.D_spectral, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.D_dropout, opt.D_spectral, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            if opt.D_label_smooth:
                target_real_label = 0.9
            else:
                target_real_label = 1.0
            self.criterionGAN = loss.GANLoss(opt.gan_mode,target_real_label=target_real_label).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            
            self.nb_preds=int(torch.prod(torch.tensor(self.netD_A(torch.zeros([1,opt.input_nc,opt.crop_size,opt.crop_size], dtype=torch.float,device=self.device)).shape)))
            self.criterionContrastive = loss.ContrastiveLoss(self.nb_preds)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            if self.opt.iter_size > 1 :
                self.iter_calculator = IterCalculator(self.loss_names)
                for loss_name in self.loss_names:
                    setattr(self, "loss_" + loss_name, 0)

            self.rec_noise = opt.rec_noise
            self.D_noise = opt.D_noise
                    
            self.niter=0

            ###Making groups
            self.networks_groups = []

            self.group_G = NetworkGroup(networks_to_optimize=["netG_A","netG_B"], networks_not_to_optimize=["netD_A","netD_B"],forward_functions=["forward"],backward_functions=["compute_G_loss"],loss_names_list=["loss_names_G"],optimizer=["optimizer_G"])
            self.networks_groups.append(self.group_G)
            if self.opt.use_contrastive_loss_D:
                self.group_D = NetworkGroup(networks_to_optimize=["netD_A","netD_B"], networks_not_to_optimize=["netG_A","netG_B"],forward_functions=None,backward_functions=["compute_D_contrastive_loss"],loss_names_list=["loss_names_D"],optimizer=["optimizer_D"])
            else:
                self.group_D = NetworkGroup(networks_to_optimize=["netD_A","netD_B"], networks_not_to_optimize=["netG_A","netG_B"],forward_functions=None,backward_functions=["compute_D_loss"],loss_names_list=["loss_names_D"],optimizer=["optimizer_D"])
            
            self.networks_groups.append(self.group_D) 


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
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        if self.rec_noise > 0.0:
            self.fake_B_noisy1 = gaussian(self.fake_B, self.rec_noise)
            self.rec_A= self.netG_B(self.fake_B_noisy1)
        else:
            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        if self.rec_noise > 0.0:
            self.fake_A_noisy1 = gaussian(self.fake_A, self.rec_noise)
            self.rec_B = self.netG_A(self.fake_A_noisy1)
        else:
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
            
        if self.D_noise > 0.0:
            self.fake_B_noisy = gaussian(self.fake_B, self.D_noise)
            self.real_A_noisy = gaussian(self.real_A, self.D_noise)
            self.fake_A_noisy = gaussian(self.fake_A, self.D_noise)
            self.real_B_noisy = gaussian(self.real_B, self.D_noise)


    def compute_D_loss_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def compute_D_A_loss(self):
        if self.D_noise > 0.0:
            fake_B = self.fake_B_pool.query(self.fake_B_noisy)
            self.loss_D_A = self.compute_D_loss_basic(self.netD_A, self.real_B_noisy, fake_B)
        else:
            fake_B = self.fake_B_pool.query(self.fake_B)
            self.loss_D_A = self.compute_D_loss_basic(self.netD_A, self.real_B, fake_B)

    def compute_D_B_loss(self):
        if self.D_noise > 0.0:
            fake_A = self.fake_A_pool.query(self.fake_A_noisy)
            self.loss_D_B = self.compute_D_loss_basic(self.netD_B, self.real_A_noisy, fake_A)
        else:
            fake_A = self.fake_A_pool.query(self.fake_A)
            self.loss_D_B = self.compute_D_loss_basic(self.netD_B, self.real_A, fake_A)


    def compute_D_loss(self):
        """Calculate GAN loss for both discriminators"""
        self.compute_D_A_loss()
        self.compute_D_B_loss()
        self.loss_D = self.loss_D_A + self.loss_D_B
        (self.loss_D/self.opt.iter_size).backward()
        
    def compute_G_loss(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        current_batch_size=self.get_current_batch_size()
        if self.opt.use_contrastive_loss_D:
            self.loss_G_A = self.criterionContrastive(-self.netD_A(self.real_B),-self.netD_A(self.fake_B))
        else:
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        
        if self.opt.use_contrastive_loss_D:
            self.loss_G_B = self.criterionContrastive(-self.netD_B(self.real_A),-self.netD_B(self.fake_A))                                                    
        else:
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        (self.loss_G/self.opt.iter_size).backward()
        
    def compute_D_contrastive_loss(self):
        """Calculate contrastive GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_B = self.fake_B.detach()
        fake_A = self.fake_A.detach()

        pred_fake_A = self.netD_B(fake_A)
        pred_fake_B = self.netD_A(fake_B)
        # Real
        pred_real_A = self.netD_B(self.real_A)
        pred_real_B = self.netD_A(self.real_B)
        
        loss_D_real_A = self.criterionContrastive(pred_real_A,pred_fake_A)
        loss_D_fake_B = self.criterionContrastive(-pred_fake_A,-pred_real_A)
        
        loss_D_real_B = self.criterionContrastive(pred_real_B,pred_fake_B)
        loss_D_fake_A = self.criterionContrastive(-pred_fake_B,-pred_real_B)
        
        # combine loss and calculate gradients
        self.loss_D_A = (loss_D_fake_A + loss_D_real_A) * 0.5
        self.loss_D_B = (loss_D_fake_B + loss_D_real_B) * 0.5

        self.loss_D = self.loss_D_A + self.loss_D_B

        (self.loss_D/self.opt.iter_size).backward()
