import torch
import itertools
from util.image_pool import ImagePool
from util.losses import L1_Charbonnier_loss
from util.madgrad import MADGRAD
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .modules import loss
from util.util import gaussian
from util.iter_calculator import IterCalculator

class CycleGANSemanticMaskModel(BaseModel):
    def name(self):
        return 'CycleGANSemanticMaskModel'

    # new, copied from cyclegansemantic model
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
            parser.add_argument('--out_mask', action='store_true', help='use loss out mask')
            parser.add_argument('--lambda_out_mask', type=float, default=10.0, help='weight for loss out mask')
            parser.add_argument('--loss_out_mask', type=str, default='L1', help='loss mask')
            parser.add_argument('--charbonnier_eps', type=float, default=1e-6, help='Charbonnier loss epsilon value')
            parser.add_argument('--disc_in_mask', action='store_true', help='use in-mask discriminator')
            parser.add_argument('--train_f_s_B', action='store_true', help='if true f_s will be trained not only on domain A but also on domain B')
            parser.add_argument('--fs_light',action='store_true', help='whether to use a light (unet) network for f_s')
            parser.add_argument('--lr_f_s', type=float, default=0.0002, help='f_s learning rate')
            parser.add_argument('--D_noise', type=float, default=0.0, help='whether to add instance noise to discriminator inputs')
            parser.add_argument('--D_label_smooth', action='store_true', help='whether to use one-sided label smoothing with discriminator')
            parser.add_argument('--rec_noise', type=float, default=0.0, help='whether to add noise to reconstruction')
            parser.add_argument('--nb_attn', type=int, default=10, help='number of attention masks')
            parser.add_argument('--nb_mask_input', type=int, default=1, help='number of attention masks which will be applied on the input image')
            parser.add_argument('--lambda_sem', type=float, default=1.0, help='weight for semantic loss')
            parser.add_argument('--madgrad',action='store_true',help='if true madgrad optim will be used')

        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        if not hasattr(opt, 'disc_in_mask'):
            opt.disc_in_mask = False
        if not hasattr(opt, 'out_mask'):
            opt.out_mask = False
        if not hasattr(opt, 'nb_attn'):
            opt.nb_attn = 10
        if not hasattr(opt, 'nb_mask_input'):
            opt.nb_mask_input = 1
        if not hasattr(opt, 'fs_light'):
            opt.fs_light = False
            
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        if self.opt.iter_size == 1:
            losses_G = ['G_A','G_B']

            if opt.out_mask:
                losses_G += ['out_mask_AB','out_mask_BA']

            losses_G += ['cycle_A', 'idt_A', 
                       'cycle_B', 'idt_B', 
                       'sem_AB', 'sem_BA']

            losses_f_s = ['f_s']
            
            
            losses_D = ['D_A', 'D_B']
            if opt.disc_in_mask:
                losses_D += ['D_A_mask', 'D_B_mask']
            
        else:
            losses_G = ['G_A_avg','G_B_avg']
            
            if opt.disc_in_mask:
                losses_G += ['D_A_mask_avg', 'D_B_mask_avg']

            losses_D = ['D_A_avg', 'D_B_avg']
            
            if opt.out_mask:
                losses_D += ['out_mask_AB_avg','out_mask_BA_avg']

            losses_f_s = ['f_s_avg']

            losses_G += ['cycle_A_avg', 'idt_A_avg', 
                       'cycle_B_avg', 'idt_B_avg', 
                       'sem_AB_avg', 'sem_BA_avg']

        self.loss_names_G = losses_G
        self.loss_names_f_s = losses_f_s
        self.loss_names_D = losses_D
        
        self.loss_names = self.loss_names_G + self.loss_names_f_s + self.loss_names_D
        
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']

        visual_names_B = ['real_B', 'fake_A', 'rec_B']

        if self.isTrain and self.opt.lambda_identity > 0.0:
           visual_names_A.append('idt_B')
           visual_names_B.append('idt_A') # inverted for original

        visual_names_seg_A = ['input_A_label','gt_pred_A','pfB_max']

        
        visual_names_seg_B = ['gt_pred_B','pfA_max']
        
        visual_names_out_mask = ['real_A_out_mask','fake_B_out_mask']

        if hasattr(self, 'input_B_label') and len(self.input_B_label) > 0: # XXX: model is created after dataset is populated so this check stands
            visual_names_seg_B.append('input_B_label')
            visual_names_out_mask.append('real_B_out_mask')
            visual_names_out_mask.append('fake_A_out_mask')
        
        visual_names_mask = ['fake_B_mask','fake_A_mask']

        visual_names_mask_in = ['real_B_mask','fake_B_mask','real_A_mask','fake_A_mask',
                                'real_B_mask_in','fake_B_mask_in','real_A_mask_in','fake_A_mask_in']
        
        self.visual_names = visual_names_A + visual_names_B + visual_names_seg_A + visual_names_seg_B 

        if opt.out_mask :
            self.visual_names += visual_names_out_mask

        if opt.disc_in_mask:
            self.visual_names += visual_names_mask_in
            
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'f_s']
            if opt.disc_in_mask:
                self.model_names += ['D_A_mask', 'D_B_mask']
            self.model_names += ['D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.netG, opt.norm, 
                                        not opt.no_dropout, opt.G_spectral, opt.init_type, opt.init_gain, self.gpu_ids,nb_attn = opt.nb_attn,nb_mask_input=opt.nb_mask_input)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.netG, opt.norm, 
                                        not opt.no_dropout, opt.G_spectral, opt.init_type, opt.init_gain, self.gpu_ids,nb_attn = opt.nb_attn,nb_mask_input=opt.nb_mask_input)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.netD,
                                            opt.n_layers_D, opt.norm, opt.D_dropout, opt.D_spectral,
                                            opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.netD,
                                            opt.n_layers_D, opt.norm, opt.D_dropout, opt.D_spectral,
                                            opt.init_type, opt.init_gain, self.gpu_ids)
            if opt.disc_in_mask:
                self.netD_A_mask = networks.define_D(opt.output_nc, opt.ndf,
                                                     opt.netD,
                                                     opt.n_layers_D, opt.norm, opt.D_dropout, opt.D_spectral,
                                                     opt.init_type, opt.init_gain, self.gpu_ids)
                self.netD_B_mask = networks.define_D(opt.input_nc, opt.ndf,
                                                     opt.netD,
                                                     opt.n_layers_D, opt.norm, opt.D_dropout, opt.D_spectral,
                                                     opt.init_type, opt.init_gain, self.gpu_ids)
            
        self.netf_s = networks.define_f(opt.input_nc, nclasses=opt.semantic_nclasses, 
                                        init_type=opt.init_type, init_gain=opt.init_gain,
                                        gpu_ids=self.gpu_ids, fs_light=opt.fs_light)
 
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            if opt.disc_in_mask:
                self.fake_A_pool_mask = ImagePool(opt.pool_size)
                self.fake_B_pool_mask = ImagePool(opt.pool_size)
                
            # define loss functions
            if opt.D_label_smooth:
                target_real_label = 0.9
            else:
                target_real_label = 1.0
            self.criterionGAN = loss.GANLoss(opt.gan_mode,target_real_label=target_real_label).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionf_s = torch.nn.modules.CrossEntropyLoss()
            if opt.out_mask:
                if opt.loss_out_mask == 'L1':
                    self.criterionMask = torch.nn.L1Loss()
                elif opt.loss_out_mask == 'MSE':
                    self.criterionMask = torch.nn.MSELoss()
                elif opt.loss_out_mask == 'Charbonnier':
                    self.criterionMask = L1_Charbonnier_loss(opt.charbonnier_eps)
                    
            # initialize optimizers
            if not opt.madgrad:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                if opt.disc_in_mask:
                    self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(),self.netD_B.parameters(),self.netD_A_mask.parameters(), self.netD_B_mask.parameters()),
                                                lr=opt.D_lr, betas=(opt.beta1, 0.999))
                else:    
                    self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.D_lr, betas=(opt.beta1, 0.999))
                self.optimizer_f_s = torch.optim.Adam(self.netf_s.parameters(), lr=opt.lr_f_s, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = MADGRAD(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                    lr=opt.lr)
                if opt.disc_in_mask:
                    self.optimizer_D = MADGRAD(itertools.chain(self.netD_A.parameters(),self.netD_B.parameters(),self.netD_A_mask.parameters(), self.netD_B_mask.parameters()),
                                                lr=opt.D_lr)
                else:    
                    self.optimizer_D = MADGRAD(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.D_lr)
                self.optimizer_f_s = MADGRAD(self.netf_s.parameters(), lr=opt.lr_f_s)
                
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.rec_noise = opt.rec_noise
            self.D_noise = opt.D_noise

            if self.opt.iter_size > 1 :
                self.iter_calculator = IterCalculator(self.loss_names)
                for loss_name in self.loss_names:
                    setattr(self, "loss_" + loss_name, 0)
            
            self.niter=0

            
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if 'A_label' in input :
            #self.input_A_label = input['A_label' if AtoB else 'B_label'].to(self.device)
            self.input_A_label = input['A_label'].to(self.device).squeeze(1)
            #self.input_A_label_dis = display_mask(self.input_A_label)  
        if 'B_label' in input and len(input['B_label']) > 0:
            self.input_B_label = input['B_label'].to(self.device).squeeze(1) # beniz: unused
            #self.image_paths = input['B_paths'] # Hack!! forcing the labels to corresopnd to B domain

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        d = 1

        if self.isTrain:
            if self.rec_noise > 0.0:
                self.fake_B_noisy1 = gaussian(self.fake_B, self.rec_noise)
                self.rec_A= self.netG_B(self.fake_B_noisy1)
            else:
                self.rec_A = self.netG_B(self.fake_B)
            
            self.fake_A = self.netG_B(self.real_B)
            if self.rec_noise > 0.0:
                self.fake_A_noisy1 = gaussian(self.fake_A, self.rec_noise)
                self.rec_B = self.netG_A(self.fake_A_noisy1)
            else:
                self.rec_B = self.netG_A(self.fake_A)

            self.pred_real_A = self.netf_s(self.real_A)
           
            
            self.gt_pred_A = F.log_softmax(self.pred_real_A,dim= d).argmax(dim=d)
            
            
            self.pred_real_B = self.netf_s(self.real_B)
            self.gt_pred_B = F.log_softmax(self.pred_real_B,dim=d).argmax(dim=d)
            
            self.pred_fake_A = self.netf_s(self.fake_A)
            
            self.pfA = F.log_softmax(self.pred_fake_A,dim=d)#.argmax(dim=d)
            self.pfA_max = self.pfA.argmax(dim=d)

            if hasattr(self,'criterionMask'):
                label_A = self.input_A_label
                label_A_in = label_A.unsqueeze(1)
                label_A_inv = torch.tensor(np.ones(label_A.size())).to(self.device) - label_A>0
                label_A_inv = label_A_inv.unsqueeze(1)
                #label_A_inv = torch.cat ([label_A_inv,label_A_inv,label_A_inv],1)
                
                self.real_A_out_mask = self.real_A *label_A_inv
                self.fake_B_out_mask = self.fake_B *label_A_inv

                if self.disc_in_mask:
                    self.real_A_mask_in = self.real_A * label_A_in
                    self.fake_B_mask_in = self.fake_B * label_A_in
                    self.real_A_mask = self.real_A #* label_A_in + self.real_A_out_mask
                    self.fake_B_mask = self.fake_B_mask_in + self.real_A_out_mask.float()
                    
                if self.D_noise > 0.0:
                    self.fake_B_noisy = gaussian(self.fake_B, self.D_noise)
                    self.real_A_noisy = gaussian(self.real_A, self.D_noise)
                        
                if hasattr(self, 'input_B_label') and len(self.input_B_label) > 0:
                
                    label_B = self.input_B_label
                    label_B_in = label_B.unsqueeze(1)
                    label_B_inv = torch.tensor(np.ones(label_B.size())).to(self.device) - label_B>0
                    label_B_inv = label_B_inv.unsqueeze(1)
                    
                    self.real_B_out_mask = self.real_B *label_B_inv
                    self.fake_A_out_mask = self.fake_A *label_B_inv
                    if self.disc_in_mask:
                        self.real_B_mask_in = self.real_B * label_B_in
                        self.fake_A_mask_in = self.fake_A * label_B_in
                        self.real_B_mask = self.real_B #* label_B_in + self.real_B_out_mask
                        self.fake_A_mask = self.fake_A_mask_in + self.real_B_out_mask.float()

                    if self.D_noise > 0.0:
                        self.fake_A_noisy = gaussian(self.fake_A, self.D_noise)
                        self.real_B_noisy = gaussian(self.real_B, self.D_noise)
                        
        self.pred_fake_B = self.netf_s(self.fake_B)
        self.pfB = F.log_softmax(self.pred_fake_B,dim=d)#.argmax(dim=d)
        self.pfB_max = self.pfB.argmax(dim=d)
           
    def compute_D_loss_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        return loss_D
    
    def compute_f_s_loss(self):
        #print('backward fs')
        label_A = self.input_A_label
        # forward only real source image through semantic classifier
        pred_A = self.netf_s(self.real_A) 
        self.loss_f_s = self.criterionf_s(pred_A, label_A)#.squeeze(1))
        if self.opt.train_f_s_B:
            label_B = self.input_B_label
            pred_B = self.netf_s(self.real_B) 
            self.loss_f_s += self.criterionf_s(pred_B, label_B)#.squeeze(1))

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

    def compute_D_A_mask_loss(self):
        fake_B_mask = self.fake_B_pool_mask.query(self.fake_B_mask)
        self.loss_D_A_mask = self.compute_D_loss_basic(self.netD_A_mask, self.real_B_mask, fake_B_mask)

    def compute_D_B_mask_loss(self):
        fake_A_mask = self.fake_A_pool_mask.query(self.fake_A_mask)
        self.loss_D_B_mask = self.compute_D_loss_basic(self.netD_B_mask, self.real_A_mask, fake_A_mask)

    def compute_D_A_mask_in_loss(self):
        fake_B_mask_in = self.fake_B_pool.query(self.fake_B_mask_in)
        self.loss_D_A = self.compute_D_loss_basic(self.netD_A, self.real_B_mask_in, fake_B_mask_in)

    def compute_D_B_mask_in_loss(self):
        fake_A_mask_in = self.fake_A_pool.query(self.fake_A_mask)
        self.loss_D_B = self.compute_D_loss_basic(self.netD_B, self.real_A_mask_in, fake_A_mask_in)

    def compute_D_loss(self):
        if self.disc_in_mask:
            self.compute_D_A_mask_in_loss()
            self.compute_D_B_mask_in_loss()
            self.compute_D_A_mask_loss()
            self.compute_D_B_mask_loss()
            self.loss_D = self.loss_D_A + self.loss_D_B + self.loss_D_A_mask + self.loss_D_B_mask
        else:
            self.compute_D_A_loss()      # calculate gradients for D_A
            self.compute_D_B_loss()      # calculate gradients for D_B
            self.loss_D = self.loss_D_A + self.loss_D_B


    def compute_G_loss(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_sem = self.opt.lambda_sem
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)

            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        if self.disc_in_mask:
            self.loss_G_A_mask = self.criterionGAN(self.netD_A(self.fake_B_mask_in), True)
            self.loss_G_B_mask = self.criterionGAN(self.netD_B(self.fake_A_mask_in), True)
            self.loss_G_A = self.criterionGAN(self.netD_A_mask(self.fake_B_mask), True)
            self.loss_G_B = self.criterionGAN(self.netD_B_mask(self.fake_A_mask), True)
        else:
            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss standard cyclegan
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        if self.disc_in_mask:
            self.loss_G += self.loss_G_A_mask + self.loss_G_B_mask

        # semantic loss AB
        self.loss_sem_AB = lambda_sem*self.criterionf_s(self.pfB, self.input_A_label)

        # semantic loss BA
        if hasattr(self, 'input_B_label'):
            self.loss_sem_BA = lambda_sem*self.criterionf_s(self.pfA, self.input_B_label)#.squeeze(1))
        else:
            self.loss_sem_BA = lambda_sem*self.criterionf_s(self.pfA, self.gt_pred_B)#.squeeze(1))
        
        # only use semantic loss when classifier has reasonably low loss
        #if True:
        if not hasattr(self, 'loss_f_s') or self.loss_f_s.detach().item() > self.opt.semantic_threshold:
            self.loss_sem_AB = 0 * self.loss_sem_AB 
            self.loss_sem_BA = 0 * self.loss_sem_BA 
        self.loss_G += self.loss_sem_BA + self.loss_sem_AB

        lambda_out_mask = self.opt.lambda_out_mask

        if hasattr(self,'criterionMask'):
            self.loss_out_mask_AB = self.criterionMask( self.real_A_out_mask, self.fake_B_out_mask) * lambda_out_mask
            if hasattr(self, 'input_B_label') and len(self.input_B_label) > 0:
                self.loss_out_mask_BA = self.criterionMask( self.real_B_out_mask, self.fake_A_out_mask) * lambda_out_mask
            else:
                self.loss_out_mask_BA = 0
            self.loss_G += self.loss_out_mask_AB + self.loss_out_mask_BA

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.niter = self.niter +1
        
        # G_A and G_B
        if self.disc_in_mask:
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_A_mask, self.netD_B_mask], False)
        else:
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netG_A, self.netG_B], True)
        self.set_requires_grad([self.netf_s], False)

        # forward
        self.forward()      # compute fake images and reconstruction images.
        
        self.compute_G_loss()             # calculate gradients for G_A and G_B
        (self.loss_G/self.opt.iter_size).backward()

        self.compute_step(self.optimizer_G,self.loss_names_G)

        # D_A and D_B
        if self.disc_in_mask:
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_A_mask, self.netD_B_mask], True)
        else:
            self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.set_requires_grad([self.netG_A, self.netG_B], False)
        self.set_requires_grad([self.netf_s], False)

        self.compute_D_loss()      # calculate gradients for all discriminators
        (self.loss_D/self.opt.iter_size).backward()
        
        self.compute_step(self.optimizer_D,self.loss_names_D)
        
        # f_s
        if self.disc_in_mask:
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_A_mask, self.netD_B_mask], False)
        else:
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netG_A, self.netG_B], False)
        self.set_requires_grad([self.netf_s], True)
        
        self.compute_f_s_loss()
        (self.loss_f_s/self.opt.iter_size).backward()

        self.compute_step(self.optimizer_f_s,self.loss_names_f_s)
