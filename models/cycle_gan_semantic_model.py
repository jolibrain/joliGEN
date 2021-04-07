import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
import numpy as np
from .modules import loss
from util.util import gaussian

class CycleGANSemanticModel(BaseModel):
    #def name(self):
    #    return 'CycleGANModel'

    # new, copied from cyclegan model
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
        parser.set_defaults(no_dropout=False)  # default CycleGAN did not use dropout, beniz: we do
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--rec_noise', type=float, default=0.0, help='whether to add noise to reconstruction')
            parser.add_argument('--use_label_B', action='store_true', help='if true domain B has labels too')
            parser.add_argument('--train_cls_B', action='store_true', help='if true cls will be trained not only on domain A but also on domain B, if true use_label_B needs to be True')

        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 
                'D_B', 'G_B', 'cycle_B', 'idt_B', 
                'sem_AB', 'sem_BA', 'CLS']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A') # beniz: inverted for original

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'CLS']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.netG, opt.norm, 
                                        not opt.no_dropout, opt.G_spectral, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.netG, opt.norm, 
                                        not opt.no_dropout, opt.G_spectral, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            #use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.netD,
                                            opt.n_layers_D, opt.norm, opt.D_dropout, opt.D_spectral, #use_sigmoid, 
                                            opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.netD,
                                            opt.n_layers_D, opt.norm, opt.D_dropout, opt.D_spectral, #use_sigmoid, 
                                            opt.init_type, opt.init_gain, self.gpu_ids)
            self.netCLS = networks.define_C(opt.output_nc, opt.ndf,opt.crop_size,
                                            init_type=opt.init_type, init_gain=opt.init_gain,
                                            gpu_ids=self.gpu_ids, nclasses=opt.semantic_nclasses)
 
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionCLS = torch.nn.modules.CrossEntropyLoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_CLS = torch.optim.Adam(self.netCLS.parameters(), lr=1e-3, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            #beniz: not adding optimizers CLS (?)

            self.rec_noise = opt.rec_noise

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        #print(input['B'])
        if 'A_label' in input:
            #self.input_A_label = input['A_label' if AtoB else 'B_label'].to(self.device)
            self.input_A_label = input['A_label'].to(self.device)
            #self.input_B_label = input['B_label' if AtoB else 'A_label'].to(self.device) # beniz: unused
            #self.image_paths = input['B_paths'] # Hack!! forcing the labels to corresopnd to B domain
        if 'B_label' in input:
            self.input_B_label = input['B_label'].to(self.device)


    def forward(self):
        self.fake_B = self.netG_A(self.real_A)

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

        if self.isTrain:
           # Forward all four images through classifier
           # Keep predictions from fake images only
           #print('real_A shape=',self.real_A.shape)
           #print('real_A=',self.real_A)
           self.pred_real_A = self.netCLS(self.real_A)
           _,self.gt_pred_A = self.pred_real_A.max(1)
           pred_real_B = self.netCLS(self.real_B)
           _,self.gt_pred_B = pred_real_B.max(1)
           self.pred_fake_A = self.netCLS(self.fake_A)
           self.pred_fake_B = self.netCLS(self.fake_B)

           _,self.pfB = self.pred_fake_B.max(1) #beniz: unused ?
        

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D
    
    def backward_CLS(self):
        label_A = self.input_A_label
        # forward only real source image through semantic classifier
        pred_A = self.netCLS(self.real_A) 
        self.loss_CLS = self.criterionCLS(pred_A, label_A)
        if self.opt.train_cls_B:
            label_B = self.input_B_label
            pred_B = self.netCLS(self.real_B) 
            self.loss_CLS += self.criterionCLS(pred_B, label_B)
        
        self.loss_CLS.backward()

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
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

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) # removed the factor 2...
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss standard cyclegan
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        # semantic loss AB
        #print('fake_B=',self.pred_fake_B)
        #print('input_A_label=',self.input_A_label)
        #print(self.pred_fake_B.shape,self.input_A_label.shape)
        self.loss_sem_AB = self.criterionCLS(self.pred_fake_B, self.input_A_label)
        #self.loss_sem_AB = self.criterionCLS(self.pred_fake_B, self.gt_pred_A)
        # semantic loss BA
        if hasattr(self,'input_B_label'):
            self.loss_sem_BA = self.criterionCLS(self.pred_fake_A, self.input_B_label)
        else:
            self.loss_sem_BA = self.criterionCLS(self.pred_fake_A, self.gt_pred_B)
        #self.loss_sem_BA = 0
        #self.loss_sem_BA = self.criterionCLS(self.pred_fake_A, self.pfB) # beniz
        
        # only use semantic loss when classifier has reasonably low loss
        #if True:
        if not hasattr(self, 'loss_CLS') or self.loss_CLS.detach().item() > 1.0:
            self.loss_sem_AB = 0 * self.loss_sem_AB 
            self.loss_sem_BA = 0 * self.loss_sem_BA 
      
        self.loss_G += self.loss_sem_BA + self.loss_sem_AB
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netG_A, self.netG_B], True)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        # CLS
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.set_requires_grad([self.netCLS], True)
        self.optimizer_CLS.zero_grad()
        self.backward_CLS()
        self.optimizer_CLS.step()


