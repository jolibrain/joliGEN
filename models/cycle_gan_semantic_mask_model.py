import torch
import itertools
from util.image_pool import ImagePool
from util.losses import L1_Charbonnier_loss
from util.madgrad import MADGRAD
from .cycle_gan_model import CycleGANModel
from . import networks
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .modules import loss
from util.iter_calculator import IterCalculator
from util.network_group import NetworkGroup

class CycleGANSemanticMaskModel(CycleGANModel):
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
        parser = CycleGANModel.modify_commandline_options(parser,is_train)
        if is_train:
            parser.add_argument('--out_mask', action='store_true', help='use loss out mask')
            parser.add_argument('--lambda_out_mask', type=float, default=10.0, help='weight for loss out mask')
            parser.add_argument('--loss_out_mask', type=str, default='L1', help='loss mask')
            parser.add_argument('--charbonnier_eps', type=float, default=1e-6, help='Charbonnier loss epsilon value')
            parser.add_argument('--disc_in_mask', action='store_true', help='use in-mask discriminator')
            parser.add_argument('--train_f_s_B', action='store_true', help='if true f_s will be trained not only on domain A but also on domain B')
            parser.add_argument('--no_train_f_s_A', action='store_true', help='if true f_s wont be trained on domain A')
            parser.add_argument('--fs_light',action='store_true', help='whether to use a light (unet) network for f_s')
            parser.add_argument('--lr_f_s', type=float, default=0.0002, help='f_s learning rate')
            parser.add_argument('--nb_attn', type=int, default=10, help='number of attention masks')
            parser.add_argument('--nb_mask_input', type=int, default=1, help='number of attention masks which will be applied on the input image')
            parser.add_argument('--lambda_sem', type=float, default=1.0, help='weight for semantic loss')
            parser.add_argument('--madgrad',action='store_true',help='if true madgrad optim will be used')

        return parser
    
    def __init__(self, opt,rank):
        super().__init__(opt,rank)
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
        losses_G = ['sem_AB', 'sem_BA']

        if opt.out_mask:
            losses_G += ['out_mask_AB','out_mask_BA']

        losses_f_s = ['f_s']
            
        losses_D = []
        if opt.disc_in_mask:
            losses_D = ['D_A_mask', 'D_B_mask']
            
        self.loss_names_G += losses_G
        self.loss_names_f_s = losses_f_s
        self.loss_names_D += losses_D
        
        self.loss_names = self.loss_names_G + self.loss_names_f_s + self.loss_names_D
            
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names += ['f_s']
            if opt.disc_in_mask:
                self.model_names += ['D_A_mask', 'D_B_mask']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        if self.isTrain:
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
            self.fake_A_pool_mask = ImagePool(opt.pool_size)
            self.fake_B_pool_mask = ImagePool(opt.pool_size)
                
            # define loss functions
            self.criterionf_s = torch.nn.modules.CrossEntropyLoss()
            
            if opt.out_mask or opt.disc_in_mask:
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
                

            self.optimizers.append(self.optimizer_f_s)

            if self.opt.iter_size > 1 :
                self.iter_calculator = IterCalculator(self.loss_names)
                for i,cur_loss in enumerate(self.loss_names):
                    self.loss_names[i] = cur_loss + '_avg'
                    setattr(self, "loss_" + self.loss_names[i], 0)


            ###Making groups
            discriminators = ["D_A","D_B"]
            if opt.disc_in_mask:
                discriminators += ["D_A_mask","D_B_mask"]
            
            self.group_f_s= NetworkGroup(networks_to_optimize=["f_s"],forward_functions=None,backward_functions=["compute_f_s_loss"],loss_names_list=["loss_names_f_s"],optimizer=["optimizer_f_s"],loss_backward=["loss_f_s"])
            self.networks_groups.append(self.group_f_s)

            
    def set_input(self, input):
        super().set_input(input)
        if 'A_label' in input :
            self.input_A_label = input['A_label'].to(self.device).squeeze(1)
        if 'B_label' in input and len(input['B_label']) > 0:
            self.input_B_label = input['B_label'].to(self.device).squeeze(1) # beniz: unused

    def data_dependent_initialize(self, data):
        self.set_input(data)
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_seg_A = ['input_A_label','gt_pred_A','pfB_max']

        if hasattr(self,'input_B_label'):
            visual_names_seg_B = ['input_B_label']
        else:
            visual_names_seg_B = []
            
        visual_names_seg_B += ['gt_pred_B','pfA_max']
        
        visual_names_out_mask_A = ['real_A_out_mask','fake_B_out_mask']

        visual_names_out_mask_B = ['real_B_out_mask','fake_A_out_mask']
        
        visual_names_mask = ['fake_B_mask','fake_A_mask']

        visual_names_mask_in = ['real_B_mask','fake_B_mask','real_A_mask','fake_A_mask',
                                'real_B_mask_in','fake_B_mask_in','real_A_mask_in','fake_A_mask_in']
        
        self.visual_names += [visual_names_seg_A , visual_names_seg_B ]

        if self.opt.out_mask :
            self.visual_names += [visual_names_out_mask_A,visual_names_out_mask_B]

        if self.opt.disc_in_mask:
            self.visual_names += [visual_names_mask_in]
        

    def forward(self):
        super().forward()
        d=1
        if self.isTrain:
            self.pred_real_A = self.netf_s(self.real_A)            
            self.gt_pred_A = F.log_softmax(self.pred_real_A,dim= d).argmax(dim=d)
            
            self.pred_real_B = self.netf_s(self.real_B)
            self.gt_pred_B = F.log_softmax(self.pred_real_B,dim=d).argmax(dim=d)
            
            self.pred_fake_A = self.netf_s(self.fake_A)
            
            self.pfA = F.log_softmax(self.pred_fake_A,dim=d)
            self.pfA_max = self.pfA.argmax(dim=d)

            if hasattr(self,'criterionMask'):
                label_A = self.input_A_label
                label_A_in = label_A.unsqueeze(1)
                label_A_inv = torch.tensor(np.ones(label_A.size())).to(self.device) - label_A>0
                label_A_inv = label_A_inv.unsqueeze(1)
                
                self.real_A_out_mask = self.real_A *label_A_inv
                self.fake_B_out_mask = self.fake_B *label_A_inv

                if self.disc_in_mask:
                    self.real_A_mask_in = self.real_A * label_A_in
                    self.fake_B_mask_in = self.fake_B * label_A_in
                    self.real_A_mask = self.real_A
                    self.fake_B_mask = self.fake_B_mask_in + self.real_A_out_mask.float()
                                            
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
                        self.real_B_mask = self.real_B
                        self.fake_A_mask = self.fake_A_mask_in + self.real_B_out_mask.float()
    
        self.pred_fake_B = self.netf_s(self.fake_B)
        self.pfB = F.log_softmax(self.pred_fake_B,dim=d)
        self.pfB_max = self.pfB.argmax(dim=d)
               
    def compute_f_s_loss(self):
        #print('backward fs')
        self.loss_f_s = 0
        if not self.opt.no_train_f_s_A:
            label_A = self.input_A_label
            # forward only real source image through semantic classifier
            pred_A = self.netf_s(self.real_A) 
            self.loss_f_s += self.criterionf_s(pred_A, label_A)#.squeeze(1))
        if self.opt.train_f_s_B:
            label_B = self.input_B_label
            pred_B = self.netf_s(self.real_B) 
            self.loss_f_s += self.criterionf_s(pred_B, label_B)#.squeeze(1))

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
        super().compute_D_loss()
        if self.disc_in_mask:
            self.compute_D_A_mask_in_loss()
            self.compute_D_B_mask_in_loss()
            self.compute_D_A_mask_loss()
            self.compute_D_B_mask_loss()
            self.loss_D += self.loss_D_A + self.loss_D_B + self.loss_D_A_mask + self.loss_D_B_mask

    def compute_G_loss(self):
        super().compute_G_loss()
        # Identity loss
        
        if self.disc_in_mask:
            self.loss_G_A_mask = self.compute_G_loss_GAN_generic(self.netD_A,"B",self.D_loss,fake_name="fake_B_mask_in")
            self.loss_G_B_mask = self.compute_G_loss_GAN_generic(self.netD_B,"A",self.D_loss,fake_name="fake_A_mask_in")
        
            self.loss_G_A = self.compute_G_loss_GAN_generic(self.netD_A_mask,"B",self.D_loss,fake_name="fake_B_mask")
            self.loss_G_B = self.compute_G_loss_GAN_generic(self.netD_B_mask,"A",self.D_loss,fake_name="fake_A_mask")
            
            # combined loss standard cyclegan
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_G_A_mask + self.loss_G_B_mask

        # semantic loss AB
        self.loss_sem_AB = self.opt.lambda_sem*self.criterionf_s(self.pfB, self.input_A_label)

        # semantic loss BA
        if hasattr(self, 'input_B_label'):
            self.loss_sem_BA = self.opt.lambda_sem*self.criterionf_s(self.pfA, self.input_B_label)#.squeeze(1))
        else:
            self.loss_sem_BA = self.opt.lambda_sem*self.criterionf_s(self.pfA, self.gt_pred_B)#.squeeze(1))
        
        # only use semantic loss when classifier has reasonably low loss
        #if True:
        if not hasattr(self, 'loss_f_s') or self.loss_f_s > self.opt.semantic_threshold:
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

