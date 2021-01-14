import sys
import torch
import itertools
from util.image_pool import ImagePool
from util.losses import L1_Charbonnier_loss
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os
from models.vgg_perceptual_loss import VGGPerceptualLoss
#import kornia.augmentation
#import sys

import random
import math
from torch import distributed as dist

from .modules import loss

from util.util import gaussian

class CycleGANSemanticMaskSty2Model(BaseModel):
    #def name(self):
    #    return 'CycleGANModel'

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
            parser.add_argument('--lambda_G', type=float, default=1.0, help='weight for generator loss')
            parser.add_argument('--out_mask', action='store_true', help='use loss out mask')
            parser.add_argument('--lambda_out_mask', type=float, default=10.0, help='weight for loss out mask')
            parser.add_argument('--loss_out_mask', type=str, default='L1', help='loss mask')
            parser.add_argument('--charbonnier_eps', type=float, default=1e-6, help='Charbonnier loss epsilon value')
            parser.add_argument('--train_f_s_B', action='store_true', help='if true f_s will be trained not only on domain A but also on domain B')
            parser.add_argument('--fs_light',action='store_true', help='whether to use a light (unet) network for f_s')
            parser.add_argument('--lr_f_s', type=float, default=0.0002, help='f_s learning rate')
            parser.add_argument('--D_noise', action='store_true', help='whether to add instance noise to discriminator inputs')
            parser.add_argument('--D_label_smooth', action='store_true', help='whether to use one-sided label smoothing with discriminator')
            parser.add_argument('--rec_noise', type=float, default=0.0, help='whether to add noise to reconstruction')
            parser.add_argument('--wplus', action='store_true', help='whether to work in W+ latent space')
            parser.add_argument('--wskip', action='store_true', help='whether to use skip connections to latent wplus heads')
            parser.add_argument('--truncation',type=float,default=1,help='whether to use truncation trick (< 1)')
            parser.add_argument('--decoder_size', type=int, default=512)
            parser.add_argument('--d_reg_every', type=int, default=16,help='regularize discriminator each x iterations, no reg if set to 0')
            parser.add_argument('--g_reg_every', type=int, default=4,help='regularize decider sty2 each x iterations, no reg if set to 0')
            parser.add_argument('--r1', type=float, default=10)
            parser.add_argument('--mixing', type=float, default=0.9)
            parser.add_argument('--path_batch_shrink', type=int, default=2)
            parser.add_argument('--path_regularize', type=float, default=2)
            parser.add_argument('--no_init_weight_D_sty2', action='store_true')
            parser.add_argument('--no_init_weight_dec_sty2', action='store_true')
            parser.add_argument('--no_init_weight_G', action='store_true')
            parser.add_argument('--load_weight_decoder', action='store_true')
            parser.add_argument('--percept_loss', action='store_true', help='whether to use perceptual loss for reconstruction and identity')
            parser.add_argument('--randomize_noise', action='store_true', help='whether to use random noise in sty2 decoder')
            parser.add_argument('--D_lightness', type=int, default=1, help='sty2 discriminator lightness, 1: normal, then 2, 4, 8 for less parameters')

            parser.add_argument('--w_loss', action='store_true')
            parser.add_argument('--lambda_w_loss', type=float, default=10.0)

            parser.add_argument('--n_loss', action='store_true')
            parser.add_argument('--lambda_n_loss', type=float, default=10.0)

            parser.add_argument('--cam_loss', action='store_true')
            parser.add_argument('--lambda_cam', type=float, default=10.0)
            parser.add_argument('--sty2_clamp', action='store_true')
            
        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        losses = ['G_A','G_B']
        losses += ['D_A', 'D_B']
        losses=[]
        if opt.out_mask:
            losses += ['out_mask_AB','out_mask_BA']

        losses += ['cycle_A', 'idt_A', 
                   'cycle_B', 'idt_B', 
                   'sem_AB', 'sem_BA', 'f_s']

        losses += ['g_nonsaturating_A','g_nonsaturating_B']

        if self.opt.g_reg_every != 0:
            losses +=['weighted_path_A','weighted_path_B']

        losses+= ['d_dec_A','d_dec_B']

        if self.opt.d_reg_every != 0:
            losses += ['grad_pen_A','grad_pen_B']#,'d_dec_reg_A', 'd_dec_reg_B']

        if opt.w_loss:
            losses += ['w_A','w_B']

        if opt.n_loss:
            losses += ['n_A','n_B']

        if opt.cam_loss:
            losses += ['cam']

        self.loss_names = losses
        self.truncation = opt.truncation
        self.randomize_noise = opt.randomize_noise
        self.r1 = opt.r1
        self.percept_loss = opt.percept_loss
        
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']

        visual_names_B = ['real_B', 'fake_A', 'rec_B']

        if self.isTrain and self.opt.lambda_identity > 0.0:
           visual_names_A.append('idt_B')
           visual_names_B.append('idt_A') # beniz: inverted for original

        visual_names_seg_A = ['input_A_label','gt_pred_A','pfB_max']

        
        visual_names_seg_B = ['input_B_label','gt_pred_B','pfA_max']
        
        visual_names_out_mask = ['real_A_out_mask','fake_B_out_mask','real_B_out_mask','fake_A_out_mask']

        visual_names_mask = ['fake_B_mask','fake_A_mask']

        visual_names_mask_in = ['real_B_mask','fake_B_mask','real_A_mask','fake_A_mask',
                                'real_B_mask_in','fake_B_mask_in','real_A_mask_in','fake_A_mask_in']
        
        self.visual_names = visual_names_A + visual_names_B + visual_names_seg_A + visual_names_seg_B 

        if opt.out_mask :
            self.visual_names += visual_names_out_mask

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'f_s']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'f_s']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        print('define gen')
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.netG, opt.norm, 
                                        not opt.no_dropout, opt.G_spectral, opt.init_type, opt.init_gain, self.gpu_ids, decoder=False, wplus=opt.wplus, wskip=opt.wskip,img_size=opt.crop_size,img_size_dec=opt.decoder_size)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.netG, opt.norm, 
                                        not opt.no_dropout, opt.G_spectral, opt.init_type, opt.init_gain, self.gpu_ids, decoder=False, wplus=opt.wplus, wskip=opt.wskip,img_size=opt.crop_size,img_size_dec=opt.decoder_size)

        # Define stylegan2 decoder
        print('define decoder')
        self.netDecoderG_A = networks.define_decoder(init_type=opt.init_type, init_gain=opt.init_gain,gpu_ids=self.gpu_ids,size=self.opt.decoder_size,init_weight=not self.opt.no_init_weight_dec_sty2,clamp=self.opt.sty2_clamp)
        self.netDecoderG_B = networks.define_decoder(init_type=opt.init_type, init_gain=opt.init_gain,gpu_ids=self.gpu_ids,size=self.opt.decoder_size,init_weight=not self.opt.no_init_weight_dec_sty2,clamp=self.opt.sty2_clamp)
        
        # Load pretrained weights stylegan2 decoder
        
        nameDGA = 'DecoderG_A'
        nameDGB = 'DecoderG_B'
        if self.opt.load_weight_decoder:
            load_filename = 'network_A.pt'
            load_path = os.path.join(self.save_dir, load_filename)
        
            net = getattr(self, 'net' + nameDGA)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict['g_ema'])
            self.set_requires_grad(net, True)
                                
            load_filename = 'network_B.pt'
            load_path = os.path.join(self.save_dir, load_filename)
        
            net = getattr(self, 'net' + nameDGB)
            
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict['g_ema'])
            self.set_requires_grad(net, True)

        if self.opt.truncation < 1:
            self.mean_latent_A = self.netDecoderG_A.module.mean_latent(4096)
            self.mean_latent_B = self.netDecoderG_B.module.mean_latent(4096)
        else:
            self.mean_latent_A = None
            self.mean_latent_B = None
        
            
                                
        self.model_names += [nameDGA,nameDGB]
    
        print('define dis dec')
        self.netDiscriminatorDecoderG_A = networks.define_discriminatorstylegan2(init_type=opt.init_type, init_gain=opt.init_gain,gpu_ids=self.gpu_ids,init_weight=not self.opt.no_init_weight_D_sty2,img_size=self.opt.crop_size,lightness=opt.D_lightness)
        self.model_names += ['DiscriminatorDecoderG_A']

        self.netDiscriminatorDecoderG_B = networks.define_discriminatorstylegan2(init_type=opt.init_type, init_gain=opt.init_gain,gpu_ids=self.gpu_ids,init_weight=not self.opt.no_init_weight_D_sty2,img_size=self.opt.crop_size,lightness=opt.D_lightness)
        self.model_names += ['DiscriminatorDecoderG_B']
            
        self.netf_s = networks.define_f(opt.input_nc, nclasses=opt.semantic_nclasses, 
                                        init_type=opt.init_type, init_gain=opt.init_gain,
                                        gpu_ids=self.gpu_ids, fs_light=opt.fs_light)

        if self.opt.cam_loss:
            self.netCamClassifier_w_B = networks.define_classifier_w(init_type=opt.init_type, init_gain=opt.init_gain,gpu_ids=self.gpu_ids,init_weight=not self.opt.no_init_weight_D_sty2,img_size_dec=self.opt.decoder_size)
            self.model_names += ['CamClassifier_w_B']
            self.netCamClassifier_w_A = networks.define_classifier_w(init_type=opt.init_type, init_gain=opt.init_gain,gpu_ids=self.gpu_ids,init_weight=not self.opt.no_init_weight_D_sty2,img_size_dec=self.opt.decoder_size)
            self.model_names += ['CamClassifier_w_A']
        
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            self.real_A_pool = ImagePool(opt.pool_size)
            self.real_B_pool = ImagePool(opt.pool_size)
                
            # define loss functions
            if opt.D_label_smooth:
                target_real_label = 0.9
            else:
                target_real_label = 1.0
            self.criterionGAN = loss.GANLoss(opt.gan_mode,target_real_label=target_real_label).to(self.device)
            if opt.percept_loss:
                self.criterionCycle = VGGPerceptualLoss().cuda()
                self.criterionCycle2 = torch.nn.MSELoss()
                self.criterionIdt = VGGPerceptualLoss().cuda()
                self.criterionIdt2 = torch.nn.MSELoss()
            else:
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

            if opt.w_loss:
                self.criterion_w =  torch.nn.MSELoss()

            if opt.n_loss:
                self.criterion_n =  torch.nn.MSELoss()

            if opt.cam_loss:
                self.criterion_cam_w = torch.nn.BCEWithLogitsLoss()
                
            # initialize optimizers
            if opt.cam_loss:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(),self.netDecoderG_A.parameters(), self.netDecoderG_B.parameters(),self.netCamClassifier_w_A.parameters(),self.netCamClassifier_w_B.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(),self.netDecoderG_A.parameters(), self.netDecoderG_B.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_f_s = torch.optim.Adam(self.netf_s.parameters(), lr=opt.lr_f_s, betas=(opt.beta1, 0.999))

            self.optimizer_D_Decoder = torch.optim.Adam(itertools.chain(self.netDiscriminatorDecoderG_A.parameters(),self.netDiscriminatorDecoderG_B.parameters()),
                                            lr=opt.D_lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            #self.optimizers.append(self.optimizer_D)
            #beniz: not adding optimizers f_s (?)

            self.rec_noise = opt.rec_noise
            self.stddev = 0.1
            self.D_noise = opt.D_noise

            self.niter=0
            self.mean_path_length_A = 0
            self.mean_path_length_B = 0
            
     
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if 'A_label' in input :
            #self.input_A_label = input['A_label' if AtoB else 'B_label'].to(self.device)
            self.input_A_label = input['A_label'].to(self.device).squeeze(1)
            #self.input_A_label_dis = display_mask(self.input_A_label)  
        if 'B_label' in input:
            self.input_B_label = input['B_label'].to(self.device).squeeze(1) # beniz: unused
            #self.image_paths = input['B_paths'] # Hack!! forcing the labels to corresopnd to B domain


    def forward(self):
        self.z_fake_B, self.n_fake_B = self.netG_A(self.real_A)

        d = 1
        
        #self.netDecoderG_A.eval()
        self.fake_B,self.latent_fake_B = self.netDecoderG_A(self.z_fake_B,input_is_latent=True,truncation=self.truncation,truncation_latent=self.mean_latent_A,randomize_noise=self.randomize_noise,return_latents=True, noise=self.n_fake_B)
        if self.opt.decoder_size > self.opt.crop_size:
            self.fake_B = F.interpolate(self.fake_B,self.opt.crop_size)
            
        if self.isTrain:
            #self.netDecoderG_B.eval()
            if self.rec_noise > 0.0:
                self.fake_B_noisy1 = gaussian(self.fake_B, self.rec_noise)
                self.z_rec_A, self.n_rec_A = self.netG_B(self.fake_B_noisy1)
            else:
                self.z_rec_A, self.n_rec_A = self.netG_B(self.fake_B)
            self.rec_A = self.netDecoderG_B(self.z_rec_A,input_is_latent=True,truncation=self.truncation,truncation_latent=self.mean_latent_B, randomize_noise=self.randomize_noise, noise=self.n_rec_A)[0]
            if self.opt.decoder_size > self.opt.crop_size:
                self.rec_A = F.interpolate(self.rec_A,self.opt.crop_size)
            
            self.z_fake_A, self.n_fake_A = self.netG_B(self.real_B)
            self.fake_A,self.latent_fake_A = self.netDecoderG_B(self.z_fake_A,input_is_latent=True,truncation=self.truncation,truncation_latent=self.mean_latent_B,randomize_noise=self.randomize_noise,return_latents=True, noise=self.n_fake_A)
            if self.opt.decoder_size > self.opt.crop_size:
                self.fake_A = F.interpolate(self.fake_A,self.opt.crop_size)
            
            if self.rec_noise > 0.0:
                self.fake_A_noisy1 = gaussian(self.fake_A, self.rec_noise)
                self.z_rec_B, self.n_rec_B = self.netG_A(self.fake_A_noisy1)
            else:
                self.z_rec_B, self.n_rec_B = self.netG_A(self.fake_A)
            self.rec_B = self.netDecoderG_A(self.z_rec_B,input_is_latent=True,truncation=self.truncation,truncation_latent=self.mean_latent_A, randomize_noise=self.randomize_noise, noise=self.n_rec_B)[0]
            if self.opt.decoder_size > self.opt.crop_size:
                self.rec_B = F.interpolate(self.rec_B,self.opt.crop_size)
            
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
                label_A_inv = torch.tensor(np.ones(label_A.size())).to(self.device) - label_A
                label_A_inv = label_A_inv.unsqueeze(1)
                #label_A_inv = torch.cat ([label_A_inv,label_A_inv,label_A_inv],1)
                
                self.real_A_out_mask = self.real_A *label_A_inv
                self.fake_B_out_mask = self.fake_B *label_A_inv
                    
                if self.D_noise:
                    self.fake_B_noisy = gaussian(self.fake_B)
                    self.real_A_noisy = gaussian(self.real_A)
                    #self.real_A_mask_in = self.aug_seq(self.real_A_mask_in)
                    #self.fake_B_mask_in = self.aug_seq(self.fake_B_mask_in)
                    #self.real_A_mask = self.aug_seq(self.real_A_mask)
                    #self.fake_B_mask = self.aug_seq(self.fake_B_mask)
                        
                if hasattr(self, 'input_B_label'):
                
                    label_B = self.input_B_label
                    label_B_in = label_B.unsqueeze(1)
                    label_B_inv = torch.tensor(np.ones(label_B.size())).to(self.device) - label_B
                    label_B_inv = label_B_inv.unsqueeze(1)
                    #label_B_inv = torch.cat ([label_B_inv,label_B_inv,label_B_inv],1)
                    
                    self.real_B_out_mask = self.real_B *label_B_inv
                    self.fake_A_out_mask = self.fake_A *label_B_inv

                    if self.D_noise:
                        self.fake_A_noisy = gaussian(self.fake_A)
                        self.real_B_noisy = gaussian(self.real_B)
                        #self.real_B_mask_in = self.aug_seq(self.real_B_mask_in)
                        #self.fake_A_mask_in = self.aug_seq(self.fake_A_mask_in)
                        #self.real_B_mask = self.aug_seq(self.real_B_mask)
                        #self.fake_A_mask = self.aug_seq(self.fake_A_mask)
                        
        self.pred_fake_B = self.netf_s(self.fake_B)
        self.pfB = F.log_softmax(self.pred_fake_B,dim=d)#.argmax(dim=d)
        self.pfB_max = self.pfB.argmax(dim=d)


           
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
    
    def backward_f_s(self):
        #print('backward fs')
        label_A = self.input_A_label
        # forward only real source image through semantic classifier
        pred_A = self.netf_s(self.real_A) 
        self.loss_f_s = self.criterionf_s(pred_A, label_A)#.squeeze(1))
        if self.opt.train_f_s_B:
            label_B = self.input_B_label
            pred_B = self.netf_s(self.real_B) 
            self.loss_f_s += self.criterionf_s(pred_B, label_B)#.squeeze(1))
        self.loss_f_s.backward()

    def backward_D_A(self):
        if self.D_noise:
            fake_B = self.fake_B_pool.query(self.fake_B_noisy)
            self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B_noisy, fake_B)
        else:
            fake_B = self.fake_B_pool.query(self.fake_B)
            self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        if self.D_noise:
            fake_A = self.fake_A_pool.query(self.fake_A_noisy)
            self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A_noisy, fake_A)
        else:
            fake_A = self.fake_A_pool.query(self.fake_A)
            self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_D_A_mask(self):
        fake_B_mask = self.fake_B_pool_mask.query(self.fake_B_mask)
        self.loss_D_A_mask = self.backward_D_basic(self.netD_A_mask, self.real_B_mask, fake_B_mask)

    def backward_D_B_mask(self):
        fake_A_mask = self.fake_A_pool_mask.query(self.fake_A_mask)
        self.loss_D_B_mask = self.backward_D_basic(self.netD_B_mask, self.real_A_mask, fake_A_mask)

    def backward_D_A_mask_in(self):
        fake_B_mask_in = self.fake_B_pool.query(self.fake_B_mask_in)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B_mask_in, fake_B_mask_in)

    def backward_D_B_mask_in(self):
        fake_A_mask_in = self.fake_A_pool.query(self.fake_A_mask)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A_mask_in, fake_A_mask_in)

    def backward_G(self):
        #print('BACKWARD G')
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_G = self.opt.lambda_G
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.z_idt_A, self.n_idt_A = self.netG_A(self.real_B)
            self.idt_A = self.netDecoderG_A(self.z_idt_A,input_is_latent=True,truncation=self.truncation,truncation_latent=self.mean_latent_A,randomize_noise=self.randomize_noise, noise=self.n_idt_A)[0]
            if self.opt.decoder_size > self.opt.crop_size:
                self.idt_A = F.interpolate(self.idt_A,self.opt.crop_size)
            
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            if self.percept_loss:
                self.loss_idt_A += self.criterionIdt2(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.z_idt_B, self.n_idt_B = self.netG_B(self.real_A)
            self.idt_B = self.netDecoderG_B(self.z_idt_B,input_is_latent=True,truncation=self.truncation,truncation_latent=self.mean_latent_B,randomize_noise=self.randomize_noise, noise=self.n_idt_B)[0]
            if self.opt.decoder_size > self.opt.crop_size:
                self.idt_B = F.interpolate(self.idt_B,self.opt.crop_size)
            
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            if self.percept_loss:
                self.loss_idt_B += self.criterionIdt2(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        if self.percept_loss:
            self.loss_cycle_A += self.criterionCycle2(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        if self.percept_loss:
            self.loss_cycle_B += self.criterionCycle2(self.rec_B, self.real_B) * lambda_B
        # combined loss standard cyclegan
        self.loss_G = self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B #self.loss_G_A + self.loss_G_B + 

        # semantic loss AB
        self.loss_sem_AB = self.criterionf_s(self.pfB, self.input_A_label)
        
        # semantic loss BA
        if hasattr(self, 'input_B_label'):
            self.loss_sem_BA = self.criterionf_s(self.pfA, self.input_B_label)#.squeeze(1))
        else:
            self.loss_sem_BA = self.criterionf_s(self.pfA, self.gt_pred_B)#.squeeze(1))
                
        # only use semantic loss when classifier has reasonably low loss
        if not hasattr(self, 'loss_f_s') or self.loss_f_s.detach().item() > 1.0:
            self.loss_sem_AB = 0 * self.loss_sem_AB 
            self.loss_sem_BA = 0 * self.loss_sem_BA 
        self.loss_G += self.loss_sem_BA + self.loss_sem_AB

        lambda_out_mask = self.opt.lambda_out_mask

        if hasattr(self,'criterionMask'):
            self.loss_out_mask_AB = self.criterionMask( self.real_A_out_mask, self.fake_B_out_mask) * lambda_out_mask
            self.loss_out_mask_BA = self.criterionMask( self.real_B_out_mask, self.fake_A_out_mask) * lambda_out_mask
            self.loss_G += self.loss_out_mask_AB + self.loss_out_mask_BA


        compute_g_regularize = True
        if self.opt.path_regularize == 0.0 or self.opt.g_reg_every == 0 or not self.niter % self.opt.g_reg_every == 0 :
            #self.loss_weighted_path_A = 0* self.loss_weighted_path_A
            #self.loss_weighted_path_B = 0* self.loss_weighted_path_B
            compute_g_regularize = False
        
        #A
        self.fake_pred_g_loss_A = self.netDiscriminatorDecoderG_A(self.fake_A)
        self.loss_g_nonsaturating_A = self.g_nonsaturating_loss(self.fake_pred_g_loss_A)
        
        if compute_g_regularize:
            self.path_loss_A, self.mean_path_length_A, self.path_lengths_A = self.g_path_regularize(
                self.fake_A, self.latent_fake_A, self.mean_path_length_A
            )

            self.loss_weighted_path_A = self.opt.path_regularize * self.opt.g_reg_every * self.path_loss_A
        
            if self.opt.path_batch_shrink:
                self.loss_weighted_path_A += 0 * self.fake_A[0, 0, 0, 0]

            self.mean_path_length_avg_A = (
                self.reduce_sum(self.mean_path_length_A).item() / self.get_world_size()
            )
        else:
            self.loss_weighted_path_A = 0#*self.loss_weighted_path_A

        #B
        self.fake_pred_g_loss_B = self.netDiscriminatorDecoderG_B(self.fake_B)
        self.loss_g_nonsaturating_B = self.g_nonsaturating_loss(self.fake_pred_g_loss_B)
        
        if compute_g_regularize:
            self.path_loss_B, self.mean_path_length_B, self.path_lengths_B = self.g_path_regularize(
                self.fake_B, self.latent_fake_B, self.mean_path_length_B
            )

            self.loss_weighted_path_B = self.opt.path_regularize * self.opt.g_reg_every * self.path_loss_B
        
            if self.opt.path_batch_shrink:
                #self.loss_weighted_path_B += 0 * self.fake_img_path_loss_B[0, 0, 0, 0]
                self.loss_weighted_path_B += 0 * self.fake_B[0, 0, 0, 0]

            self.mean_path_length_avg_B = (
                self.reduce_sum(self.mean_path_length_B).item() / self.get_world_size()
            )
        else:
            self.loss_weighted_path_B = 0#*self.loss_weighted_path_B

        self.loss_G += self.opt.lambda_G*(self.loss_g_nonsaturating_A + self.loss_g_nonsaturating_B)

        if not self.opt.path_regularize == 0.0 and not self.opt.g_reg_every == 0 and self.niter % self.opt.g_reg_every == 0 :
            self.loss_G += self.loss_weighted_path_A + self.loss_weighted_path_B

        if self.opt.w_loss:
            p = random.uniform(0, 1)
            if p<0.5:#idt as reference
                self.loss_w_A = self.criterion_w(self.z_idt_B.clone().detach(),self.z_rec_A) * self.opt.lambda_w_loss
                self.loss_w_B = self.criterion_w(self.z_idt_A.clone().detach(),self.z_rec_B) * self.opt.lambda_w_loss
            else:#rec as reference
                self.loss_w_A = self.criterion_w(self.z_idt_B,self.z_rec_A.clone().detach()) * self.opt.lambda_w_loss
                self.loss_w_B = self.criterion_w(self.z_idt_A,self.z_rec_B.clone().detach()) * self.opt.lambda_w_loss

            self.loss_G += self.loss_w_A + self.loss_w_B

        if self.opt.n_loss:
            p = random.uniform(0, 1)
            temp_n_idt_B = [temp.flatten()for temp in self.n_idt_B]
            temp_n_idt_A = [temp.flatten()for temp in self.n_idt_A]
            temp_n_rec_B = [temp.flatten()for temp in self.n_rec_B]
            temp_n_rec_A = [temp.flatten()for temp in self.n_rec_A]
            if p<0.5:#idt as reference
                self.loss_n_A = self.criterion_n(torch.cat(temp_n_idt_B).clone().detach(),torch.cat(temp_n_rec_A)) * self.opt.lambda_n_loss
                self.loss_n_B = self.criterion_n(torch.cat(temp_n_idt_A).clone().detach(),torch.cat(temp_n_rec_B)) * self.opt.lambda_n_loss
            else:#rec as reference
                self.loss_n_A = self.criterion_n(torch.cat(temp_n_idt_B),torch.cat(temp_n_rec_A).clone().detach()) * self.opt.lambda_n_loss
                self.loss_n_B = self.criterion_n(torch.cat(temp_n_idt_A),torch.cat(temp_n_rec_B).clone().detach()) * self.opt.lambda_n_loss

            self.loss_G += self.loss_n_A + self.loss_n_B

        if self.opt.cam_loss:
            self.pred_w_fake_A = self.netCamClassifier_w_A(torch.stack(self.z_fake_A))
            self.pred_w_rec_A = self.netCamClassifier_w_A(torch.stack(self.z_rec_A))
            self.pred_w_idt_A = self.netCamClassifier_w_A(torch.stack(self.z_idt_A))
            
            self.pred_w_fake_B = self.netCamClassifier_w_B(torch.stack(self.z_fake_B))
            self.pred_w_rec_B = self.netCamClassifier_w_A(torch.stack(self.z_rec_B))
            self.pred_w_idt_B = self.netCamClassifier_w_B(torch.stack(self.z_idt_B))
        
            self.loss_cam = self.criterion_cam_w(self.pred_w_fake_A,torch.ones_like(self.pred_w_fake_A).to(self.device)) * self.opt.lambda_cam
            self.loss_cam += self.criterion_cam_w(self.pred_w_fake_B,torch.ones_like(self.pred_w_fake_B).to(self.device))* self.opt.lambda_cam
            self.loss_cam += self.criterion_cam_w(self.pred_w_rec_B,torch.ones_like(self.pred_w_rec_B).to(self.device))* self.opt.lambda_cam
            self.loss_cam += self.criterion_cam_w(self.pred_w_rec_A,torch.ones_like(self.pred_w_rec_A).to(self.device))* self.opt.lambda_cam

            self.loss_cam += self.criterion_cam_w(self.pred_w_idt_A,torch.zeros_like(self.pred_w_idt_A).to(self.device)) * self.opt.lambda_cam
            self.loss_cam += self.criterion_cam_w(self.pred_w_idt_B,torch.zeros_like(self.pred_w_idt_B).to(self.device)) * self.opt.lambda_cam

            self.loss_G += self.loss_cam
        
        self.loss_G.backward()

    def backward_discriminator_decoder(self):
        real_pred_A = self.netDiscriminatorDecoderG_A(self.real_A)
        fake_pred_A = self.netDiscriminatorDecoderG_A(self.fake_A_pool.query(self.fake_A))

        self.loss_d_dec_A = self.d_logistic_loss(real_pred_A,fake_pred_A).unsqueeze(0)

        #print(self.loss_d_dec_A)
        

        
        real_pred_B = self.netDiscriminatorDecoderG_B(self.real_B)
        fake_pred_B = self.netDiscriminatorDecoderG_B(self.fake_B_pool.query(self.fake_B))
        self.loss_d_dec_B = self.d_logistic_loss(real_pred_B,fake_pred_B).unsqueeze(0)

        self.loss_d_dec = self.loss_d_dec_A + self.loss_d_dec_B
        #print(self.d_loss)
        #print(self.d_loss.shape)
        
        if self.opt.d_reg_every != 0:
            if self.niter %self.opt.d_reg_every == 0:
                temp = real_pred_A/real_pred_A.detach()
        
                #self.real_A.requires_grad = True
                #real_pred_A_2 = self.netDiscriminatorDecoderG_A(self.real_A)
                cur_real_A = self.real_A_pool.query(self.real_A)
                cur_real_A.requires_grad = True
                real_pred_A_2 = self.netDiscriminatorDecoderG_A(cur_real_A)
                #r1_loss_A = self.d_r1_loss(real_pred_A_2, cur_real_A)

                self.loss_grad_pen_A = self.gradient_penalty(cur_real_A,real_pred_A_2,self.r1)
                
                #self.loss_d_dec_reg_A=self.opt.r1 / 2 * r1_loss_A * self.opt.d_reg_every * temp
                
                #self.real_B.requires_grad = True
                #real_pred_B_2 = self.netDiscriminatorDecoderG_B(self.real_B)
                cur_real_B = self.real_B_pool.query(self.real_B)
                cur_real_B.requires_grad = True
                real_pred_B_2 = self.netDiscriminatorDecoderG_B(cur_real_B)
                #r1_loss_B = self.d_r1_loss(real_pred_B_2, cur_real_B)
            
                self.loss_grad_pen_B = self.gradient_penalty(cur_real_B,real_pred_B_2,self.r1)
                
                #self.loss_d_dec_reg_B=self.opt.r1 / 2 * r1_loss_B * self.opt.d_reg_every * temp
        
                #self.loss_d_dec_reg_A = 0 * self.loss_d_dec_reg_A
                #self.loss_d_dec_reg_B = 0 * self.loss_d_dec_reg_B
            else:
                self.loss_grad_pen_A = 0# * self.loss_grad_pen_A
                self.loss_grad_pen_B = 0# * self.loss_grad_pen_B

            #self.loss_d_dec += self.loss_d_dec_reg_A + self.loss_d_dec_reg_B
            self.loss_d_dec += self.loss_grad_pen_A + self.loss_grad_pen_B

        self.loss_d_dec.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B

        self.set_requires_grad([self.netDiscriminatorDecoderG_A,self.netDiscriminatorDecoderG_B], False)
        self.set_requires_grad([self.netG_A, self.netG_B], True)
        self.set_requires_grad([self.netDecoderG_A, self.netDecoderG_B], True)
        self.netDecoderG_A.zero_grad()
        self.netDecoderG_B.zero_grad()
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

        self.set_requires_grad([self.netf_s], True)
        # f_s
        self.optimizer_f_s.zero_grad()
        self.backward_f_s()
        self.optimizer_f_s.step()

        self.optimizer_D_Decoder.zero_grad()
        self.niter = self.niter +1
        self.set_requires_grad([self.netDiscriminatorDecoderG_A,self.netDiscriminatorDecoderG_B], True)
        self.backward_discriminator_decoder()
        self.optimizer_D_Decoder.step()
        self.set_requires_grad([self.netDiscriminatorDecoderG_A,self.netDiscriminatorDecoderG_B], False)

    def d_logistic_loss(self,real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()


    def d_r1_loss(self,real_pred, real_img):
        grad_real, = torch.autograd.grad(
            outputs=real_pred.sum(), inputs=real_img#, create_graph=True,allow_unused=True
        )
        
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        
        return grad_penalty


    def g_nonsaturating_loss(self,fake_pred):
        loss = F.softplus(-fake_pred).mean()
        return loss


    def g_path_regularize(self,fake_img, latents, mean_path_length, decay=0.01):
        noise = torch.randn_like(fake_img) / math.sqrt(
            fake_img.shape[2] * fake_img.shape[3]
        )
        #print(noise.shape)
        
        #print(fake_img.shape)
        noise.requires_grad=True
        #latents.requires_grad=True
        #print(latents.shape)
        #print((fake_img * noise).sum())
        #print(latents.grad)
        #print((fake_img * noise).sum().grad)
        grad, = torch.autograd.grad(
            outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True#,allow_unused=True
        )
        #print(grad)
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

        path_penalty = (path_lengths - path_mean).pow(2).mean()

        return path_penalty, path_mean.detach(), path_lengths

    def make_noise(self,batch, latent_dim, n_noise, device):
        
        if n_noise == 1:
            return torch.randn(batch, latent_dim, device=device)

        noises = torch.randn(n_noise, batch, latent_dim, device=device)#.unbind(0)

        #print('ok')

        return noises

    def mixing_noise(self,batch, latent_dim, prob, device):
        log_size = int(math.log(128, 2))
        n_latent = log_size * 2 - 2
        temp = random.random()
        #temp=0.95
        #print('temp',temp)
        #print(prob)
        temp_noise = self.make_noise(batch, latent_dim, 2, device)
        if prob > 0 and temp < prob:
            #print('ok')
            inject_index = random.randint(1, n_latent - 1)
        else:
            inject_index = n_latent
        #temp_noise = self.make_noise(batch, latent_dim, 1, device)
        #print(temp_noise.shape)
        #print(temp_noise[0].shape)
        latent = temp_noise[0].unsqueeze(1).repeat(1, inject_index, 1)
        latent2 = temp_noise[1].unsqueeze(1).repeat(1, n_latent - inject_index, 1)
        latent = torch.cat([latent, latent2], 1)
        latents = []
        #print(latent.shape)
        return latent

    def reduce_sum(self,tensor):
        if not dist.is_available():
            return tensor

        if not dist.is_initialized():
            return tensor

        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        return tensor

    def get_world_size(self):
        if not dist.is_available():
            return 1

        if not dist.is_initialized():
            return 1

        return dist.get_world_size()

    def gradient_penalty(self,images, output, weight = 10):
        batch_size = images.shape[0]
        gradients = torch.autograd.grad(outputs=output, inputs=images,
                               grad_outputs=torch.ones(output.size()).cuda(),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
