import numpy as np
import torch
from .cut_model import CUTModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from .modules import loss
import torch.nn.functional as F
from util.util import gaussian
from util.iter_calculator import IterCalculator

class CUTSemanticMaskModel(CUTModel):
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
        parser = CUTModel.modify_commandline_options(parser, is_train=True)
        parser.add_argument('--train_f_s_B', action='store_true', help='if true f_s will be trained not only on domain A but also on domain B')
        parser.add_argument('--fs_light',action='store_true', help='whether to use a light (unet) network for f_s')
        parser.add_argument('--lr_f_s', type=float, default=0.0002, help='f_s learning rate')
        parser.add_argument('--D_noise', type=float, default=0.0, help='whether to add instance noise to discriminator inputs')
        
        parser.add_argument('--out_mask', action='store_true', help='use loss out mask')
        parser.add_argument('--lambda_out_mask', type=float, default=10.0, help='weight for loss out mask')
        parser.add_argument('--loss_out_mask', type=str, default='L1', help='loss mask')

        parser.add_argument('--contrastive_noise', type=float, default=0.0, help='noise on constrastive classifier')
        parser.add_argument('--lambda_sem', type=float, default=1.0, help='weight for semantic loss')

        return parser
        
    def __init__(self, opt):
        super().__init__(opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        if self.opt.iter_size == 1:
            losses_G = ['sem']
            losses_f_s = ['f_s']
        else:
            losses_G = ['sem_avg']
            losses_f_s = ['f_s_avg']

        self.loss_names_G += losses_G
        self.loss_names_f_s = losses_f_s

        self.loss_names = self.loss_names_G + self.loss_names_D + self.loss_names_f_s

        self.visual_names += ['input_A_label','gt_pred_A','pfB_max']
            
        if opt.out_mask and self.isTrain:
            self.loss_names += ['out_mask']
            self.visual_names += ['real_A_out_mask','fake_B_out_mask']
        if self.isTrain:
            self.model_names += ['f_s']
            
        # define networks (both generator and discriminator)
        if self.isTrain:
            self.netf_s = networks.define_f(opt.input_nc, nclasses=opt.semantic_nclasses, 
                                            init_type=opt.init_type, init_gain=opt.init_gain,
                                            gpu_ids=self.gpu_ids, fs_light=opt.fs_light)

            # define loss functions
            self.criterionf_s = torch.nn.modules.CrossEntropyLoss()
            
            if opt.out_mask:
                if opt.loss_out_mask == 'L1':
                    self.criterionMask = torch.nn.L1Loss()
                elif opt.loss_out_mask == 'MSE':
                    self.criterionMask = torch.nn.MSELoss()
                elif opt.loss_out_mask == 'Charbonnier':
                    self.criterionMask = L1_Charbonnier_loss(opt.charbonnier_eps)
           
            self.optimizer_f_s = torch.optim.Adam(self.netf_s.parameters(), lr=opt.lr_f_s, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_f_s)

            if self.opt.iter_size > 1 :
                self.iter_calculator = IterCalculator(self.loss_names)
                for loss_name in self.loss_names:
                    setattr(self, "loss_" + loss_name, 0)            

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
            self.compute_D_loss()                  
            self.compute_f_s_loss()                  
            self.compute_G_loss()                   
            self.loss_D.backward()# calculate gradients for D
            self.loss_f_s.backward()# calculate gradients for f_s
            self.loss_G.backward()# calculate gradients for G
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
        self.set_requires_grad(self.netf_s, False)

        # forward
        self.forward()
        
        self.compute_G_loss()
        (self.loss_G/self.opt.iter_size).backward()

        self.compute_step(self.optimizer_G,self.loss_names_G)

        if self.opt.netF == 'mlp_sample' and self.niter % self.opt.iter_size == 0:
            self.optimizer_F.step()
            self.optimizer_F.zero_grad()
                        
        # update D
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netF, False)
        self.set_requires_grad(self.netf_s, False)

        if self.opt.use_contrastive_loss_D:
            self.compute_D_contrastive_loss()      # calculate gradients for D_A and D_B
        else:
            self.compute_D_loss()
        (self.loss_D/self.opt.iter_size).backward()
        
        self.compute_step(self.optimizer_D,self.loss_names_D)
        
        # update f_s
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netF, False)
        self.set_requires_grad(self.netf_s, True)            

        self.compute_f_s_loss()
        (self.loss_f_s/self.opt.iter_size).backward()
        
        self.compute_step(self.optimizer_f_s,self.loss_names_f_s)            
        

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
            self.input_A_label = input['A_label'].to(self.device).squeeze(1)
        if self.opt.train_f_s_B and 'B_label' in input:
            self.input_B_label = input['B_label'].to(self.device).squeeze(1)

        
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
        self.pred_real_A = self.netf_s(self.real_A)    
        self.gt_pred_A = F.log_softmax(self.pred_real_A,dim= d).argmax(dim=d)
            
        self.pred_fake_B = self.netf_s(self.fake_B)
        self.pfB = F.log_softmax(self.pred_fake_B,dim=d)#.argmax(dim=d)
        self.pfB_max = self.pfB.argmax(dim=d)

        if hasattr(self,'criterionMask'):
                label_A = self.input_A_label
                label_A_in = label_A.unsqueeze(1)
                label_A_inv = torch.tensor(np.ones(label_A.size())).to(self.device) - label_A>0.5
                label_A_inv = label_A_inv.unsqueeze(1)
                self.real_A_out_mask = self.real_A *label_A_inv
                self.fake_B_out_mask = self.fake_B *label_A_inv            

                if self.opt.D_noise > 0.0:
                    self.fake_B_noisy = gaussian(self.fake_B, self.opt.D_noise)
                    self.real_B_noisy = gaussian(self.real_B, self.opt.D_noise)
                
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

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            if self.opt.use_contrastive_loss_D:
                current_batch_size=self.get_current_batch_size()
                temp=torch.cat((self.netD(self.real_B).flatten().unsqueeze(1),pred_fake.flatten().unsqueeze(0).repeat(self.nb_preds*current_batch_size,1)),dim=1)
                self.loss_G_GAN = self.cross_entropy_loss(-temp,torch.zeros(temp.shape[0], dtype=torch.long,device=temp.device)).mean()
            else:
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
        self.loss_sem = self.opt.lambda_sem*self.criterionf_s(self.pfB, self.input_A_label)
        if self.loss_f_s.detach().item() > 1.0:
            self.loss_sem = 0 * self.loss_sem
        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_sem

        if hasattr(self,'criterionMask'):
            self.loss_out_mask = self.criterionMask( self.real_A_out_mask, self.fake_B_out_mask) * self.opt.lambda_out_mask
            self.loss_G += self.loss_out_mask

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

    def compute_f_s_loss(self):
        label_A = self.input_A_label
        # forward only real source image through semantic classifier
        pred_A = self.netf_s(self.real_A)
        self.loss_f_s = self.criterionf_s(pred_A, label_A)#.squeeze(1))
        if self.opt.train_f_s_B:
            label_B = self.input_B_label
            pred_B = self.netf_s(self.real_B) 
            self.loss_f_s += self.criterionf_s(pred_B, label_B)#.squeeze(1))

    def compute_D_contrastive_loss(self):
        """Calculate contrastive GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_B = self.fake_B.detach()
        pred_fake_B = self.netD(fake_B)
        
        # Real
        pred_real_B = self.netD(self.real_B)
        
        current_batch_size=self.get_current_batch_size()
        
        temp=torch.cat((pred_real_B.flatten().unsqueeze(1),pred_fake_B.flatten().unsqueeze(0).repeat(self.nb_preds*current_batch_size,1)),dim=1)
        self.loss_D_real = self.cross_entropy_loss(temp,torch.zeros(temp.shape[0], dtype=torch.long,device=temp.device)).mean()
        
        temp=torch.cat((-pred_fake_B.flatten().unsqueeze(1),-pred_real_B.flatten().unsqueeze(0).repeat(self.nb_preds*current_batch_size,1)),dim=1)
        self.loss_D_fake = self.cross_entropy_loss(temp,torch.zeros(temp.shape[0], dtype=torch.long,device=temp.device)).mean()

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
