import torch
import itertools
from util.image_pool import ImagePool
from .cycle_gan_model import CycleGANModel
from . import networks
from torch.autograd import Variable
import numpy as np
from .modules import loss
from util.iter_calculator import IterCalculator

class CycleGANSemanticModel(CycleGANModel):
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
        parser = CycleGANModel.modify_commandline_options(parser,is_train)
        if is_train:
            parser.add_argument('--train_cls_B', action='store_true', help='if true cls will be trained not only on domain A but also on domain B, if true use_label_B needs to be True')
            parser.add_argument('--cls_template', help='classifier/regressor model type, from torchvision (resnet18, ...), default is custom simple model', default='basic')
            parser.add_argument('--cls_pretrained', action='store_true', help='whether to use a pretrained model, available for non "basic" model only')
            parser.add_argument('--lr_f_s', type=float, default=0.0002, help='f_s learning rate')
            parser.add_argument('--regression', action='store_true', help='if true cls will be a regressor and not a classifier')
            parser.add_argument('--lambda_sem', type=float, default=1.0, help='weight for semantic loss')
            parser.add_argument('--lambda_CLS', type=float, default=1.0, help='weight for CLS loss')
            parser.add_argument('--l1_regression', action='store_true', help='if true l1 loss will be used to compute regressor loss')
            
        return parser
    
    def __init__(self, opt):
        super().__init__(opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        if self.opt.iter_size == 1:
            losses_G = ['sem_AB', 'sem_BA']
            losses_CLS = ['CLS']            
        else:
            losses_G = ['sem_AB_avg', 'sem_BA_avg']
            losses_CLS = ['CLS_avg']

        self.loss_names_G += losses_G
        self.loss_names_CLS = losses_CLS

        self.loss_names = self.loss_names_G + self.loss_names_D + self.loss_names_CLS
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names += ['CLS']
            
        if self.isTrain:
            self.netCLS = networks.define_C(opt.output_nc, opt.ndf,opt.crop_size,
                                            init_type=opt.init_type, init_gain=opt.init_gain,
                                            gpu_ids=self.gpu_ids, nclasses=opt.semantic_nclasses,
                                            template=opt.cls_template, pretrained=opt.cls_pretrained)
 
        if self.isTrain:
            if opt.regression:
                if opt.l1_regression:
                    self.criterionCLS = torch.nn.L1Loss()
                else:
                    self.criterionCLS = torch.nn.modules.MSELoss()
            else:
                self.criterionCLS = torch.nn.modules.CrossEntropyLoss()
                
            # initialize optimizers
            self.optimizer_CLS = torch.optim.Adam(self.netCLS.parameters(), lr=opt.lr_f_s, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_CLS)

            self.rec_noise = opt.rec_noise

            if self.opt.iter_size > 1 :
                self.iter_calculator = IterCalculator(self.loss_names)
                for loss_name in self.loss_names:
                    setattr(self, "loss_" + loss_name, 0)

                    
    def set_input(self, input):
        super().set_input(input)
        if 'A_label' in input:
            if not self.opt.regression:
                self.input_A_label = input['A_label'].to(self.device)
            else:
                self.input_A_label = input['A_label'].to(torch.float).to(device=self.device)
        if 'B_label' in input:
            if not self.opt.regression:
                self.input_B_label = input['B_label'].to(self.device)
            else:
                self.input_B_label = input['B_label'].to(torch.float).to(device=self.device)
                
                
    def forward(self):
        super().forward()

        if self.isTrain:
           self.pred_real_A = self.netCLS(self.real_A)
           if not self.opt.regression:
               _,self.gt_pred_A = self.pred_real_A.max(1)
           self.pred_real_B = self.netCLS(self.real_B)
           if not self.opt.regression:
               _,self.gt_pred_B = self.pred_real_B.max(1)
           self.pred_fake_A = self.netCLS(self.fake_A)
           self.pred_fake_B = self.netCLS(self.fake_B)

           if not self.opt.regression:
               _,self.pfB = self.pred_fake_B.max(1) #beniz: unused ?
        
    
    def compute_CLS_loss(self):
        label_A = self.input_A_label
        # forward only real source image through semantic classifier
        pred_A = self.netCLS(self.real_A)
        if not self.opt.regression:
            self.loss_CLS = self.opt.lambda_CLS * self.criterionCLS(pred_A, label_A)
        else:
            self.loss_CLS = self.opt.lambda_CLS * self.criterionCLS(pred_A.squeeze(1), label_A)
        if self.opt.train_cls_B:
            label_B = self.input_B_label
            pred_B = self.netCLS(self.real_B)
            if not self.opt.regression:
                self.loss_CLS += self.opt.lambda_CLS * self.criterionCLS(pred_B, label_B)
            else:
                self.loss_CLS += self.opt.lambda_CLS * self.criterionCLS(pred_B.squeeze(1), label_B)

    def compute_G_loss(self):
        super().compute_G_loss()
        
        # semantic loss AB
        if not self.opt.regression:
            self.loss_sem_AB = self.criterionCLS(self.pred_fake_B, self.input_A_label)
        else:
            self.loss_sem_AB = self.criterionCLS(self.pred_fake_B.squeeze(1), self.input_A_label)
            
        # semantic loss BA
        if hasattr(self,'input_B_label'):
            if not self.opt.regression:
                self.loss_sem_BA = self.criterionCLS(self.pred_fake_A, self.input_B_label)
            else:
                self.loss_sem_BA = self.criterionCLS(self.pred_fake_A.squeeze(1), self.input_B_label)
        else:
            if not self.opt.regression:
                self.loss_sem_BA = self.criterionCLS(self.pred_fake_A, self.gt_pred_B)
            else:
                self.loss_sem_BA = self.criterionCLS(self.pred_fake_A.squeeze(1), self.pred_real_B.squeeze(1))
                
        # only use semantic loss when classifier has reasonably low loss
        #if True:
        if not hasattr(self, 'loss_CLS') or self.loss_CLS.detach().item() > self.opt.semantic_threshold:
            self.loss_sem_AB = 0 * self.loss_sem_AB 
            self.loss_sem_BA = 0 * self.loss_sem_BA 

        self.loss_sem_AB *= self.opt.lambda_sem
        self.loss_sem_BA *= self.opt.lambda_sem
            
        self.loss_G += self.loss_sem_BA + self.loss_sem_AB

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.niter = self.niter +1
        
        # G_A and G_B
        
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netG_A, self.netG_B], True)
        self.set_requires_grad([self.netCLS], False)
        
        self.forward()      # compute fake images and reconstruction images.
        
        self.compute_G_loss()             # calculate gradients for G_A and G_B
        (self.loss_G/self.opt.iter_size).backward()
        
        self.compute_step(self.optimizer_G,self.loss_names_G)
        
        # D_A and D_B
        self.set_requires_grad([self.netG_A, self.netG_B], False)            
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.set_requires_grad([self.netCLS], False)

        if self.opt.use_contrastive_loss_D:
            self.compute_D_contrastive_loss()      # calculate gradients for D_A and D_B
        else:
            self.compute_D_loss()      # calculate gradients for D
        (self.loss_D/self.opt.iter_size).backward()
        
        self.compute_step(self.optimizer_D,self.loss_names_D)
                
        # CLS
        self.set_requires_grad([self.netG_A, self.netG_B], False)            
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.set_requires_grad([self.netCLS], True)

        self.compute_CLS_loss()
        (self.loss_CLS/self.opt.iter_size).backward()

        self.compute_step(self.optimizer_CLS,self.loss_names_CLS)


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

        current_batch_size=self.get_current_batch_size()
        
        temp=torch.cat((pred_real_A.flatten().unsqueeze(1),pred_fake_A.flatten().unsqueeze(0).repeat(self.nb_preds*current_batch_size,1)),dim=1)
        loss_D_real_A = self.cross_entropy_loss(temp,torch.zeros(temp.shape[0], dtype=torch.long,device=temp.device)).mean()

        temp=torch.cat((-pred_fake_A.flatten().unsqueeze(1),-pred_real_A.flatten().unsqueeze(0).repeat(self.nb_preds*current_batch_size,1)),dim=1)
        loss_D_fake_B = self.cross_entropy_loss(temp,torch.zeros(temp.shape[0], dtype=torch.long,device=temp.device)).mean()
        
        temp=torch.cat((pred_real_B.flatten().unsqueeze(1),pred_fake_B.flatten().unsqueeze(0).repeat(self.nb_preds*current_batch_size,1)),dim=1)
        loss_D_real_B = self.cross_entropy_loss(temp,torch.zeros(temp.shape[0], dtype=torch.long,device=temp.device)).mean()
        
        temp=torch.cat((-pred_fake_B.flatten().unsqueeze(1),-pred_real_B.flatten().unsqueeze(0).repeat(self.nb_preds*current_batch_size,1)),dim=1)
        loss_D_fake_A = self.cross_entropy_loss(temp,torch.zeros(temp.shape[0], dtype=torch.long,device=temp.device)).mean()
        
        # combine loss and calculate gradients
        self.loss_D_A = (loss_D_fake_A + loss_D_real_A) * 0.5
        self.loss_D_B = (loss_D_fake_B + loss_D_real_B) * 0.5

        self.loss_D = self.loss_D_A + self.loss_D_B
