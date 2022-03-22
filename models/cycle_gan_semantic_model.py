import torch
import itertools
from util.image_pool import ImagePool
from .cycle_gan_model import CycleGANModel
from . import networks
from torch.autograd import Variable
import numpy as np
from .modules import loss
from util.iter_calculator import IterCalculator
from util.network_group import NetworkGroup

class CycleGANSemanticModel(CycleGANModel):
    #def name(self):
    #    return 'CycleGANModel'

    # new, copied from cyclegan model
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CycleGANModel.modify_commandline_options(parser,is_train)
        return parser
    
    def __init__(self, opt,rank):
        super().__init__(opt,rank)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        losses_G = ['sem_AB', 'sem_BA']
        losses_CLS = ['CLS']            
        
        self.loss_names_G += losses_G
        self.loss_names_CLS = losses_CLS

        self.loss_names = self.loss_names_G + self.loss_names_D + self.loss_names_CLS
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names += ['CLS']
            
        if self.isTrain:
            self.netCLS = networks.define_C(**vars(opt))
 
        if self.isTrain:
            if opt.train_sem_regression:
                if opt.train_sem_l1_regression:
                    self.criterionCLS = torch.nn.L1Loss()
                else:
                    self.criterionCLS = torch.nn.modules.MSELoss()
            else:
                self.criterionCLS = torch.nn.modules.CrossEntropyLoss()
                
            # initialize optimizers
            self.optimizer_CLS = torch.optim.Adam(self.netCLS.parameters(), lr=opt.train_sem_lr_f_s, betas=(opt.train_beta1, opt.train_beta2))
            self.optimizers.append(self.optimizer_CLS)

            self.rec_noise = opt.alg_cyclegan_rec_noise

            if self.opt.train_iter_size > 1 :
                self.iter_calculator = IterCalculator(self.loss_names)
                for i,cur_loss in enumerate(self.loss_names):
                    self.loss_names[i] = cur_loss + '_avg'
                    setattr(self, "loss_" + self.loss_names[i], 0)

            ###Making groups
            self.group_CLS = NetworkGroup(networks_to_optimize=["CLS"],forward_functions=None,backward_functions=["compute_CLS_loss"],loss_names_list=["loss_names_CLS"],optimizer=["optimizer_CLS"],loss_backward=["loss_CLS"])
            self.networks_groups.append(self.group_CLS)
                    
    def set_input(self, input):
        super().set_input(input)
        if 'A_label' in input:
            if not self.opt.train_sem_regression:
                self.input_A_label = input['A_label'].to(self.device)
            else:
                self.input_A_label = input['A_label'].to(torch.float).to(device=self.device)
        if 'B_label' in input:
            if not self.opt.train_sem_regression:
                self.input_B_label = input['B_label'].to(self.device)
            else:
                self.input_B_label = input['B_label'].to(torch.float).to(device=self.device)
                
                
    def forward(self):
        super().forward()

        if self.isTrain:
           self.pred_real_A = self.netCLS(self.real_A)
           if not self.opt.train_sem_regression:
               _,self.gt_pred_A = self.pred_real_A.max(1)
           self.pred_real_B = self.netCLS(self.real_B)
           if not self.opt.train_sem_regression:
               _,self.gt_pred_B = self.pred_real_B.max(1)
           self.pred_fake_A = self.netCLS(self.fake_A)
           self.pred_fake_B = self.netCLS(self.fake_B)

           if not self.opt.train_sem_regression:
               _,self.pfB = self.pred_fake_B.max(1) #beniz: unused ?
        
    
    def compute_CLS_loss(self):
        label_A = self.input_A_label
        # forward only real source image through semantic classifier
        pred_A = self.netCLS(self.real_A)
        if not self.opt.train_sem_regression:
            self.loss_CLS = self.opt.train_sem_lambda * self.criterionCLS(pred_A, label_A)
        else:
            self.loss_CLS = self.opt.train_sem_lambda * self.criterionCLS(pred_A.squeeze(1), label_A)
        if self.opt.train_sem_cls_B:
            label_B = self.input_B_label
            pred_B = self.netCLS(self.real_B)
            if not self.opt.train_sem_regression:
                self.loss_CLS += self.opt.train_sem_lambda * self.criterionCLS(pred_B, label_B)
            else:
                self.loss_CLS += self.opt.train_sem_lambda * self.criterionCLS(pred_B.squeeze(1), label_B)

    def compute_G_loss(self):
        super().compute_G_loss()
        
        # semantic loss AB
        if not self.opt.train_sem_regression:
            self.loss_sem_AB = self.criterionCLS(self.pred_fake_B, self.input_A_label)
        else:
            self.loss_sem_AB = self.criterionCLS(self.pred_fake_B.squeeze(1), self.input_A_label)
            
        # semantic loss BA
        if hasattr(self,'input_B_label'):
            if not self.opt.train_sem_regression:
                self.loss_sem_BA = self.criterionCLS(self.pred_fake_A, self.input_B_label)
            else:
                self.loss_sem_BA = self.criterionCLS(self.pred_fake_A.squeeze(1), self.input_B_label)
        else:
            if not self.opt.train_sem_regression:
                self.loss_sem_BA = self.criterionCLS(self.pred_fake_A, self.gt_pred_B)
            else:
                self.loss_sem_BA = self.criterionCLS(self.pred_fake_A.squeeze(1), self.pred_real_B.squeeze(1))
                
        # only use semantic loss when classifier has reasonably low loss
        #if True:
        if not hasattr(self, 'loss_CLS') or self.loss_CLS > self.opt.f_s_semantic_threshold:
            self.loss_sem_AB = 0 * self.loss_sem_AB 
            self.loss_sem_BA = 0 * self.loss_sem_BA 

        self.loss_sem_AB *= self.opt.train_sem_lambda
        self.loss_sem_BA *= self.opt.train_sem_lambda
            
        self.loss_G += self.loss_sem_BA + self.loss_sem_AB
