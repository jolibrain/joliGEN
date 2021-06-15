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
from util.network_group import NetworkGroup

class CUTSemanticModel(CUTModel):
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
        parser.add_argument('--train_cls_B', action='store_true', help='if true cls will be trained not only on domain A but also on domain B')
        parser.add_argument('--cls_template', help='classifier/regressor model type, from torchvision (resnet18, ...), default is custom simple model', default='basic')
        parser.add_argument('--cls_pretrained', action='store_true', help='whether to use a pretrained model, available for non "basic" model only')    
        parser.add_argument('--lr_f_s', type=float, default=0.0002, help='f_s learning rate')
        parser.add_argument('--contrastive_noise', type=float, default=0.0, help='noise on constrastive classifier')
        parser.add_argument('--regression', action='store_true', help='if true cls will be a regressor and not a classifier')
        parser.add_argument('--lambda_sem', type=float, default=1.0, help='weight for semantic loss')
        parser.add_argument('--l1_regression', action='store_true', help='if true l1 loss will be used to compute regressor loss')

        return parser

    def __init__(self, opt):
        super().__init__(opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>

        if self.opt.iter_size == 1:
            losses_G = ['sem']

            losses_CLS = ['CLS']            
            
        else:
            losses_G = ['sem_avg']

            losses_CLS = ['CLS_avg']            

        self.loss_names_G += losses_G
        self.loss_names_CLS = losses_CLS
        self.loss_names = self.loss_names_G + self.loss_names_CLS + self.loss_names_D

        # define networks (both generator and discriminator)
        if self.isTrain:
            self.netCLS = networks.define_C(opt.output_nc, opt.ndf,opt.crop_size,
                                            init_type=opt.init_type, init_gain=opt.init_gain,
                                            gpu_ids=self.gpu_ids, nclasses=opt.semantic_nclasses)
 
            # define loss functions
            self.criterionCLS = torch.nn.modules.CrossEntropyLoss()

            self.optimizer_CLS = torch.optim.Adam(self.netCLS.parameters(), lr=opt.lr_f_s, betas=(opt.beta1, 0.999))
            
            if opt.regression:
                if opt.l1_regression:
                    self.criterionCLS = torch.nn.L1Loss()
                else:
                    self.criterionCLS = torch.nn.modules.MSELoss()
            else:
                self.criterionCLS = torch.nn.modules.CrossEntropyLoss()
            
            self.optimizers.append(self.optimizer_CLS)

            if self.opt.iter_size > 1 :
                self.iter_calculator = IterCalculator(self.loss_names)
                for loss_name in self.loss_names:
                    setattr(self, "loss_" + loss_name, 0)
            
            self.niter=0

            ###Making groups            
            self.group_CLS = NetworkGroup(networks_to_optimize=["CLS"],forward_functions=None,backward_functions=["compute_CLS_loss"],loss_names_list=["loss_names_CLS"],optimizer=["optimizer_CLS"],loss_backward=["loss_CLS"])
            self.networks_groups.append(self.group_CLS)

    def set_input_first_gpu(self,data):
        super().set_input_first_gpu(data)
        self.input_A_label=self.input_A_label[:self.bs_per_gpu]
        if hasattr(self,'input_B_label'):
            self.input_B_label=self.input_B_label[:self.bs_per_gpu]
            
            
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        super().set_input(input)
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
        super().forward()
        d = 1
        self.pred_real_A = self.netCLS(self.real_A)
        if not self.opt.regression:
               _,self.gt_pred_A = self.pred_real_A.max(1)
        
        self.pred_fake_B = self.netCLS(self.fake_B)
        if not self.opt.regression:
            _,self.pfB = self.pred_fake_B.max(1)
            
    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        super().compute_G_loss()
        if not self.opt.regression:
            self.loss_sem = self.criterionCLS(self.pred_fake_B, self.input_A_label)
        else:
            self.loss_sem = self.criterionCLS(self.pred_fake_B.squeeze(1), self.input_A_label)
        if not hasattr(self, 'loss_CLS') or self.loss_CLS.detach().item() > self.opt.semantic_threshold:
            self.loss_sem = 0 * self.loss_sem
        self.loss_G += self.loss_sem
    
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
