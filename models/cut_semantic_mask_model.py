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

class CUTSemanticMaskModel(CUTModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.om/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser = CUTModel.modify_commandline_options(parser, is_train=True)
        parser.add_argument('--train_f_s_B', action='store_true', help='if true f_s will be trained not only on domain A but also on domain B')
        parser.add_argument('--no_train_f_s_A', action='store_true', help='if true f_s wont be trained on domain A')
        parser.add_argument('--fs_light',action='store_true', help='whether to use a light (unet) network for f_s')
        parser.add_argument('--lr_f_s', type=float, default=0.0002, help='f_s learning rate')        
        parser.add_argument('--out_mask', action='store_true', help='use loss out mask')
        parser.add_argument('--lambda_out_mask', type=float, default=10.0, help='weight for loss out mask')
        parser.add_argument('--loss_out_mask', type=str, default='L1', help='loss mask')

        parser.add_argument('--contrastive_noise', type=float, default=0.0, help='noise on constrastive classifier')
        parser.add_argument('--lambda_sem', type=float, default=1.0, help='weight for semantic loss')

        return parser
        
    def __init__(self, opt,rank):
        super().__init__(opt,rank)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        losses_G = ['sem']
        if opt.out_mask:
            losses_G += ['out_mask']

        losses_f_s = ['f_s']

        self.loss_names_G += losses_G
        self.loss_names_f_s = losses_f_s

        self.loss_names = self.loss_names_G + self.loss_names_D + self.loss_names_f_s
            
        # define networks (both generator and discriminator)
        if self.isTrain:
            self.netf_s = networks.define_f(opt.input_nc, nclasses=opt.semantic_nclasses, 
                                            init_type=opt.init_type, init_gain=opt.init_gain,
                                            gpu_ids=self.gpu_ids, fs_light=opt.fs_light)

            self.model_names += ['f_s']


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
                for i,cur_loss in enumerate(self.loss_names):
                    self.loss_names[i] = cur_loss + '_avg'
                    setattr(self, "loss_" + self.loss_names[i], 0)

            ###Making groups
            self.group_f_s = NetworkGroup(networks_to_optimize=["f_s"],forward_functions=None,backward_functions=["compute_f_s_loss"],loss_names_list=["loss_names_f_s"],optimizer=["optimizer_f_s"],loss_backward=["loss_f_s"])
            self.networks_groups.append(self.group_f_s)

    def set_input_first_gpu(self,data):
        super().set_input_first_gpu(data)
        self.input_A_label=self.input_A_label[:self.bs_per_gpu]
        if hasattr(self,'input_B_label'):
            self.input_B_label=self.input_B_label[:self.bs_per_gpu]
        
    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        super().data_dependent_initialize(data)
        visual_names_seg_A = ['input_A_label','gt_pred_A','pfB_max']

        if hasattr(self,'input_B_label'):
            visual_names_seg_B = ['input_B_label']
        else:
            visual_names_seg_B = []

        visual_names_seg_B += ['gt_pred_B']

        self.visual_names += [visual_names_seg_A,visual_names_seg_B]
        
        if self.opt.out_mask and self.isTrain:
            visual_names_out_mask_A = ['real_A_out_mask','fake_B_out_mask']
            self.visual_names += [visual_names_out_mask_A]
        
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        super().set_input(input)
        if 'A_label' in input :
            self.input_A_label = input['A_label'].to(self.device).squeeze(1)
        if self.opt.train_f_s_B and 'B_label' in input:
            self.input_B_label = input['B_label'].to(self.device).squeeze(1)

        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        super().forward()
        
        d = 1
        self.pred_real_A = self.netf_s(self.real_A)    
        self.gt_pred_A = F.log_softmax(self.pred_real_A,dim= d).argmax(dim=d)

        self.pred_real_B = self.netf_s(self.real_B)
        self.gt_pred_B = F.log_softmax(self.pred_real_B,dim=d).argmax(dim=d)
            
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
                
    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        super().compute_G_loss()
        
        self.loss_sem = self.opt.lambda_sem*self.criterionf_s(self.pfB, self.input_A_label)
        if not hasattr(self, 'loss_f_s') or self.loss_f_s > self.opt.semantic_threshold:
            self.loss_sem = 0 * self.loss_sem
        self.loss_G += self.loss_sem

        if hasattr(self,'criterionMask'):
            self.loss_out_mask = self.criterionMask( self.real_A_out_mask, self.fake_B_out_mask) * self.opt.lambda_out_mask
            self.loss_G += self.loss_out_mask

    def compute_f_s_loss(self):
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
