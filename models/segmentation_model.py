import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class SegmentationModel(BaseModel):
    #def name(self):
    #    return 'CycleGANModel'

    # new, copied from cyclegansemantic mask model
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

        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['f_s','f_s']


        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names = ['real_A','input_A_label','gt_pred_A']

        self.visual_names = visual_names
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['f_s']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netf_s = networks.define_f(opt.input_nc, nclasses=opt.semantic_nclasses, 
                                        init_type=opt.init_type, init_gain=opt.init_gain,
                                        gpu_ids=self.gpu_ids)
 
        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            # define loss functions
            self.criterionf_s = torch.nn.modules.NLLLoss()
            self.criterionf_s = torch.nn.modules.CrossEntropyLoss()
            # initialize optimizers
            self.optimizer_f_s = torch.optim.Adam(self.netf_s.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            print('f defined')
            self.optimizers = []
            self.optimizers.append(self.optimizer_f_s)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.input_A_label = input['A_label'].to(self.device).squeeze(1)

    def forward(self):
        self.pred_real_A = self.netf_s(self.real_A)
        d = 1
        self.gt_pred_A = F.log_softmax(self.pred_real_A,dim= d).argmax(dim=d)
        
    
    def backward_f_s(self):
        label_A = self.input_A_label
        # forward only real source image through semantic classifier
        pred_A = self.netf_s(self.real_A)
        self.loss_f_s = self.criterionf_s(pred_A, label_A)#.squeeze(1))
        self.loss_f_s.backward()

        
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        # forward
        self.forward()      # compute fake images and reconostruction images.

        # f_s
        self.set_requires_grad([self.netf_s], True)
        self.optimizer_f_s.zero_grad()
        self.backward_f_s()
        self.optimizer_f_s.step()
