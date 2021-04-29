import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
from .modules.utils import get_scheduler
from torchviz import make_dot

#for FID
from data.base_dataset import get_transform
from .modules.fid.pytorch_fid.fid_score import _compute_statistics_of_path,calculate_frechet_distance
from util.util import save_image,tensor2im
import numpy as np

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        if hasattr(opt, 'disc_in_mask'):
            self.disc_in_mask = opt.disc_in_mask
        else:
            self.disc_in_mask = False
        if hasattr(opt,'fs_light'):
            self.fs_light = opt.fs_light
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.display_param = []
        self.set_display_param()
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

        if opt.compute_fid:
            self.transform = get_transform(opt, grayscale=(opt.input_nc == 1))
            dims=2048
            batch=1
            self.netFid=networks.define_inception(self.gpu_ids[0],dims)
            pathA=opt.dataroot + '/trainA'
            if not os.path.isfile(opt.checkpoints_dir+'fid_mu_sigma_A.npz'):
                self.realmA,self.realsA=_compute_statistics_of_path(pathA, self.netFid, batch, dims, self.gpu_ids[0],self.transform,nb_max_img=opt.nb_img_max_fid)
                np.savez(opt.checkpoints_dir+'fid_mu_sigma_A.npz', mu=self.realmA, sigma=self.realsA)
            else:

                print('Mu and sigma loaded for domain A')
                self.realmA,self.realsA=_compute_statistics_of_path(opt.checkpoints_dir+'fid_mu_sigma_A.npz', self.netFid, batch, dims, self.gpu_ids[0],self.transform,nb_max_img=opt.nb_img_max_fid)
                
            pathB=opt.dataroot + '/trainB'
            if not os.path.isfile(opt.checkpoints_dir+'fid_mu_sigma_B.npz'):
                self.realmB,self.realsB=_compute_statistics_of_path(pathB, self.netFid, batch, dims, self.gpu_ids[0],self.transform,nb_max_img=opt.nb_img_max_fid)
                np.savez(opt.checkpoints_dir+'fid_mu_sigma_B.npz', mu=self.realmB, sigma=self.realsB)
            else:

                print('Mu and sigma loaded for domain B')
                self.realmB,self.realsB=_compute_statistics_of_path(opt.checkpoints_dir+'fid_mu_sigma_B.npz', self.netFid, batch, dims, self.gpu_ids[0],self.transform,nb_max_img=opt.nb_img_max_fid)
                
            pathA=self.save_dir + '/fakeA/'
            if not os.path.exists(pathA):
                os.mkdir(pathA)

            pathB=self.save_dir + '/fakeB/'
            if not os.path.exists(pathB):
                os.mkdir(pathB)
            self.fidA=0
            self.fidB=0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def parallelize(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                setattr(self, 'net' + name, torch.nn.DataParallel(net, self.opt.gpu_ids))

        
    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr_G = self.optimizers[0].param_groups[0]['lr']
        #lr_D = self.optimizers[1].param_groups[0]['lr']
        #print('learning rate G = %.7f' % lr_G, ' / learning rate D = %.7f' % lr_D)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_display_param(self):
        param = OrderedDict()
        for name in self.display_param:
            if isinstance(name, str):
                param[name] = getattr(self.opt, name)
        return param

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                    
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

                
                if hasattr(state_dict, 'g_ema'):
                    net.load_state_dict(state_dict['g_ema'])
                else:
                    net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def save_networks_img(self,data):
        self.set_input(data)
        paths=[]
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            path=self.opt.checkpoints_dir + self.opt.name +'/networks/' + name
            if not 'Decoder' in name:
                temp=net(self.real_A)
            else:
                temp=net(self.netG_A(self.real_A).detach()) #decoders take w+ in input
            make_dot(temp,params=dict(net.named_parameters())).render(path, format='png')
            paths.append(path)
        return paths

    def set_display_param(self,params=None):
        if params is None:
            params = vars(self.opt).keys()
        for param in params:
            self.display_param.append(param)
        self.display_param.sort()


    def compute_fid(self,n_epoch,n_iter):
        dims=2048
        batch=1
        pathA=self.save_dir + '/fakeA/'+str(n_iter)+'_' +str(n_epoch)
        if not os.path.exists(pathA):
            os.mkdir(pathA)
        for i,temp_fake_A in enumerate(self.fake_A_pool.get_all()):
            save_image(tensor2im(temp_fake_A), pathA+'/'+str(i)+'.png', aspect_ratio=1.0)
        self.fakemA,self.fakesA=_compute_statistics_of_path(pathA, self.netFid, batch, dims, self.gpu_ids[0],nb_max_img=self.opt.nb_img_max_fid)
            
        pathB=self.save_dir + '/fakeB/'+str(n_iter)+'_' +str(n_epoch)
        if not os.path.exists(pathB):
            os.mkdir(pathB)
            
        for j,temp_fake_B in enumerate(self.fake_B_pool.get_all()):
            save_image(tensor2im(temp_fake_B), pathB+'/'+str(j)+'.png', aspect_ratio=1.0)
        self.fakemB,self.fakesB=_compute_statistics_of_path(pathB, self.netFid, batch, dims, self.gpu_ids[0],nb_max_img=self.opt.nb_img_max_fid)

        self.fidA=calculate_frechet_distance(self.realmA, self.realsA,self.fakemA,self.fakesA)
        self.fidB=calculate_frechet_distance(self.realmB, self.realsB,self.fakemB,self.fakesB)

    def get_current_fids(self):
        
        fids = OrderedDict()
        for name in ['fidA','fidB']:
            if isinstance(name, str):
                fids[name] = float(getattr(self, name))  # float(...) works for both scalar tensor and float number
        return fids

    def compute_step(self,optimizers,loss_names):
        if not isinstance(optimizers,list):
            optimizers = [optimizers]
        if self.niter % self.opt.iter_size ==0:
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            if self.opt.iter_size > 1:
                self.iter_calculator.compute_last_step(loss_names)
                for loss_name in loss_names:
                    setattr(self, "loss_" + loss_name , getattr(self.iter_calculator, "loss_" + loss_name ))               
        elif self.opt.iter_size > 1:
            for loss_name in loss_names:
                value=getattr(self, "loss_" + loss_name[:-4])/self.opt.iter_size
                if torch.is_tensor(value):
                    value = value.detach()                    
                self.iter_calculator.compute_step(loss_name,value)

    def get_current_batch_size(self):
        return self.real_A.shape[0]
