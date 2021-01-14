from torch import nn
from torch.nn import init
import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import make_layers, get_upsample_filter,_crop
from torchvision.models import vgg
import math

class Classifier(nn.Module):
    def __init__(self, input_nc, ndf, nclasses,img_size, norm_layer=nn.BatchNorm2d):
        super(Classifier, self).__init__()

        kw = 3
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        log_size=int(math.log(img_size,2))
        for n in range(log_size-2):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2),
                norm_layer(ndf * nf_mult, affine=True), #beniz: pb with dimensions with batch_size = 1
                nn.LeakyReLU(0.2, True)
            ]
        self.before_linear = nn.Sequential(*sequence)
        
        sequence = [
            nn.Linear(ndf * nf_mult, 1024),
            nn.Linear(1024, nclasses)
        ]

        self.after_linear = nn.Sequential(*sequence)
    
    def forward(self, x):
        bs = x.size(0)
        out = self.after_linear(self.before_linear(x).view(bs, -1))
        return out


class VGG16_FCN8s(nn.Module):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ])

    def __init__(self, num_cls=19, pretrained=True, weights_init=None, 
            output_last_ft=False):
        super().__init__()
        self.output_last_ft = output_last_ft
        self.vgg = make_layers(vgg.cfgs['D'])
        self.vgg_head = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(4096, num_cls, 1)
            )
        self.upscore2 = self.upscore_pool4 = Bilinear(2, num_cls)
        self.upscore8 = Bilinear(8, num_cls)
        self.score_pool4 = nn.Conv2d(512, num_cls, 1)
        for param in self.score_pool4.parameters():
            init.constant_(param, 0)
        self.score_pool3 = nn.Conv2d(256, num_cls, 1)
        for param in self.score_pool3.parameters():
            init.constant_(param, 0)
        
        if pretrained:
            if weights_init is not None:
                self.load_weights(torch.load(weights_init))
            else:
                self.load_base_weights()
 
    def load_base_vgg(self, weights_state_dict):
        vgg_state_dict = self.get_dict_by_prefix(weights_state_dict, 'vgg.')
        self.vgg.load_state_dict(vgg_state_dict)
     
    def load_vgg_head(self, weights_state_dict):
        vgg_head_state_dict = self.get_dict_by_prefix(weights_state_dict, 'vgg_head.') 
        self.vgg_head.load_state_dict(vgg_head_state_dict)
    
    def get_dict_by_prefix(self, weights_state_dict, prefix):
        return {k[len(prefix):]: v 
                for k,v in weights_state_dict.items()
                if k.startswith(prefix)}


    def load_weights(self, weights_state_dict):
        self.load_base_vgg(weights_state_dict)
        self.load_vgg_head(weights_state_dict)

    def split_vgg_head(self):
        self.classifier = list(self.vgg_head.children())[-1]
        self.vgg_head_feat = nn.Sequential(*list(self.vgg_head.children())[:-1])


    def forward(self, x):
        input = x
        x = F.pad(x, (99, 99, 99, 99), mode='constant', value=0)
        intermediates = {}
        fts_to_save = {16: 'pool3', 23: 'pool4'}
        for i, module in enumerate(self.vgg):
            x = module(x)
            if i in fts_to_save:
                intermediates[fts_to_save[i]] = x
       
        ft_to_save = 5 # Dropout before classifier
        last_ft = {}
        for i, module in enumerate(self.vgg_head):
            x = module(x)
            if i == ft_to_save:
                last_ft = x      
        
        _, _, h, w = x.size()
        upscore2 = self.upscore2(x)
        pool4 = intermediates['pool4']
        score_pool4 = self.score_pool4(0.01 * pool4)
        score_pool4c = _crop(score_pool4, upscore2, offset=5)
        fuse_pool4 = upscore2 + score_pool4c
        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        pool3 = intermediates['pool3']
        score_pool3 = self.score_pool3(0.0001 * pool3)
        score_pool3c = _crop(score_pool3, upscore_pool4, offset=9)
        fuse_pool3 = upscore_pool4 + score_pool3c
        upscore8 = self.upscore8(fuse_pool3)
        score = _crop(upscore8, input, offset=31)
        if self.output_last_ft: 
            return score, last_ft
        else:
            return score

    def load_base_weights(self):
        """This is complicated because we converted the base model to be fully
        convolutional, so some surgery needs to happen here."""
        base_state_dict = model_zoo.load_url(vgg.model_urls['vgg16'])
        vgg_state_dict = {k[len('features.'):]: v
                          for k, v in base_state_dict.items()
                          if k.startswith('features.')}
        self.vgg.load_state_dict(vgg_state_dict)
        vgg_head_params = self.vgg_head.parameters()
        for k, v in base_state_dict.items():
            if not k.startswith('classifier.'):
                continue
            if k.startswith('classifier.6.'):
                # skip final classifier output
                continue
            vgg_head_param = next(vgg_head_params)
            vgg_head_param.data = v.view(vgg_head_param.size())

class Bilinear(nn.Module):

    def __init__(self, factor, num_channels):
        super().__init__()
        self.factor = factor
        filter = get_upsample_filter(factor * 2)
        w = torch.zeros(num_channels, num_channels, factor * 2, factor * 2)
        for i in range(num_channels):
            w[i, i] = filter
        self.register_buffer('w', w)

    def forward(self, x):
        return F.conv_transpose2d(x, Variable(self.w), stride=self.factor)


class Classifier_w(nn.Module):                                                                             
    def __init__(self, init_type='normal', init_gain=0.02, gpu_ids=[],img_size_dec=256):
        super(Classifier_w, self).__init__()
        n_w_plus = 2*int(math.log(img_size_dec,2)-1)
        model = [nn.Flatten(),nn.utils.spectral_norm(nn.Linear(n_w_plus*512,1)),nn.LeakyReLU(0.2,True)]
        self.model = init_net(nn.Sequential(*model), init_type, init_gain, gpu_ids)
        
    def forward(self, x):
        out = self.model(x.permute(1,0,2))
        return out
