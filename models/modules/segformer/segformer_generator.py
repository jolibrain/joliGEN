import os
from mmseg.models import build_segmentor
from torch import nn
import mmcv

from models.modules.resnet_architecture.resnet_generator import ResnetDecoder
from models.modules.attn_network import BaseGenerator_attn
from .utils import configure_encoder_decoder,configure_mit
from models.modules.mobile_modules import SeparableConv2d

class Segformer(nn.Module):
    def __init__(self,opt,num_classes=10,final_conv=False):
        super().__init__()
        self.opt = opt
        cfg = mmcv.Config.fromfile(os.path.join(self.opt.jg_dir,self.opt.G_config_segformer))
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        cfg.model.decode_head.num_classes = num_classes
        self.net = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

        configure_encoder_decoder(self.net)
        self.net.img_size = self.opt.data_crop_size
        configure_mit(self.net.backbone)

        self.use_final_conv = final_conv
        
        if self.use_final_conv:
            self.final_conv = ResnetDecoder(num_classes, 3, ngf=64)

    def compute_feats(self, input, extract_layer_ids=[]):
        outs,feats = self.net.extract_feat(input, extract_layer_ids)
        return outs,feats

    def forward(self, input):
        outs,_ = self.compute_feats(input)
        out = self.net.decode(outs,use_resize=not self.use_final_conv)
        if self.use_final_conv:
            out = self.final_conv(out)
        return out

    def get_feats(self, input, extract_layer_ids):
        _,feats = self.compute_feats(input, extract_layer_ids)        
        return feats

class SegformerGenerator_attn(BaseGenerator_attn):
    # initializers
    def __init__(self,opt=None,final_conv=False): #nb_mask_attn : total number of attention masks, nb_mask_input :number of attention mask applied to input img directly
        super(SegformerGenerator_attn, self).__init__(opt)
        self.use_final_conv = final_conv
        self.tanh = nn.Tanh()

        cfg = mmcv.Config.fromfile(os.path.join(self.opt.jg_dir,self.opt.G_config_segformer))
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        cfg.model.auxiliary_head = cfg.model.decode_head.copy()
        if self.use_final_conv:
            num_cls = 256
        else:
            num_cls = 3*(self.nb_mask_attn-self.nb_mask_input)
        cfg.model.decode_head.num_classes = num_cls
        cfg.model.auxiliary_head.num_classes = self.nb_mask_attn
        self.segformer = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        self.segformer.train()
        self.softmax_ = nn.Softmax(dim=1)

        configure_encoder_decoder(self.segformer)
        self.segformer.img_size = self.opt.data_crop_size
        configure_mit(self.segformer.backbone)

        self.use_final_conv = final_conv
        
        if self.use_final_conv:
            self.final_conv = ResnetDecoder(num_cls,3*(self.nb_mask_attn-self.nb_mask_input) , ngf=64)

    def compute_feats(self, input, extract_layer_ids=[]):
        outs,feats = self.segformer.extract_feat(input, extract_layer_ids)
        return outs,feats

    def compute_attention_content(self,outs):
        image = self.segformer.decode(outs,use_resize=not self.use_final_conv)
        if self.use_final_conv:
            image = self.final_conv(image)

        attention = self.segformer.decode_2(outs,use_resize=not self.use_final_conv)
        images = []


        for i in range(self.nb_mask_attn - self.nb_mask_input):
            images.append(image[:, 3*i:3*(i+1), :, :])
            
        attention = self.softmax_(attention)
        attentions =[]
        
        for i in range(self.nb_mask_attn):
            attentions.append(attention[:, i:i+1, :, :].repeat(1, 3, 1, 1))

        return attentions,images
