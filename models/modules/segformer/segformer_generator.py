import os
from mmseg.models import build_segmentor
from torch import nn
import mmcv

from models.modules.resnet_architecture.mobile_resnet_generator import MobileResnetDecoder

from .utils import configure_encoder_decoder,configure_new_forward_mit

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
        configure_new_forward_mit(self.net.backbone)

        self.use_final_conv = final_conv
        
        if self.use_final_conv:
            self.final_conv = MobileResnetDecoder(num_classes, 3, ngf=64)
        

    def forward(self, input, extract_layer_ids=[], encode_only=False):
        out,feats = self.net.encode_decode(input,None, extract_layer_ids=extract_layer_ids,use_resize=not self.use_final_conv)

        if encode_only:
            return feats

        if self.use_final_conv:
            out = self.final_conv(out)
        
        return out

class SegformerGenerator_attn(nn.Module):
    # initializers
    def __init__(self,opt=None,final_conv=False): #nb_attn : total number of attention masks, nb_mask_input :number of attention mask applied to input img directly
        super(SegformerGenerator_attn, self).__init__()
        self.opt = opt
        self.nb_attn = self.opt.G_attn_nb_mask_attn
        self.nb_mask_input = self.opt.G_attn_nb_mask_input
        self.use_final_conv = final_conv
        self.tanh = nn.Tanh()

        cfg = mmcv.Config.fromfile(os.path.join(self.opt.jg_dir,self.opt.G_config_segformer))
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        cfg.model.auxiliary_head = cfg.model.decode_head.copy()
        if self.use_final_conv:
            num_cls = 256
        else:
            num_cls = 3*(self.nb_attn-self.nb_mask_input)
        cfg.model.decode_head.num_classes = num_cls
        cfg.model.auxiliary_head.num_classes = self.nb_attn
        self.segformer = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        self.segformer.train()
        self.softmax_ = nn.Softmax(dim=1)

        configure_encoder_decoder(self.segformer)
        configure_new_forward_mit(self.segformer.backbone)

        self.use_final_conv = final_conv
        
        if self.use_final_conv:
            self.final_conv = MobileResnetDecoder(num_cls,3*(self.nb_attn-self.nb_mask_input) , ngf=64)


    # forward method
    def forward(self, input, extract_layer_ids=[], encode_only=False,get_attention_masks=False):
        image,attention,feats = self.segformer.encode_decode(input,None, extract_layer_ids=extract_layer_ids,use_resize=not self.use_final_conv)
        if encode_only:
            return feats
        images = []

        if self.use_final_conv:
            image = self.final_conv(image)

        for i in range(self.nb_attn - self.nb_mask_input):
            images.append(image[:, 3*i:3*(i+1), :, :])
            
        attention = self.softmax_(attention)
        attentions =[]
        
        for i in range(self.nb_attn):
            attentions.append(attention[:, i:i+1, :, :].repeat(1, 3, 1, 1))

        outputs = []
        
        for i in range(self.nb_attn-self.nb_mask_input):
            outputs.append(images[i]*attentions[i])
        for i in range(self.nb_attn-self.nb_mask_input,self.nb_attn):
            outputs.append(input * attentions[i])

        if get_attention_masks:
            return images,attentions,outputs
            
        o = outputs[0]
        for i in range(1,self.nb_attn):
            o += outputs[i]
        return o

    def get_attention_masks(self,input):
        return self.forward(input,get_attention_masks=True)
