from mmseg.models.utils import nlc_to_nchw
from mmseg.ops import resize

def configure_new_forward_mit(obj):
    def new_forward_mit(x, extract_layer_ids=[]):
        outs = []
        feats = []

        for i, layer in enumerate(obj.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in obj.out_indices:
                outs.append(x)
            if i in extract_layer_ids:
                feats.append(x)

        if len(feats)>0:
            return outs,feats
        else:
            return outs

    obj.forward = new_forward_mit


def configure_new_extract_feat_encoder_encoder(obj):
    def new_extract_feat_encoder_encoder(img,extract_layer_ids=[]):
        """Extract features from images."""
        x = obj.backbone(img,extract_layer_ids)
        if len(extract_layer_ids)>0:
            x,feats = x
            return x,feats
        if obj.with_neck:
            x = obj.neck(x)
        return x
    obj.extract_feat = new_extract_feat_encoder_encoder

def configure_new_encode_decode_encoder_encoder(obj):
    def new_encode_decode_encoder_encoder(img, img_metas, extract_layer_ids=[],use_resize=True):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = obj.extract_feat(img,extract_layer_ids=extract_layer_ids)
        if len(extract_layer_ids)>0:
            x,feats = x
        out = obj._decode_head_forward_test(x, img_metas)
        if use_resize:
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=obj.align_corners)

        if hasattr(obj,'auxiliary_head'):
            out2 = obj._auxiliary_head_forward_test(x, img_metas)
            out2 = resize(
                input=out2,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=obj.align_corners)
            if len(extract_layer_ids)>0:
                return out, out2,feats
            else:
                return out, out2, None

        if len(extract_layer_ids)>0:
            return out,feats
        else:
            return out, None

    obj.encode_decode = new_encode_decode_encoder_encoder


def configure_new_auxiliary_head_forward_test_encoder_decoder(obj):
    def new_auxiliary_head_forward_test(x, img_metas):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        seg_logits = obj.auxiliary_head.forward_test(x, img_metas, obj.test_cfg)
        return seg_logits
    obj._auxiliary_head_forward_test = new_auxiliary_head_forward_test

def configure_encoder_decoder(obj):
    configure_new_extract_feat_encoder_encoder(obj)
    configure_new_encode_decode_encoder_encoder(obj)
    configure_new_auxiliary_head_forward_test_encoder_decoder(obj)
