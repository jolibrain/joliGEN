from models.modules.segformer.shape_convert import nlc_to_nchw


def configure_compute_feat_mit(obj):
    def compute_feat_mit(x, extract_layer_ids=[]):
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

        return outs, feats

    obj.compute_feat = compute_feat_mit


def configure_new_extract_feat_encoder_encoder(obj):
    def new_extract_feat_encoder_encoder(img, extract_layer_ids=[]):
        """Extract features from images."""
        x = obj.backbone.compute_feat(img, extract_layer_ids)
        x, feats = x
        if obj.with_neck:
            x = obj.neck(x)
        return x, feats

    obj.extract_feat = new_extract_feat_encoder_encoder


def configure_decode_encoder_encoder(obj):
    def decode_encoder_encoder(outs):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        out = obj._decode_head_forward_test(outs, img_metas=None)

        return out

    obj.decode = decode_encoder_encoder


def configure_decode_2_encoder_encoder(obj):
    def decode_2_encoder_encoder(outs):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        out2 = obj._auxiliary_head_forward_test(outs, img_metas=None)

        return out2

    obj.decode_2 = decode_2_encoder_encoder


def configure_new_auxiliary_head_forward_test_encoder_decoder(obj):
    def new_auxiliary_head_forward_test(x, img_metas):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        seg_logits = obj.auxiliary_head.forward_test(x, img_metas, obj.test_cfg)
        return seg_logits

    obj._auxiliary_head_forward_test = new_auxiliary_head_forward_test


def configure_encoder_decoder(obj):
    configure_new_extract_feat_encoder_encoder(obj)
    configure_decode_encoder_encoder(obj)
    configure_decode_2_encoder_encoder(obj)
    configure_new_auxiliary_head_forward_test_encoder_decoder(obj)


def configure_mit(obj):
    configure_compute_feat_mit(obj)
