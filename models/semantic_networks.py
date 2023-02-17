import os

from .modules.classifiers import (
    Classifier,
    VGG16_FCN8s,
    torch_model,
    TORCH_MODEL_CLASSES,
)
from .modules.UNet_classification import UNet
from .modules.segformer.segformer_generator import Segformer

from .modules.utils import init_net, get_weights


def define_C(
    model_output_nc,
    cls_nf,
    data_crop_size,
    cls_semantic_nclasses,
    train_sem_cls_template,
    model_init_type,
    model_init_gain,
    train_sem_cls_pretrained,
    **unused_options
):
    img_size = data_crop_size
    if train_sem_cls_template == "basic":
        netC = Classifier(model_output_nc, cls_nf, cls_semantic_nclasses, img_size)
    else:
        netC = torch_model(
            model_output_nc,
            cls_nf,
            cls_semantic_nclasses,
            img_size,
            train_sem_cls_template,
            train_sem_cls_pretrained,
        )
    return init_net(netC, model_init_type, model_init_gain)


def define_f(
    f_s_net,
    model_input_nc,
    f_s_semantic_nclasses,
    model_init_type,
    model_init_gain,
    f_s_config_segformer,
    f_s_weight_segformer,
    jg_dir,
    data_crop_size,
    **unused_options
):
    if f_s_net == "vgg":
        net = VGG16_FCN8s(
            f_s_semantic_nclasses,
            pretrained=False,
            weights_init=None,
            output_last_ft=False,
        )
    elif f_s_net == "unet":
        net = UNet(classes=f_s_semantic_nclasses, input_nc=model_input_nc)
    elif f_s_net == "segformer":
        net = Segformer(
            jg_dir,
            f_s_config_segformer,
            model_input_nc,
            img_size=data_crop_size,
            num_classes=f_s_semantic_nclasses,
            final_conv=False,
        )
        if f_s_weight_segformer:
            weight_path = os.path.join(jg_dir, f_s_weight_segformer)
            if not os.path.exists(weight_path):
                print("Downloading pretrained segformer weights for f_s.")
                download_segformer_weight(weight_path)

            weights = get_weights(weight_path)

            try:
                net.net.load_state_dict(weights, strict=False)
            except:
                print(
                    "f_s pretrained segformer decode_head size may have the wrong number of classes, fixing"
                )
                pretrained_dict = {k: v for k, v in weights.items() if k in weights}
                decode_head_keys = []
                for k in pretrained_dict.keys():
                    if "decode_head" in k:
                        decode_head_keys.append(k)
                for k in decode_head_keys:
                    del weights[k]

                net.net.load_state_dict(weights, strict=False)
        return net

    return init_net(net, model_init_type, model_init_gain)
