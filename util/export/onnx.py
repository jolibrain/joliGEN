import torch


from models import gan_networks


def export_onnx(opt, cuda, model_in_file, model_out_file, opset_version):
    model = gan_networks.define_G(**vars(opt))

    model.eval()
    model.load_state_dict(torch.load(model_in_file))

    if cuda:
        model = model.cuda()

    # export to ONNX via tracing
    if cuda:
        device = "cuda"
    else:
        device = "cpu"
    dummy_input = torch.randn(
        1, opt.model_input_nc, opt.data_crop_size, opt.data_crop_size, device=device
    )

    torch.onnx.export(
        model,
        dummy_input,
        model_out_file,
        verbose=False,
        opset_version=opset_version,
    )
