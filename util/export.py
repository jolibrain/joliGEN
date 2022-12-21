import torch


from models import gan_networks


def export(opt, cuda, model_in_file, model_out_file, opset_version, export_type):
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

    if export_type == "onnx":
        torch.onnx.export(
            model,
            dummy_input,
            model_out_file,
            verbose=False,
            opset_version=opset_version,
        )

    elif export_type == "jit":
        jit_model = torch.jit.trace(model, dummy_input)
        jit_model.save(model_out_file)

    else:
        raise ValueError(f"{export_type} is not available")
