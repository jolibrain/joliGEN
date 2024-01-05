import torch

from models import gan_networks, diffusion_networks


class ConsistencyWrapper(torch.nn.Module):
    """
    Consistency model wrapper for onnx & jit trace
    """

    def __init__(self, model, sigmas):
        super().__init__()
        self.model = model
        self.sigmas = sigmas

    def forward(self, x, mask):
        return self.model.restoration(x, None, self.sigmas, mask)


def export(opt, cuda, model_in_file, model_out_file, opset_version, export_type):
    if opt.model_type == "palette":
        raise ValueError('export() is not supported for model type "palette"')

    if opt.model_type == "cm":
        opt.alg_palette_sampling_method = ""
        opt.alg_diffusion_cond_embed = opt.alg_diffusion_cond_image_creation
        opt.alg_diffusion_cond_embed_dim = 256

        model = diffusion_networks.define_G(**vars(opt))
    else:
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

    dummy_image = torch.randn(
        1, opt.model_input_nc, opt.data_crop_size, opt.data_crop_size, device=device
    )
    dummy_inputs = [dummy_image]

    if opt.model_type == "cm":
        # at the moment, consistency models have two inputs: origin image and mask
        # TODO allow to change number of sigmas
        sigmas = [80.0, 24.4, 5.84, 0.9, 0.661]
        model = ConsistencyWrapper(model, sigmas)
        dummy_inputs += [
            torch.randn(
                1,
                opt.model_input_nc,
                opt.data_crop_size,
                opt.data_crop_size,
                device=device,
            ),
        ]

    dummy_inputs = tuple(dummy_inputs)

    if export_type == "onnx":
        torch.onnx.export(
            model,
            dummy_inputs,
            model_out_file,
            verbose=False,
            opset_version=opset_version,
        )

    elif export_type == "jit":
        jit_model = torch.jit.trace(model, dummy_inputs)
        jit_model.save(model_out_file)

    else:
        raise ValueError(f"{export_type} is not available")
