import torch
from torch import nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig


def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained(
        "stabilityai/sd-turbo", subfolder="scheduler"
    )
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [
            self.skip_conv_1,
            self.skip_conv_2,
            self.skip_conv_3,
            self.skip_conv_4,
        ]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


class Img2ImgTurbo(nn.Module):
    def __init__(self, in_channels, out_channels, lora_rank_unet, lora_rank_vae):
        super().__init__()

        # TODO: other params
        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/sd-turbo", subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/sd-turbo", subfolder="text_encoder"
        ).cuda()
        self.sched = make_1step_sched()

        ## Load the VAE
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.requires_grad_(False)
        vae.encoder.forward = my_vae_encoder_fwd.__get__(
            vae.encoder, vae.encoder.__class__
        )
        vae.decoder.forward = my_vae_decoder_fwd.__get__(
            vae.decoder, vae.decoder.__class__
        )
        vae.requires_grad_(True)

        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(
            512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(
            256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(
            128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        vae.decoder.ignore_skip = False

        # Load the UNet
        unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/sd-turbo", subfolder="unet"
        )
        unet.requires_grad_(False)

        # Model initialization
        print("Initializing model with random weights")
        torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
        target_modules_vae = [
            "conv1",
            "conv2",
            "conv_in",
            "conv_shortcut",
            "conv",
            "conv_out",
            "skip_conv_1",
            "skip_conv_2",
            "skip_conv_3",
            "skip_conv_4",
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ]
        vae_lora_config = LoraConfig(
            r=lora_rank_vae,
            init_lora_weights="gaussian",
            target_modules=target_modules_vae,
        )
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        target_modules_unet = [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "conv",
            "conv1",
            "conv2",
            "conv_shortcut",
            "conv_out",
            "proj_in",
            "proj_out",
            "ff.net.2",
            "ff.net.0.proj",
            "conv_in",  # additional
            "linear_1",
            "linear_2",
            "time_emb_proj",
        ]
        unet_lora_config = LoraConfig(
            r=lora_rank_unet,
            init_lora_weights="gaussian",
            target_modules=target_modules_unet,
        )
        unet.add_adapter(unet_lora_config)
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device="cuda").long()  ##TODO: beware
        self.text_encoder.requires_grad_(False)

        ## force train mode
        ##TODO: test mode
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

        unet.enable_xformers_memory_efficient_attention()
        unet.enable_gradient_checkpointing()

    def forward(self, x, prompt):
        caption_tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.cuda()
        caption_enc = self.text_encoder(caption_tokens)[0]

        # match batch size
        # match batch size
        batch_size = caption_enc.shape[0]
        repeated_encs = [
            caption_enc[i].repeat(int(x.shape[0] / batch_size), 1, 1)
            for i in range(caption_enc.shape[0])
        ]

        # Concatenate the repeated encodings along the batch dimension
        captions_enc = torch.cat(repeated_encs, dim=0)

        # deterministic forward
        encoded_control = (
            self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
        )
        model_pred = self.unet(
            encoded_control,
            self.timesteps,
            encoder_hidden_states=captions_enc,
        ).sample
        x_denoised = self.sched.step(
            model_pred, self.timesteps, encoded_control, return_dict=True
        ).prev_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        x = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(
            -1, 1
        )
        return x

    def compute_feats(self, input, extract_layer_ids=[]):

        # deterministic forward
        encoded_control = (
            self.vae.encode(input).latent_dist.sample() * self.vae.config.scaling_factor
        )
        feats = self.vae.encoder.current_down_blocks
        return feats

    def get_feats(self, input, extract_layer_ids=[]):
        feats = self.compute_feats(input, extract_layer_ids)
        return feats

    def load_lora_config(self, lora_config_path):
        sd = torch.load(lora_config_path, map_location="cpu")

        unet_lora_config = LoraConfig(
            r=sd["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=sd["unet_lora_target_modules"],
        )
        vae_lora_config = LoraConfig(
            r=sd["rank_vae"],
            init_lora_weights="gaussian",
            target_modules=sd["vae_lora_target_modules"],
        )

        _sd_vae = self.vae.state_dict()
        for k in sd["state_dict_vae"]:
            _sd_vae[k] = sd["state_dict_vae"][k]
        self.vae.load_state_dict(_sd_vae)
        _sd_unet = self.unet.state_dict()
        for k in sd["state_dict_unet"]:
            _sd_unet[k] = sd["state_dict_unet"][k]
        self.unet.load_state_dict(_sd_unet)

        return

    def save_lora_config(self, save_lora_path):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {
            k: v
            for k, v in self.unet.state_dict().items()
            if "lora" in k or "conv_in" in k
        }
        sd["state_dict_vae"] = {
            k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k
        }
        torch.save(sd, save_lora_path)
        return
