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
    def __init__(self, in_channels, out_channels):
        ##TODO
        super().__init__()

        # TODO: other params
        lora_rank_unet = 8
        lora_rank_vae = 4

        self.tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/sd-turbo", subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/sd-turbo", subfolder="text_encoder"
        ).cuda()
        self.sched = make_1step_sched()

        ## Load the VAE
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(
            vae.encoder, vae.encoder.__class__
        )
        vae.decoder.forward = my_vae_decoder_fwd.__get__(
            vae.decoder, vae.decoder.__class__
        )

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
        ]
        unet_lora_config = LoraConfig(
            r=lora_rank_unet,
            init_lora_weights="gaussian",
            target_modules=target_modules_unet,
        )
        unet.add_adapter(unet_lora_config)
        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
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

    def forward(self, x):
        # print("x", x.shape)

        ##TODO: self.prompt
        prompt = "driving in the night"
        caption_tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.cuda()
        caption_enc = self.text_encoder(caption_tokens)[0]
        # for i in range(x.shape[0]):
        # cat caption to match batch size
        #    caption_enc = torch.cat([caption_enc, caption_enc[0].unsqueeze(0)], dim=0)
        # print("caption_enc", caption_enc.shape)

        # deterministic forward
        encoded_control = (
            self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
        )
        # print("encoded_control", encoded_control.shape)
        model_pred = self.unet(
            encoded_control,
            self.timesteps,
            encoder_hidden_states=caption_enc,
        ).sample
        x_denoised = self.sched.step(
            model_pred, self.timesteps, encoded_control, return_dict=True
        ).prev_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        x = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(
            -1, 1
        )
        return x

    ##TODO: for CUT
    def compute_feats(self, input, extract_layer_ids=[]):
        ##TODO
        prompt = "driving in the night"
        caption_tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.cuda()
        caption_enc = self.text_encoder(caption_tokens)[0]

        # deterministic forward
        encoded_control = (
            self.vae.encode(input).latent_dist.sample() * self.vae.config.scaling_factor
        )
        feats = self.vae.encoder.current_down_blocks
        # print('feats vae=', feats)
        return feats

    def get_feats(self, input, extract_layer_ids=[]):
        feats = self.compute_feats(input, extract_layer_ids)
        return feats

    ##TODO: load lora config and weights
    def load_lora_config(self, lora_config_path):
        ##TODO
        return
