from unet_generator_attn import UNet


class ControlledUNet(UNetModel):
    def forward(self, input, gammas, control=None, only_mid_control=False):

        with torch.no_grad():
            h, _, emb = self.compute_feats(input, gammas=gammas)

        if control is not None:
            h += control.pop()

        hs = control

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb)

        h = h.type(input.dtype)
        return self.out(h)


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key="class", ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0.0 and not disable_dropout:
            mask = 1.0 - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1 - mask) * torch.ones_like(c) * (self.n_classes - 1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = (
            self.n_classes - 1
        )  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


class ControlNet(UnetModel):
    def __init__(self, *kwargs):
        super().__init__(*kwargs)

        del self.output_blocks
        del self.out

        hint_channels = self.in_channels

        self.input_hint_block = nn.Sequential(
            nn.conv2d(hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.conv2d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.conv2d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(nn.conv2d(256, model_channels, 3, padding=1)),
        )  # emebdding is not used here

        model_channels = self.inner_channel

        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        for ch in self.input_block_chans:
            self.zero_convs.append(self.make_zero_conv(ch))

        ch_middle = self.input_block_chans[-1]

        self.middle_block_out = self.make_zero_conv(ch_middle)

    def compute_feats(self, input, gammas, hint=None):
        guided_hint = self.input_hint_block(hint)

        input += guided_hint

        if gammas is None:
            b = input.shape[0]
            gammas = torch.ones((b,)).to(input.device)

        gammas = gammas.view(-1)

        emb = self.cond_embed(gamma_embedding(gammas, self.inner_channel))

        hs = []

        h = input.type(torch.float32)
        for module, zero_conv in zip(self.input_blocks, zero_convs):
            h = module(h, emb)
            hs.append(zero_conv(h))
        h = self.middle_block(h, emb)
        hs.append(self.middle_block_out(h, emb))

        outs, feats = h, hs
        return feats
