from contextlib import ExitStack

import torch

from util.network_group import NetworkGroup

from . import gan_networks
from .b2b_model import B2BModel


class B2BCAFMModel(B2BModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = B2BModel.modify_commandline_options(parser, is_train=is_train)
        if is_train:
            parser.add_argument(
                "--alg_b2b_cafm_D_steps",
                type=int,
                default=16,
                help="Number of CAFM discriminator update slots per generator update.",
            )
            parser.add_argument(
                "--alg_b2b_cafm_D_warmup",
                type=int,
                default=5000,
                help="Number of CAFM logical optimizer steps that train only the discriminator.",
            )
            parser.add_argument(
                "--alg_b2b_cafm_cp_scale",
                type=float,
                default=0.001,
                help="Logit-centering regularization weight for CAFM discriminator.",
            )
            parser.add_argument(
                "--alg_b2b_cafm_ot_scale",
                type=float,
                default=0.0,
                help="Velocity norm regularization weight for CAFM generator.",
            )
            parser.add_argument(
                "--alg_b2b_cafm_lambda_fm",
                type=float,
                default=0.0,
                help="Optional B2B velocity matching loss weight during CAFM training.",
            )
            parser.add_argument(
                "--alg_b2b_cafm_grad_clip_G",
                type=float,
                default=0.0,
                help="Optional generator gradient clipping value for CAFM training. Use <=0 to disable.",
            )
            parser.add_argument(
                "--alg_b2b_cafm_grad_clip_D",
                type=float,
                default=0.0,
                help="Optional discriminator gradient clipping value for CAFM training. Use <=0 to disable.",
            )
        return parser

    @staticmethod
    def after_parse(opt):
        opt = B2BModel.after_parse(opt)

        if isinstance(opt.D_netDs, str):
            opt.D_netDs = [opt.D_netDs]
        if opt.D_netDs == ["projected_d", "basic"]:
            opt.D_netDs = ["cafm_jit"]
        if opt.D_netDs != ["cafm_jit"]:
            raise ValueError("b2b_cafm requires --D_netDs cafm_jit")

        if opt.G_netG not in ["vit", "vit_vid"]:
            raise ValueError("b2b_cafm requires --G_netG vit or vit_vid")
        if getattr(opt, "alg_b2b_cafm_D_steps", 16) <= 0:
            raise ValueError("--alg_b2b_cafm_D_steps must be > 0")
        if getattr(opt, "alg_b2b_cafm_D_warmup", 5000) < 0:
            raise ValueError("--alg_b2b_cafm_D_warmup must be >= 0")
        if getattr(opt, "alg_b2b_cafm_cp_scale", 0.001) < 0:
            raise ValueError("--alg_b2b_cafm_cp_scale must be >= 0")
        if getattr(opt, "alg_b2b_cafm_ot_scale", 0.0) < 0:
            raise ValueError("--alg_b2b_cafm_ot_scale must be >= 0")
        if getattr(opt, "alg_b2b_cafm_lambda_fm", 0.0) < 0:
            raise ValueError("--alg_b2b_cafm_lambda_fm must be >= 0")
        return opt

    def __init__(self, opt, rank):
        super().__init__(opt, rank)

        self._cafm_loaded_discriminator = False

        if self.isTrain:
            self.netDs = gan_networks.define_D(**vars(opt))
            self.discriminators_names = ["D_B_" + name for name in self.netDs.keys()]
            if self.discriminators_names != ["D_B_cafm_jit"]:
                raise RuntimeError(
                    "b2b_cafm expects exactly one cafm_jit discriminator"
                )

            for D_name, netD in self.netDs.items():
                setattr(self, "netD_B_" + D_name, netD.to(self.device))

            self.model_names += self.discriminators_names
            D_parameters = self.netD_B_cafm_jit.parameters()
            self.optimizer_D = opt.optim(
                opt,
                D_parameters,
                lr=opt.train_D_lr,
                betas=(opt.train_beta1, opt.train_beta2),
                weight_decay=opt.train_optim_weight_decay,
                eps=opt.train_optim_eps,
            )
            self.optimizers.append(self.optimizer_D)

            self.group_D = NetworkGroup(
                networks_to_optimize=self.discriminators_names,
                forward_functions=None,
                backward_functions=["compute_cafm_D_loss"],
                loss_names_list=["loss_names_D"],
                optimizer=["optimizer_D"],
                loss_backward=["loss_D_tot"],
            )
            self.networks_groups.append(self.group_D)

            self.loss_names_G = ["G_tot", "G_cafm", "G_fm", "G_ot"]
            self.loss_names_D = [
                "D_tot",
                "D_cafm",
                "D_cp",
                "D_jvp_real",
                "D_jvp_fake",
            ]
            self.loss_names = self.loss_names_G + self.loss_names_D
            self.iter_calculator_init()
            self._zero_cafm_losses()

    def setup(self, opt):
        super().setup(opt)
        if self.isTrain and not self._cafm_loaded_discriminator:
            self._init_cafm_discriminator_from_generator()

    def load_networks(self, epoch):
        try:
            super().load_networks(epoch)
            self._cafm_loaded_discriminator = self.isTrain
        except FileNotFoundError as exc:
            if "net_D_B_cafm_jit" not in str(exc):
                raise
            self._cafm_loaded_discriminator = False
            print(
                "CAFM discriminator checkpoint is missing; "
                "it will be initialized from the loaded B2B generator backbone."
            )

    def _raw_cafm_discriminator(self):
        netD = self.netD_B_cafm_jit
        if hasattr(netD, "module"):
            netD = netD.module
        return netD.dis

    def _init_cafm_discriminator_from_generator(self):
        if not self.isTrain:
            return
        raw_D = self._raw_cafm_discriminator()
        raw_G = self.netG_A
        if hasattr(raw_G, "module"):
            raw_G = raw_G.module
        raw_G = raw_G.b2b_model

        d_state = raw_D.state_dict()
        filtered = {
            key: value
            for key, value in raw_G.state_dict().items()
            if key in d_state and d_state[key].shape == value.shape
        }
        raw_D.load_state_dict(filtered, strict=False)
        print(
            f"Initialized CAFM discriminator from B2B generator: "
            f"{len(filtered)}/{len(d_state)} tensors matched."
        )

    def _zero_cafm_losses(self):
        value = torch.zeros(size=(), device=self.device)
        for name in self.loss_names_G + self.loss_names_D:
            setattr(self, "loss_" + name, value)
            avg_name = "loss_" + name + "_avg"
            if hasattr(self, avg_name):
                setattr(self, avg_name, value)

    def _logical_cafm_step(self):
        iter_size = max(1, self.opt.train_iter_size)
        return (self.niter - 1) // iter_size + 1

    def _is_cafm_discriminator_step(self):
        logical_step = self._logical_cafm_step()
        warmup = self.opt.alg_b2b_cafm_D_warmup
        if logical_step <= warmup:
            return True
        cycle_step = (logical_step - warmup - 1) % (self.opt.alg_b2b_cafm_D_steps + 1)
        return cycle_step < self.opt.alg_b2b_cafm_D_steps

    def _cafm_labels(self):
        if self.num_classes > 0:
            return self.label_cls
        return torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

    def _compute_cafm_state(self, enable_generator_grad):
        context = torch.enable_grad() if enable_generator_grad else torch.no_grad()
        with context:
            x_pred, z, v, t, x_target = self.netG_A.b2b_forward(
                self.gt_image,
                self.mask,
                self.cond_image,
                self._cafm_labels(),
                getattr(self, "use_gt", None),
                getattr(self, "ref_idx", None),
            )
            if self.mask is not None:
                x_pred = x_pred * self.mask + (1.0 - self.mask) * x_target
            v_pred = (x_pred - z) / (1.0 - t).clamp_min(self.netG_A.t_eps)
        return z, v, v_pred, t, self._cafm_labels()

    def _flatten_cafm_batch(self, z, v, v_pred, t, labels):
        if z.ndim == 4:
            if labels.ndim != 1 or labels.shape[0] != z.shape[0]:
                raise RuntimeError(
                    f"Expected image labels shape [{z.shape[0]}], got {tuple(labels.shape)}"
                )
            return z, v, v_pred, t.reshape(-1), labels.reshape(-1)

        if z.ndim != 5:
            raise RuntimeError(f"Expected image or video tensor, got {tuple(z.shape)}")

        B, F_, C, H, W = z.shape
        z = z.reshape(B * F_, C, H, W)
        v = v.reshape(B * F_, C, H, W)
        v_pred = v_pred.reshape(B * F_, C, H, W)
        t = t.reshape(B * F_)

        if labels.ndim == 1 and labels.shape[0] == B:
            labels = labels[:, None].expand(B, F_).reshape(B * F_)
        elif labels.ndim == 2 and labels.shape == (B, F_):
            labels = labels.reshape(B * F_)
        else:
            raise RuntimeError(
                f"Expected video labels shape [{B}] or [{B}, {F_}], got {tuple(labels.shape)}"
            )
        return z, v, v_pred, t, labels

    def _compute_velocity_matching_loss(self, v_pred, v):
        if self.mask is not None:
            mask = torch.clamp(self.mask, min=0, max=1)
            if self.opt.alg_b2b_loss_masked_region_only:
                return self._masked_region_loss(v_pred, v, mask)
            return self.loss_fn(mask * v_pred, mask * v)
        return self.loss_fn(v_pred, v)

    def compute_cafm_D_loss(self):
        z, v, v_pred, t, labels = self._compute_cafm_state(enable_generator_grad=False)
        z, v, v_pred, t, labels = self._flatten_cafm_batch(z, v, v_pred, t, labels)
        tangent_x = torch.stack([v.detach(), v_pred.detach()])
        tangent_t = torch.ones((2, t.shape[0]), dtype=t.dtype, device=t.device)

        with torch.cuda.amp.autocast(enabled=self.with_amp):
            out, out_jvp = self.netD_B_cafm_jit(
                x=z.detach(),
                y=labels,
                t=t.detach(),
                dx=tangent_x,
                dt=tangent_t,
            )
            logits_real, logits_fake = out_jvp.chunk(2, dim=0)
            self.loss_D_jvp_real = logits_real.sub(1.0).square().mean()
            self.loss_D_jvp_fake = logits_fake.add(1.0).square().mean()
            self.loss_D_cafm = self.loss_D_jvp_real + self.loss_D_jvp_fake
            self.loss_D_cp = out.square().mean() * self.opt.alg_b2b_cafm_cp_scale
            self.loss_D_tot = self.loss_D_cafm + self.loss_D_cp

    def compute_cafm_G_loss(self):
        z, v, v_pred, t, labels = self._compute_cafm_state(enable_generator_grad=True)
        self.loss_G_fm = torch.zeros(size=(), device=self.device)
        if self.opt.alg_b2b_cafm_lambda_fm > 0:
            fm_loss = self._compute_velocity_matching_loss(v_pred, v.detach())
            if isinstance(fm_loss, dict):
                fm_loss = sum(fm_loss.values())
            self.loss_G_fm = fm_loss * self.opt.alg_b2b_cafm_lambda_fm

        z, v, v_pred, t, labels = self._flatten_cafm_batch(z, v, v_pred, t, labels)
        tangent_t = torch.ones_like(t)

        with torch.cuda.amp.autocast(enabled=self.with_amp):
            _, logits_fake = self.netD_B_cafm_jit(
                x=z.detach(),
                y=labels,
                t=t.detach(),
                dx=v_pred,
                dt=tangent_t,
            )
            self.loss_G_cafm = logits_fake.sub(1.0).square().mean()
            self.loss_G_ot = v_pred.square().mean() * self.opt.alg_b2b_cafm_ot_scale
            self.loss_G_tot = self.loss_G_cafm + self.loss_G_ot + self.loss_G_fm

    def _backward_loss(self, loss):
        loss = loss / self.opt.train_iter_size
        if self.use_cuda:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _clip_gradients(self, network_name, clip_value):
        if clip_value <= 0:
            return
        network = getattr(self, "net" + network_name)
        torch.nn.utils.clip_grad_norm_(network.parameters(), clip_value)

    def optimize_parameters(self):
        self.niter += 1

        with ExitStack() as stack:
            if len(self.opt.gpu_ids) > 1 and self.niter % self.opt.train_iter_size != 0:
                for network in self.model_names:
                    stack.enter_context(getattr(self, "net" + network).no_sync())

            if self._is_cafm_discriminator_step():
                self.set_requires_grad(self.netG_A, False)
                self.set_requires_grad(self.netD_B_cafm_jit, True)
                self.compute_cafm_D_loss()
                self._backward_loss(self.loss_D_tot)
                if self.niter % self.opt.train_iter_size == 0:
                    self._clip_gradients(
                        "D_B_cafm_jit", self.opt.alg_b2b_cafm_grad_clip_D
                    )
                self.compute_step(["optimizer_D"], self.loss_names_D)
            else:
                self.set_requires_grad(self.netG_A, True)
                self.set_requires_grad(self.netD_B_cafm_jit, False)
                self.compute_cafm_G_loss()
                self._backward_loss(self.loss_G_tot)
                if self.niter % self.opt.train_iter_size == 0:
                    self._clip_gradients("G_A", self.opt.alg_b2b_cafm_grad_clip_G)
                self.compute_step(["optimizer_G"], self.loss_names_G)
                if self.opt.train_G_ema and self.niter % self.opt.train_iter_size == 0:
                    self.ema_step("G_A")

            for cur_object in self.objects_to_update:
                cur_object.update(self.niter)
