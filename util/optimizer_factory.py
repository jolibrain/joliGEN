import warnings

import torch

from util.lion_pytorch import Lion

_MUON_EXCLUDED_NAME_FRAGMENTS = (
    "embedding.weight",
    "embedding_table.weight",
    ".embedding.",
    ".embeddings.",
    "embed_tokens.weight",
    "token_embedding.weight",
    "position_embeddings",
    "pos_embed",
    "cls_token",
    "mask_token",
    "class_embedding",
    ".wte.",
    ".wpe.",
)


def build_named_parameters(*sources):
    named_params = []
    seen_params = set()

    for prefix, source in sources:
        if source is None:
            continue

        if hasattr(source, "named_parameters"):
            iterable = source.named_parameters()
        else:
            iterable = source

        for idx, item in enumerate(iterable):
            if isinstance(item, dict):
                raise TypeError(
                    "Optimizer parameter groups are not supported by build_named_parameters."
                )

            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                name, param = item
            else:
                name, param = f"param_{idx}", item

            if param is None or not getattr(param, "requires_grad", False):
                continue

            param_id = id(param)
            if param_id in seen_params:
                continue
            seen_params.add(param_id)

            if prefix:
                full_name = f"{prefix}.{name}" if name else prefix
            else:
                full_name = name
            named_params.append((full_name, param))

    return named_params


def _is_muon_eligible(name, param):
    if not param.requires_grad or param.ndim != 2:
        return False

    lowered = name.lower()
    return not any(fragment in lowered for fragment in _MUON_EXCLUDED_NAME_FRAGMENTS)


def _split_muon_parameters(named_params):
    muon_params = []
    aux_params = []

    for name, param in named_params:
        if _is_muon_eligible(name, param):
            muon_params.append((name, param))
        else:
            aux_params.append(param)

    return muon_params, aux_params


def _build_single_optimizer(opt, params, lr, betas, weight_decay, eps):
    if opt.train_optim == "adam":
        return torch.optim.Adam(params, lr, betas, weight_decay=weight_decay, eps=eps)
    if opt.train_optim == "radam":
        return torch.optim.RAdam(params, lr, betas, weight_decay=weight_decay, eps=eps)
    if opt.train_optim == "adamw":
        return torch.optim.AdamW(params, lr, betas, weight_decay=weight_decay, eps=eps)
    if opt.train_optim == "lion":
        return Lion(params, lr, betas, weight_decay)
    if opt.train_optim == "adam8bit":
        import bitsandbytes as bnb

        return bnb.optim.Adam8bit(params, lr, betas, weight_decay=weight_decay, eps=eps)

    raise ValueError(f"Unsupported optimizer '{opt.train_optim}'")


def build_optimizer_bundle(
    opt,
    params,
    lr,
    betas,
    weight_decay,
    eps,
    optimizer_name=None,
):
    print("Using", opt.train_optim, "as optimizer")

    named_params = build_named_parameters(("", params))
    if len(named_params) == 0:
        raise ValueError("Cannot build an optimizer with no trainable parameters.")

    base_name = optimizer_name if optimizer_name is not None else "optimizer"

    if opt.train_optim != "muon":
        raw_params = [param for _, param in named_params]
        return [
            (
                base_name,
                _build_single_optimizer(opt, raw_params, lr, betas, weight_decay, eps),
            )
        ]

    if not hasattr(torch.optim, "Muon"):
        raise ValueError("torch.optim.Muon is not available in this PyTorch build.")

    muon_named_params, aux_params = _split_muon_parameters(named_params)
    optimizer_bundle = []

    if muon_named_params:
        optimizer_bundle.append(
            (
                base_name,
                torch.optim.Muon(
                    muon_named_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=betas[0],
                    nesterov=opt.train_muon_nesterov,
                    eps=eps,
                    ns_steps=opt.train_muon_ns_steps,
                    adjust_lr_fn=opt.train_muon_adjust_lr_fn,
                ),
            )
        )

    if aux_params:
        aux_name = base_name if not muon_named_params else base_name + "_aux"
        optimizer_bundle.append(
            (
                aux_name,
                torch.optim.AdamW(
                    aux_params,
                    lr=lr,
                    betas=betas,
                    weight_decay=weight_decay,
                    eps=eps,
                ),
            )
        )

    if not muon_named_params:
        warnings.warn(
            f"No Muon-eligible parameters found for {base_name}; using AdamW for all parameters instead."
        )

    return optimizer_bundle
