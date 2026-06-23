VALID_B2B_GLOBAL_CONTEXT_MODES = ("none", "adaln", "tokens", "both")


def _normalize_mode(mode):
    mode = str(mode).lower()
    if mode not in VALID_B2B_GLOBAL_CONTEXT_MODES:
        raise ValueError(
            "alg_b2b_global_context_mode must be one of "
            f"{VALID_B2B_GLOBAL_CONTEXT_MODES}, got {mode!r}"
        )
    return mode


def b2b_global_context_mode_from_opt(opt):
    mode = getattr(opt, "alg_b2b_global_context_mode", None)
    if mode not in (None, ""):
        return _normalize_mode(mode)
    if getattr(opt, "alg_b2b_global_context_conditioning", False):
        return "adaln"
    return "none"


def b2b_global_context_mode_from_train_json(train_json):
    alg = train_json.get("alg", {})
    mode = alg.get("b2b_global_context_mode", None)
    if mode not in (None, ""):
        return _normalize_mode(mode)
    if bool(alg.get("b2b_global_context_conditioning", False)):
        return "adaln"
    return "none"


def b2b_global_context_enabled(mode):
    return _normalize_mode(mode) != "none"


def b2b_global_context_enabled_from_opt(opt):
    return b2b_global_context_enabled(b2b_global_context_mode_from_opt(opt))


def b2b_global_context_enabled_from_train_json(train_json):
    return b2b_global_context_enabled(
        b2b_global_context_mode_from_train_json(train_json)
    )


def b2b_global_context_adaln_enabled(mode):
    return _normalize_mode(mode) in ("adaln", "both")


def b2b_global_context_tokens_enabled(mode):
    return _normalize_mode(mode) in ("tokens", "both")
