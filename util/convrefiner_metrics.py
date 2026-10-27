"""Paired quality diagnostics for a frozen backbone and its ConvRefiner."""

import random
import time

import numpy as np
import torch


def radial_power(image, bins=64):
    """Luma, mean removal, Hann window and radial power in cycles/pixel."""
    luma = (image.float() * image.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(
        0
    )
    height, width = luma.shape
    window = (
        torch.hann_window(height, periodic=False, device=luma.device)[:, None]
        * torch.hann_window(width, periodic=False, device=luma.device)[None]
    )
    power = torch.fft.fft2(
        (luma - luma.mean()) * window
    ).abs().square() / window.square().sum().clamp_min(1e-12)
    fy = torch.fft.fftfreq(height, device=luma.device)[:, None]
    fx = torch.fft.fftfreq(width, device=luma.device)[None]
    index = (
        ((fx.square() + fy.square()).sqrt() / (0.5**0.5) * bins)
        .long()
        .clamp_max(bins - 1)
    )
    total = torch.zeros(bins, device=luma.device).scatter_add_(
        0, index.flatten(), power.flatten()
    )
    count = torch.bincount(index.flatten(), minlength=bins)
    return total / count.clamp_min(1), count > 0


class RefinerMetrics:
    def __init__(self, lpips=None, dists=None):
        self.lpips, self.dists = lpips, dists
        self.mse_sum = self.pixel_count = 0.0
        self.temporal_sum = self.temporal_count = 0.0
        self.perceptual = {"LPIPS": [], "DISTS": []}
        self.powers = []

    @torch.no_grad()
    def add(self, prediction, target, mask):
        prediction = (prediction.float() + 1) / 2
        target = (target.float() + 1) / 2
        if prediction.ndim == 4:
            prediction, target, mask = (
                prediction[:, None],
                target[:, None],
                mask[:, None],
            )
        self.mse_sum += ((prediction - target).square() * mask).sum().item()
        self.pixel_count += mask.sum().item() * prediction.shape[2]
        if prediction.shape[1] > 1:
            error = (prediction[:, 1:] - prediction[:, :-1]) - (
                target[:, 1:] - target[:, :-1]
            )
            valid = torch.maximum(mask[:, 1:], mask[:, :-1])
            self.temporal_sum += (error.abs() * valid).sum().item()
            self.temporal_count += valid.sum().item() * prediction.shape[2]
        for pred, real, region in zip(
            prediction.flatten(0, 1), target.flatten(0, 1), mask.flatten(0, 1)
        ):
            coordinates = torch.nonzero(region[0] > 0, as_tuple=False)
            if not len(coordinates):
                continue
            y0, x0 = (coordinates.min(0).values - 8).clamp_min(0).tolist()
            y1, x1 = (coordinates.max(0).values + 9).tolist()
            # Matched rectangular crops, never a mask multiplied into the FFT.
            p, r = pred[:, y0:y1, x0:x1], real[:, y0:y1, x0:x1]
            if min(p.shape[-2:]) >= 16:
                pp, valid = radial_power(p)
                rp, _ = radial_power(r)
                self.powers.append((pp.cpu(), rp.cpu(), valid.cpu()))
            # Networks require adequate spatial support; use the full composited crop.
            for name, metric in (("LPIPS", self.lpips), ("DISTS", self.dists)):
                if metric is not None:
                    self.perceptual[name].append(
                        float(metric(pred[None].clamp(0, 1), real[None].clamp(0, 1)))
                    )

    def result(self):
        mse = self.mse_sum / max(self.pixel_count, 1)
        result = {
            "masked_PSNR": (
                float(-10 * np.log10(max(mse, 1e-12))) if self.pixel_count else None
            ),
            "temporal_difference_MAE": (
                self.temporal_sum / self.temporal_count if self.temporal_count else None
            ),
        }
        result.update(
            {
                name: float(np.mean(values)) if values else None
                for name, values in self.perceptual.items()
            }
        )
        if self.powers:
            pred, real, valid = [torch.stack(values) for values in zip(*self.powers)]
            count = valid.sum(0)
            pred = (pred * valid).sum(0) / count.clamp_min(1)
            real = (real * valid).sum(0) / count.clamp_min(1)
            frequency = (torch.arange(len(pred)) + 0.5) / len(pred) * (0.5**0.5)
            high = (frequency >= 0.5) & (frequency <= 0.7) & (count > 0)
            deficit = 10 * torch.log10(pred.clamp_min(1e-12) / real.clamp_min(1e-12))
            result.update(
                HF_deficit_dB=float(deficit[high].mean()),
                HF_absolute_deficit_dB=float(deficit[high].abs().mean()),
                radial_frequency=frequency.tolist(),
                radial_predicted_power=pred.tolist(),
                radial_real_power=real.tolist(),
            )
        return result


def acceptance(baseline, refined):
    checks = {}
    for metric in (
        "masked_PSNR",
        "LPIPS",
        "DISTS",
        "temporal_difference_MAE",
        "HF_absolute_deficit_dB",
    ):
        a, b = baseline.get(metric), refined.get(metric)
        if a is None or b is None:
            checks[metric] = None
        elif metric == "masked_PSNR":
            checks[metric] = b >= a - 0.5
        elif metric == "HF_absolute_deficit_dB":
            checks[metric] = b < a
        else:
            checks[metric] = b <= a * 1.05 + 1e-12
    return {
        "checks": checks,
        "numeric_pass": all(value is True for value in checks.values()),
        "visual_review_required": True,
    }


@torch.no_grad()
def evaluate_refiner(model, dataloader, max_batches=16, seed=42):
    """Identical data and noise for each paired comparison, without changing training RNG."""
    rng = model._refiner_rng_state()
    was_train = model.opt.isTrain
    net = getattr(model, "netG_A_ema", model.netG_A)
    net = model._unwrap_net(net)
    was_enabled, was_mode = net.refiner_enabled, net.training
    saved_gt_prob = model.opt.alg_b2b_use_gt_prob
    # inference() selects EMA with isTrain=False. Always provide the same clean reference.
    model.opt.isTrain = False
    model.opt.alg_b2b_use_gt_prob = 1.0
    metrics, timings = {}, {}
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        net.eval()
        for index, data in enumerate(dataloader):
            if index >= max_batches:
                break
            model.set_input(data)
            truth, mask = model.gt_image, model.mask
            for enabled, label in ((False, "baseline"), (True, "refined")):
                # inference() expands gt_image/projection_mask for its visual grids.
                model.gt_image = truth
                net.refiner_enabled = enabled
                torch.manual_seed(seed + index)
                if model.device.type == "cuda":
                    torch.cuda.synchronize(model.device)
                start = time.perf_counter()
                model.inference(truth.shape[0])
                if model.device.type == "cuda":
                    torch.cuda.synchronize(model.device)
                timings.setdefault(label, []).append(time.perf_counter() - start)
                for steps, prediction in model.outputs_per_step.items():
                    key = (steps, label)
                    if key not in metrics:
                        metrics[key] = RefinerMetrics(
                            getattr(model, "criterionLPIPS", None),
                            getattr(model, "criterionDISTS", None),
                        )
                    metrics[key].add(prediction, truth, mask)
        report = {}
        for steps in sorted({key[0] for key in metrics}):
            baseline, refined = (
                metrics[(steps, "baseline")].result(),
                metrics[(steps, "refined")].result(),
            )
            report[str(steps)] = {
                "baseline": baseline,
                "refined": refined,
                "acceptance": acceptance(baseline, refined),
            }
        report["mean_seconds_all_samplers_per_batch"] = {
            label: float(np.mean(values)) for label, values in timings.items()
        }
        report["protocol"] = (
            "paired fixed-seed windows with clean reference; temporal MAE is against GT differences, not optical-flow warping"
        )
        return report
    finally:
        net.refiner_enabled = was_enabled
        net.train(was_mode)
        model.opt.isTrain = was_train
        model.opt.alg_b2b_use_gt_prob = saved_gt_prob
        torch.set_rng_state(rng["torch"])
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        if rng["cuda"] is not None:
            torch.cuda.set_rng_state(rng["cuda"])
