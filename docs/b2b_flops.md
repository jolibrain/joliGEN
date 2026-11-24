# B2B forward complexity with THOP

This optional tool profiles representative denoiser forwards in Baseline,
SmartBrush BBox, and SmartBrush Hybrid. Model sets and paths are configurable;
no dataset or generation sequence is required. It uses the checkpoint loader
and `predict_clean` path from this checkout, including active prediction heads
and enabled refiners. Existing training metrics are unchanged.

## Installation and commands

Use your project Python environment. Install the pinned profiler into an isolated,
ignored directory without replacing the existing `thop` distribution or Torch:

```bash
python -m pip install --no-deps --target results/thop_deps ultralytics-thop==2.2.0
```

For a single model set, pass the two checkpoint files directly. The profiler reads
each checkpoint's adjacent `train_config.json`. The class label defaults to 0.

```bash
python scripts/profile_b2b_flops.py --baseline /path/to/baseline/model.pth \
  --smartbrush /path/to/smartbrush/model.pth --device cuda:0 --output-dir results/flops_pair
```

When profiling finishes, the command prints the path to `summary.csv`.

To profile multiple model sets, copy `scripts/profile_b2b_flops.example.json`
to `results/flops.local.json` and fill in each pair of checkpoint paths,
training configuration paths, and class label. For EMA weights, specify the
actual `_ema.pth` file.

```bash
python scripts/profile_b2b_flops.py --config results/flops.local.json --check-only
python scripts/profile_b2b_flops.py --config results/flops.local.json \
  --model-set lv --device cuda:0 --output-dir results/flops_lv
# Omit --model-set to profile every configured set:
python scripts/profile_b2b_flops.py --config results/flops.local.json \
  --device cuda:0 --output-dir results/flops_all
```

Check-only checks paths, B2B configuration and the isolated profiler version;
it never loads model weights or executes forwards. CPU execution is supported.
Output directories must be new. A failed run raises and does not publish a
partial summary; rerun with a fresh destination. Real profiling can require
substantial memory even though no generation sequence is run.

## Interpretation

`summary.csv` has three rows per model set. Baseline and BBox are each one
forward; Hybrid reports coarse and refinement costs separately and their sum
(one forward per stage). Its parameters are shared and counted once. This is
not a complete generation cost: sampler steps, guidance calls, frame reuse,
preprocessing, mask projection, and rendering are outside scope. Equal BBox
and Hybrid stage costs are expected when their shapes and network paths match.

Synthetic FP32 inputs use batch 1, seed 42, and timestep 0.5. Shapes come from
the training configuration, including frame count, channels and conditioning.
Network resolution uses `data.crop_size`, matching the model constructor and
denoiser export. `data.online_creation.crop_size_A` and `data.load_size` describe
source preprocessing and do not override network resolution. Coarse
SmartBrush inputs use bbox precision (reference frame exact); refinement uses
exact precision. Severity is zero. Mask-size features are synthetic, with six
features per frame. Object-reference conditioned models are currently rejected
explicitly; this does not affect the supplied eight checkpoints.

`details.json` records input and conditioning shapes, scalar conditioning,
checkpoint/config SHA256 hashes, versions, counting policy and diagnostics.
Weights and datasets are never written by the tool.

## Counting policy

The pinned upstream counter is [ultralytics-thop 2.2.0](https://github.com/ultralytics/thop/tree/v2.2.0).
Use full input shapes, without spatial extrapolation. Retain THOP's native
module rules and functional matmul/bmm/attention rules. Add a separate
`baddbmm` product counter because that operation is absent from the pinned
functional rules. The supplement counts the matrix product, excluding the
additive input and alpha/beta arithmetic. No attention-module hook is added,
so projections and the attention-value product are not counted twice.

Estimated GFLOPs = 2 × (native THOP count + supplemental baddbmm MACs) / 1e9.
Native THOP includes some normalization and pooling estimates, so this is a
THOP convention estimate, not an exact count of hardware instructions or a
pure matrix-MAC total. Functional scalar arithmetic, normalization, activation
and data movement are excluded. Raw native and supplemental counts remain
separate. This differs from fvcore conventions counting one MAC as one FLOP.

Unknown leaf modules and observed unsupported functional products yield a null
estimated GFLOPs value, with diagnostics. Known scalar/data-only leaves (custom
RMSNorm and rotary embeddings, SiLU/GELU and dropout) are declared exclusions.
`complete` means no such unresolved diagnostics for this executed forward,
not exhaustive coverage of arbitrary models, branches, or scalar arithmetic.
In the temporal transformer, THOP counts Q/K/V and output projections, the
attention-value product, feed-forward projections and LayerNorm. The supplemental
counter counts the query-key product. Functional softmax, residual additions
and GEGLU elementwise arithmetic are outside this estimate. The CLI reports
unresolved diagnostics and omits THOP's per-class registration messages.

## Tests

```bash
python -m pytest -q tests/test_b2b_flops.py
```

Tests use small synthetic CPU models and analytical counts, never real checkpoints.
Run and inspect LV locally before comparing all model sets or publishing results.
