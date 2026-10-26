# B2B ConvRefiner finetuning

ConvRefiner learns a residual on the B2B clean RGB prediction at **each denoiser evaluation**, before velocity conversion and CFG. The pretrained JiT/JiTViD backbone, temporal modules, conditioning encoders, and SmartBrush mask head stay frozen. Known pixels and reference frames remain protected by the existing sampler projection.

The spatial network follows [MFlow Fig. 7 and Appendix A.7](https://arxiv.org/html/2610.23854v1): four timestep-conditioned spatial residual blocks and a zero-initialized output convolution. The 64-channel width and temporal branches are JoliGEN defaults. Temporal mode adds a zero-initialized `(3,1,1)` residual convolution after each block. Frames are processed jointly within the existing window; the model still returns one correction per frame and requires no extra future frames. This is not motion alignment, and temporal quality must be checked experimentally.

## Prepare the lvring run

```bash
python scripts/prepare_b2b_convrefiner.py \
  --output_dir checkpoints/convrefiner_lvring_recipe
```

The default source is `beniz_lvring_ft_32bit_rotrebox_handcond_2_smartbrush_4_new_aug_ft3`, using the immutable epoch-320 EMA checkpoint. Override `--source_config`, `--epoch`, or `--raw_source` for another source. The tool refuses to overwrite its output directory.

It produces `pilot.json`, `smoke.json`, `datasets.json`, `split.json`, and recording-specific path manifests. It reserves 10% of recording groups with seed 42; adjacent frames cannot cross the split. Each recording becomes a child of the existing `multi_dataset` wrapper, retaining the source video transforms and object class labels. Images and masks are referenced by absolute path, not copied. Source data, checkpoints, and configuration are not modified. The backbone may already have seen held-out recordings; this is a holdout from **refiner training** only.

The prepared local recipe lives in the ignored `checkpoints/` directory. Regenerate it on another machine instead of copying manifests with machine-specific paths.

```bash
# Choose an available GPU. The example overrides the source run's GPU selection.
python train.py --config_json checkpoints/convrefiner_lvring_recipe/smoke.json --gpu_ids 1
python train.py --config_json checkpoints/convrefiner_lvring_recipe/pilot.json --gpu_ids 1
```

The smoke recipe runs one short epoch with at most two frames per recording. The pilot uses 30 epochs, FP32, batch size 6, accumulation 32, AdamW at `1e-4`, betas `(0.9,0.95)`, and no weight decay. It retains SmartBrush, context tokens, masked pseudo-Huber reconstruction, LPIPS/DISTS with weight `0.1`, and the 2/5-step samplers. Runtime is not inferred from the paper's percentage: each microbatch performs two frozen-backbone forwards and refiner backward passes.

## Options and objective

| Option | Default | Meaning |
| --- | --- | --- |
| `alg_b2b_convrefiner` | `none` | `spatial` or `temporal` adds the refiner; legacy configurations remain unchanged. |
| `alg_b2b_convrefiner_width` | `64` | Feature channels, a positive multiple of 32. |
| `alg_b2b_convrefiner_finetune` | `false` | Train only the refiner and discriminator. Requires a source checkpoint or resume. |
| `alg_b2b_convrefiner_source_ema` | `false` | Explicitly load the source's sibling EMA checkpoint for fresh finetuning. |
| `alg_b2b_convrefiner_gan_weight` | `0.02` | Generator hinge-adversarial weight. |
| `alg_b2b_convrefiner_gan_schedule` | `near_clean` | Original near-clean training, or `sampler_aligned` adversarial batches. Inference is unchanged. |
| `alg_b2b_convrefiner_gan_near_clean_prob` | `0.2` | Near-clean probability in sampler-aligned mode. |
| `alg_b2b_convrefiner_gan_timestep_radius` | `0.05` | Uniform jitter radius around each configured sampler's final evaluation time. |
| `alg_b2b_convrefiner_resume_from` | empty | Full-state resume from a different run directory; requires `train_continue` and empty `train_continue_from`. |
| `alg_b2b_convrefiner_eval_batches` | `0` | Paired validation batches per test recording at every epoch; the pilot uses 16. |

Training makes an ordinary all-timestep draw and a separate `t ~ U(0.9,0.98)` draw. The refiner loss is `L_all + 0.25 L_near + 0.02 L_GAN`. Reconstruction uses the configured velocity loss. Perceptual losses apply only to generated, nonempty frames at `t >= 0.3`. The discriminator is a three-layer, width-64, framewise PatchGAN conditioned by an extra timestep channel, with instance normalization and hinge loss. It sees real and refined RGB composited with identical known background. Patch losses are weighted by editable-mask coverage; reference-only or empty-mask frames contribute zero.

Only RGB is refined. Mask logits at an identical backbone input remain unchanged; generated masks can change over a rollout because refined RGB changes subsequent denoiser inputs. LoRA training, min-SNR, and `torch.compile` are rejected in refiner finetune mode. The first implementation supports RGB JiT/JiTViD with training crops of at least 32 pixels.

### Sampler-aligned adversarial training

For the existing uniform Heun/final-Euler sampler, the final network evaluation is at `1 - 1/N` for `N` integration steps. The original GAN interval `[0.9,0.98]` is never visited by the generated frames at 2 or 5 steps. The opt-in `sampler_aligned` schedule draws one band per adversarial microbatch, then independently samples each example within that band:

- 40% of batches: `U(0.45,0.55)` for the 2-step sampler.
- 40% of batches: `U(0.75,0.85)` for the 5-step sampler.
- 20% of batches: original `U(0.9,0.98)` supervision.

The first two probabilities split the non-near-clean mass equally across **unique** configured step counts. Bands are clipped to `[0,1]`; step counts must be at least two. Ground-truth reference frames remain at `t=1` and are excluded by their masks. The all-t batch, optimizer settings, backbone freeze, inference schedules, and number of forward/backward passes are unchanged. Both GAN and the downweighted reconstruction/perceptual loss use the selected adversarial band: `L_all + 0.25 L_adv + gan_weight L_GAN`.

In this mode, `G_refiner_adv_avg` replaces the misleading name `G_refiner_near_avg`. It averages the **unweighted** reconstruction/perceptual loss across the mixture, so it is not directly comparable to the old near-only curve. Schedule metadata is saved with training state, and band selection uses the checkpointed Python RNG. This is an experimental adaptation for short-step video inpainting, not the paper's original recipe. These inputs are still forward-noised training examples, not sampler rollouts.

Prepare a separate branch without copying or overwriting checkpoints:

```bash
python scripts/prepare_b2b_convrefiner_branch.py \
  --source_config checkpoints/convrefiner_lvring_recipe/pilot.json \
  --epoch 5 \
  --output_config checkpoints/convrefiner_lvring_recipe/sampler_aligned_from_5.json
python train.py \
  --config_json checkpoints/convrefiner_lvring_recipe/sampler_aligned_from_5.json \
  --gpu_ids 1
```

The branch restores raw G, EMA, D, optimizers, schedulers, scaler, RNG, and the saved cursor from epoch 5, then writes to a new run name. It keeps the configured total epoch budget (30 for the pilot), rather than adding 30 epochs. Do not run concurrently on the original run's GPU unless sufficient resources are available. The preparation tool refuses an existing output config or destination run. It does not start training.

To resume **the branch's own** later checkpoints, clear its initial source directory:

```bash
python train.py \
  --config_json checkpoints/convrefiner_lvring_recipe/sampler_aligned_from_5.json \
  --train_continue --train_continue_from "" \
  --alg_b2b_convrefiner_resume_from "" --train_epoch latest --gpu_ids 1
```

## Checkpoints and resume

Fresh finetuning uses `train_continue_from` plus `train_epoch`; only new refiner/discriminator weights may be absent. `train_finetune` is deliberately rejected because its generic loader suppresses exceptions. Loading an incomplete trained refiner, or using a refiner checkpoint with a disabled-refiner configuration, raises an error.

Each checkpoint includes the full `*_net_G_A.pth`, its EMA sibling when enabled, `*_net_D_refiner.pth`, and `*_refiner_training.pth` containing optimizer, scheduler, scaler, counters, and per-rank RNG states. Resume with the saved run's configuration and `--train_continue --train_continue_from "" --train_epoch latest`; the original `320` suffix selects the source checkpoint only when starting the experiment. Training-state files contain Python/NumPy RNG objects and must come from a trusted run.

EMA updates occur only on optimizer steps. Checkpoint requests during gradient accumulation are skipped; epoch-end saving first flushes any partial accumulation group. Resume restores the saved epoch/batch cursor and requires the same distributed world size. Optimizer state is restored, but mid-epoch resumption with worker-side random augmentation is not guaranteed to reproduce the exact subsequent sample stream.

## Evaluation and deployment

The pilot writes `<epoch>_refiner_eval_<recording>.json` each epoch. To evaluate separately:

```bash
python scripts/eval_b2b_convrefiner.py \
  --train_config /path/to/new/run/train_config.json \
  --epoch latest --device cuda:1 --output /tmp/refiner_report.json
```

Validation compares identical held-out windows and noise with refinement off/on. It uses a clean reference frame and the configured samplers, reports masked PSNR, full-crop LPIPS/DISTS, temporal-difference MAE against ground truth, matched-region radial power spectra, and combined sampler latency. Spectra use luma, mean subtraction, and a Hann window on matching rectangular crops; masks are never multiplied into images before the FFT. Temporal-difference MAE is not optical-flow-warped error. This paired window protocol does not replace long autoregressive video review.

The numeric acceptance report requires reduced absolute spectral deficit, at most 0.5 dB PSNR loss, and at most 5% degradation in LPIPS/DISTS/temporal error. Missing metrics do not count as a pass. Inspect long autoregressive sequences for flicker, ghosting, ringing, and boundary artifacts before accepting a model. A high-frequency improvement alone is not proof of useful detail.

Existing PyTorch, ONNX, and TensorRT denoiser interfaces include refinement with unchanged tensor names. Export using the new run's config and checkpoint:

```bash
python scripts/b2b_export_onnx.py \
  --model_in_file /path/to/new/run/latest_net_G_A.pth \
  --train_config /path/to/new/run/train_config.json \
  --use_ema --export_mode denoiser --model_out_file /tmp/refiner.onnx
```

Rebuild the TensorRT engine from this ONNX file. SmartBrush continues to require denoiser export. The discriminator is training-only and is not exported. Native per-frame timesteps and the deployment runner's existing timestep policy are preserved; compare backends with equivalent timestep inputs.

## Tests

```bash
python -m pytest tests/test_b2b_convrefiner.py tests/test_convrefiner_recipe_metrics.py
CUDA_VISIBLE_DEVICES=1 JOLIGEN_TEST_TRT=1 python -m pytest tests/test_b2b_convrefiner.py -k tensorrt
CUDA_VISIBLE_DEVICES=1 JOLIGEN_REFINER_SOURCE_CONFIG=checkpoints/convrefiner_lvring_recipe/smoke.json \
  python -m pytest tests/test_b2b_convrefiner.py -k real_checkpoint
```

The distributed test uses two CPU processes and local Gloo sockets. Hardware tests are opt-in. The real-checkpoint test performs one update and one held-out paired evaluation; it does not start the pilot or alter the source checkpoint.
