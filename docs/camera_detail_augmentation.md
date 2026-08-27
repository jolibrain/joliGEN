# Camera and Detail Augmentation

JoliGEN's DiffAugment pipeline provides `camera_color` and `detail` policies
for capture-domain robustness. Policies listed with commas are sampled
independently using `dataaug.diff_aug_proba`.

For B2B ring training with `self_supervised_vid_mask_online`, a recommended
starting configuration is:

```json
"dataaug": {
    "diff_aug_policy": "camera_color,detail,wild",
    "diff_aug_proba": 0.5,
    "diff_aug_camera_color_strength": 1.0,
    "diff_aug_detail_strength": 1.0,
    "diff_aug_camera_color_pre_crop": true
}
```

`camera_color` varies white balance, RGB response, exposure, levels, gamma,
saturation, tone response, camera color mixing, vignetting, and illumination
gradients. `detail` independently samples softening, unsharp masking, local
contrast, or a mild phone-like denoise/sharpen pipeline. Strength values are in
the range `[0, 1]`; zero makes that policy an identity operation.

The pre-crop flag is deliberately opt-in. It applies one camera plan to all
full frames in a temporal sample before local crops and masked global context
are constructed. This keeps their lighting and spatial camera effects aligned
and prevents the model-side pipeline from applying `camera_color` a second
time. It is supported only for B2B with the
`self_supervised_vid_mask_online` dataset.

For other datasets, leave `diff_aug_camera_color_pre_crop` disabled. The
generic tensor policy remains available, but spatial gradients use normalized
coordinates independently for each tensor. Existing `color` and `wild`
policies remain unchanged; `camera_color` usually replaces `color`, while
combining both intentionally produces stronger color variation.
