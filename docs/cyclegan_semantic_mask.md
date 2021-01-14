# Cyclegan with mask labels

All the options and parameters available for [cyclegan](cyclegan.md) can be tuned here too.

### Training parameters

|Parameter|Default|Values|Role|
|-|-|-|-|
|lambda_out_mask|10.0|Positive float|weight for out mask loss|
|loss_out_mask|L1|L1/MSE/Charbonnier|mask loss|
|lr_f_s|0.0002|Positive float|f_s learning rate|
|D_noise|0.0|Positive float|whether to add instance noise to discriminator inputs|
|rec_noise|0.0|Positive float|whether to add noise to reconstruction|
|nb_attn|10|Postive integer|number of attention masks (for attention generator)|
|nb_mask_input|0|Postive integer <= nb_attn|number of attention masks applied on input image (for attention generator)|


<br>

### Training options

|Option|Role|
|-|-|
|out_mask|activate the use of pixel out mask loss|
|disc_in_mask|activate the use of in-mask discriminator|
|train_f_s-B|train f_s on domain B (besides domain A)|
|fs_light|use a lighter network architecture for f_s|
|D_label_smooth|whether to use one-sided label smoothing with discriminator|
