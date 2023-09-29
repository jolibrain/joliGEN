python3 train.py --dataroot /data3/juliew/joligen_gan/joliGEN_WIP/datasets/noglasses2glasses_ffhq\
    --checkpoints_dir ./checkpoints\
    --name noglasses2glasses\
    --output_display_env noglasses2glasses\
    --config_json examples/example_gan_noglasses2glasses.json\
    --D_netDs projected_d unet_128_d
