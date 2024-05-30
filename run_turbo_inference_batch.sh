python3 scripts/gen_batch_image.py \
--model_in_file /data1/juliew/checkpoints/test_bdd100K_turbo_clear2snow/latest_net_G_A.pth \
--prompt_file /data1/juliew/dataset/bdd100k_weather_clear2snowy/testA/prompts.txt \
--output_folder /data1/juliew/joliGEN/bdd100_joliGEN_turbo_clear2snow/inference0611_deterministe  \
--gpuid 0 \
--compare \
--img_width 640  \
--img_height 360 \

