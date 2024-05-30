cd scripts
python3 gen_single_image.py \
--model_in_file /data1/juliew/checkpoints/test_bdd100K_turbo_clear2snow/latest_net_G_A.pth \
--img_in /media/baie_nas/data/datasets/bdd100k/bdd100k/images/100k/train//00f0dd0f-5e9c9557.jpg   \
--img_out /data1/juliew/joliGEN/bdd100_joliGEN_turbo_clear2snow/inference0612_change_prompt/00f0dd0f-5e9c9557_snow.jpg  \
--prompt driving in the snowy weather \
--gpuid  0  \
--compare \
--img_width 640  \
--img_height 360 \

