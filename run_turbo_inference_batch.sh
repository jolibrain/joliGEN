python3 scripts/gen_batch_image.py \
--model_in_file /data1/juliew/checkpoints/test_bdd100K_turbo_multiprompt/latest_net_G_A.pth \
--prompt_file /data1/juliew/dataset/bdd100K_weather_clear2fograinsnowovercast/testBrain/prompts.txt \
--output_folder /data1/juliew/joliGEN/inference_test/ \
--gpuid 0 \
--compare \

