DIR=$1

if [ $# -ne 1 ]
then
    echo "Usage: $0 [DIR]"
    exit 1
fi

echo "Specified [$DIR]"

current_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "Current dir is [$current_dir]"

export PYTHONDONTWRITEBYTECODE=1
export NCCL_P2P_DISABLE=1

######## doc auto generation
echo "Running doc auto generation"
python3 ${current_dir}/../scripts/generate_doc.py --save_to ""
OUT=$?

if [ $OUT != 0 ]; then
   exit 1
fi

####### CLI help
echo "CLI help"
python3 ${current_dir}/../train.py --help
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

python3 ${current_dir}/../train.py --help f_s
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

python3 ${current_dir}/../train.py --help alg
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

python3 ${current_dir}/../train.py --help alg_cut
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

python3 ${current_dir}/../train.py --help alg_palette
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

python3 ${current_dir}/../train.py --help alg_cm
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

####### no sem tests
echo "Running no semantics training tests"
URL=https://joligen.com/datasets/horse2zebra.zip
ZIP_FILE=$DIR/horse2zebra.zip
TARGET_NOSEM_DIR=$DIR/horse2zebra
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_NOSEM_DIR
unzip $ZIP_FILE -d $DIR
rm $ZIP_FILE

python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_nosemantic.py" --dataroot "$TARGET_NOSEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

#
####### test img2img_turbo
#echo "Running test img2img_turbo"
#
#python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_img2img_turbo.py" --dataroot "$TARGET_NOSEM_DIR"
#OUT=$?
#
#if [ $OUT != 0 ]; then
#    exit 1
#fi
#
####### mask semantics test
echo "Running mask semantics training tests"
URL=https://joligen.com/datasets/noglasses2glasses_ffhq_mini.zip
ZIP_FILE=$DIR/noglasses2glasses_ffhq_mini.zip
TARGET_MASK_SEM_DIR=$DIR/noglasses2glasses_ffhq_mini
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_MASK_SEM_DIR
unzip $ZIP_FILE -d $DIR
rm $ZIP_FILE

python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_semantic_mask.py" --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### self supervised mask semantics test
echo "Running self supervised mask semantics test"
python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_semantic_self_supervised.py" --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### test cut
echo "Running cut test"
python3 "${current_dir}/../test.py" \
        --save_config \
        --test_model_dir $DIR/joligen_utest_cut/ \
        --test_metrics_list FID KID PSNR LPIPS
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### diffusion process test
echo "Running diffusion process test"
python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_diffusion.py" --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### consistency model process test
echo "Running consistency model process test"
python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_cm.py" --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### shortcut model process test
echo "Running shortcut model process test"
python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_sc.py" --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### b2b model process test
echo "Running b2b model process test"
python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_b2b.py" --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi


###### GAN+supervised super-resolution process test
echo "Running GAN+supervised super-resolution process test"
python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_sr_gan.py" --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### diffusion super-resolution process test
echo "Running diffusion super-resolution process test"
python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_sr_diffusion.py" --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

######## mask cls as bbox semantics test
echo "Running mask cls as bbox semantics test"
python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_semantic_mask_online.py" --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### mask cls semantics test with online dataloading
echo "Running mask online semantics training tests"
URL=https://joligen.com/datasets/online_mario2sonic_lite2.zip 
ZIP_FILE=$DIR/online_mario2sonic_lite2.zip 
TARGET_MASK_SEM_ONLINE_DIR=$DIR/online_mario2sonic_lite2
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_MASK_SEM_ONLINE_DIR
unzip $ZIP_FILE -d $DIR
rm $ZIP_FILE
ln -s $TARGET_MASK_SEM_ONLINE_DIR/trainA $TARGET_MASK_SEM_ONLINE_DIR/testA
ln -s $TARGET_MASK_SEM_ONLINE_DIR/trainB $TARGET_MASK_SEM_ONLINE_DIR/testB

python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_semantic_mask_online.py" --dataroot "$TARGET_MASK_SEM_ONLINE_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### diffusion process test online
echo "Running diffusion process test online"
python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_diffusion_online.py" --dataroot "$TARGET_MASK_SEM_ONLINE_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### test palette
echo "Running test palette"
python3 "${current_dir}/../test.py" \
        --save_config \
        --test_model_dir $DIR/joligen_utest_palette/ \
        --test_metrics_list FID KID PSNR LPIPS
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

####### vid diffusion tests
echo "Running vid diffusion training tests"
URL=https://www.joligen.com/datasets/bdd100K_vid.zip 
ZIP_FILE=$DIR/bdd100K_vid.zip
TARGET_MASK_SEM_ONLINE_DIR=$DIR/bdd100k_vid
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_MASK_SEM_ONLINE_DIR
unzip $ZIP_FILE -d $DIR
rm $ZIP_FILE
ln -s $TARGET_MASK_SEM_ONLINE_DIR/trainA $TARGET_MASK_SEM_ONLINE_DIR/testA
ln -s $TARGET_MASK_SEM_ONLINE_DIR/trainB $TARGET_MASK_SEM_ONLINE_DIR/testB

python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_vid_diffusion_online.py" --dataroot "$TARGET_MASK_SEM_ONLINE_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### test vid palette
echo "Running test vid palette"
python3 "${current_dir}/../test.py" \
        --save_config \
        --test_model_dir $DIR/joligen_utest_vid_palette/ \
        --test_metrics_list SSIM PSNR LPIPS
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

####### cm_vid diffusion tests
echo "Running cm_vid diffusion training tests"
python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_cm_vid_diffusion_online.py" --dataroot "$TARGET_MASK_SEM_ONLINE_DIR"
=======
python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_ddpm_infer_ddim_online.py" --dataroot "$TARGET_MASK_SEM_ONLINE_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### test vid palette
echo "Running test vid palette train and ddim infer"
python3 "${current_dir}/../test.py" \
        --save_config \
        --test_model_dir $DIR/joligen_utest_vid_palette/ \
        --test_metrics_list SSIM PSNR LPIPS
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### test cycle_gan
# echo "Running test cycle_gan"
# python3 "${current_dir}/../test.py" \
# 	--test_model_dir $DIR/joligen_utest_cycle_gan/ \
# 	--test_metrics_list FID KID PSNR LPIPS
# OUT=$?

# if [ $OUT != 0 ]; then
#     exit 1
# fi

####### mask cls semantics test
echo "Running mask and class semantics training tests"
URL=https://joligen.com/datasets/daytime2dawn_dusk_lite.zip
ZIP_FILE=$DIR/daytime2dawn_dusk_lite.zip
TARGET_MASK_CLS_SEM_DIR=$DIR/daytime2dawn_dusk_lite
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_MASK_CLS_SEM_DIR
unzip $ZIP_FILE -d $DIR
rm $ZIP_FILE

python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_semantic_mask_cls.py" --dataroot "$TARGET_MASK_CLS_SEM_DIR"

OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

####### cls semantics test
echo "Running mask and class semantics training tests"
URL=https://joligen.com/datasets/mnist2USPS.zip
ZIP_FILE=$DIR/mnist2USPS.zip
TARGET_CLS_SEM_DIR=$DIR/mnist2USPS
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_CLS_SEM_DIR
unzip $ZIP_FILE -d $DIR
rm $ZIP_FILE

python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_semantic_cls.py" --dataroot "$TARGET_CLS_SEM_DIR"

OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

####### mask ref test
echo "Running mask ref training tests"
URL=https://joligen.com/datasets/viton_mask_ref_mini.zip
ZIP_FILE=$DIR/viton_mask_ref_mini.zip
TARGET_MASK_REF_DIR=$DIR/viton_mask_ref_mini
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_MASK_REF_DIR
unzip $ZIP_FILE -d $DIR
rm $ZIP_FILE

python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_mask_ref.py" --dataroot "$TARGET_MASK_REF_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

####### mask ref online test
echo "Running mask ref online training tests"
URL=https://joligen.com/datasets/viton_bbox_ref_mini.zip
ZIP_FILE=$DIR/viton_bbox_ref_mini.zip
TARGET_MASK_ONLINE_REF_DIR=$DIR/viton_bbox_ref_mini
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_MASK_ONLINE_REF_DIR
unzip $ZIP_FILE -d $DIR
rm $ZIP_FILE

python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_mask_online_ref.py" --dataroot "$TARGET_MASK_ONLINE_REF_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

####### pix2pix test
echo "Running pix2pix diffusion tests"
URL=https://joligen.com/datasets/SEN2VEN_mini.zip
ZIP_FILE=$DIR/SEN2VEN_mini.zip
TARGET_PIX2PIX_DIR=$DIR/SEN2VEN_mini
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_PIX2PIX_DIR
unzip $ZIP_FILE -d $DIR
rm $ZIP_FILE

echo "Running pix2pix GAN tests"
python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_pix2pix_gan.py" --dataroot "$TARGET_PIX2PIX_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

python3 -m pytest --rootdir ${current_dir} -p no:cacheprovider -s "${current_dir}/../tests/test_run_pix2pix_diffusion.py" --dataroot "$TARGET_PIX2PIX_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

#### Client server test
SERVER_HOST="localhost"
SERVER_PORT=8047

python3 -m pytest --rootdir ${current_dir} ${current_dir}/../tests/test_client_server.py --host $SERVER_HOST --port $SERVER_PORT

OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

######### api server tests
echo "Running api server tests"

python3 -m pytest --rootdir ${current_dir} \
    -p no:cacheprovider \
    -s "${current_dir}/../tests/test_api_predict_common.py"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

python3 -m pytest --rootdir ${current_dir} \
    -p no:cacheprovider \
    -s "${current_dir}/../tests/test_api_predict_gan.py" \
    --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

python3 -m pytest --rootdir ${current_dir} \
    -p no:cacheprovider \
    -s "${current_dir}/../tests/test_api_predict_diffusion.py" \
    --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

# End of tests
# Clean up
echo "Deleting target dir $DIR"
rm -rf $DIR/*
