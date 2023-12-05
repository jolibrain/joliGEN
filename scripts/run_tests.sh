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

####### doc auto generation
echo "Running doc auto generation"
python3 ${current_dir}/../scripts/generate_doc.py --save_to ""
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

python3 -m pytest -p no:cacheprovider -s "${current_dir}/../tests/test_run_nosemantic.py" --dataroot "$TARGET_NOSEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

####### mask semantics test
echo "Running mask semantics training tests"
URL=https://joligen.com/datasets/noglasses2glasses_ffhq_mini.zip
ZIP_FILE=$DIR/noglasses2glasses_ffhq_mini.zip
TARGET_MASK_SEM_DIR=$DIR/noglasses2glasses_ffhq_mini
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_MASK_SEM_DIR
unzip $ZIP_FILE -d $DIR
rm $ZIP_FILE

python3 -m pytest -p no:cacheprovider -s "${current_dir}/../tests/test_run_semantic_mask.py" --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### self supervised mask semantics test
python3 -m pytest -p no:cacheprovider -s "${current_dir}/../tests/test_run_semantic_self_supervised.py" --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### diffusion process test
python3 -m pytest -p no:cacheprovider -s "${current_dir}/../tests/test_run_diffusion.py" --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### diffusion super-resolution process test
python3 -m pytest -p no:cacheprovider -s "${current_dir}/../tests/test_run_sr_diffusion.py" --dataroot "$TARGET_MASK_SEM_DIR/trainA/img"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

####### mask cls as bbox semantics test
python3 -m pytest -p no:cacheprovider -s "${current_dir}/../tests/test_run_semantic_mask_online.py" --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

####### mask cls semantics test with online dataloading
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


python3 -m pytest -p no:cacheprovider -s "${current_dir}/../tests/test_run_semantic_mask_online.py" --dataroot "$TARGET_MASK_SEM_ONLINE_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi


###### diffusion process test online
python3 -m pytest -p no:cacheprovider -s "${current_dir}/../tests/test_run_diffusion_online.py" --dataroot "$TARGET_MASK_SEM_ONLINE_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### test cut
echo "Running test cut"
python3 "${current_dir}/../test.py" \
	--save_config \
	--test_model_dir $DIR/joligen_utest_cut/ \
	--test_metrics_list FID KID PSNR LPIPS
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

###### test cycle_gan
echo "Running test cycle_gan"
python3 "${current_dir}/../test.py" \
	--test_model_dir $DIR/joligen_utest_cycle_gan/ \
	--test_metrics_list FID KID PSNR LPIPS
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

####### mask cls semantics test
echo "Running mask and class semantics training tests"
URL=https://joligen.com/datasets/daytime2dawn_dusk_lite.zip
ZIP_FILE=$DIR/daytime2dawn_dusk_lite.zip
TARGET_MASK_CLS_SEM_DIR=$DIR/daytime2dawn_dusk_lite
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_MASK_CLS_SEM_DIR
unzip $ZIP_FILE -d $DIR
rm $ZIP_FILE

python3 -m pytest -p no:cacheprovider -s "${current_dir}/../tests/test_run_semantic_mask_cls.py" --dataroot "$TARGET_MASK_CLS_SEM_DIR"

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

python3 -m pytest -p no:cacheprovider -s "${current_dir}/../tests/test_run_semantic_cls.py" --dataroot "$TARGET_CLS_SEM_DIR"


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

python3 -m pytest -p no:cacheprovider -s "${current_dir}/../tests/test_run_mask_ref.py" --dataroot "$TARGET_MASK_REF_DIR"
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

python3 -m pytest -p no:cacheprovider -s "${current_dir}/../tests/test_run_mask_online_ref.py" --dataroot "$TARGET_MASK_ONLINE_REF_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

####### pix2pix test
echo "Running pix2pix tests"
URL=https://joligen.com/datasets/SEN2VEN_mini.zip
ZIP_FILE=$DIR/SEN2VEN_mini.zip
TARGET_PIX2PIX_DIR=$DIR/SEN2VEN_mini
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_PIX2PIX_DIR
unzip $ZIP_FILE -d $DIR
rm $ZIP_FILE

python3 -m pytest -p no:cacheprovider -s "${current_dir}/../tests/test_run_pix2pix_diffusion.py" --dataroot "$TARGET_PIX2PIX_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

echo "Deleting target dir $DIR"
rm -rf $DIR/*

#### Client server test
SERVER_HOST="localhost"
SERVER_PORT=8047

python3 -m pytest ${current_dir}/../tests/test_client_server.py --host $SERVER_HOST --port $SERVER_PORT

OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

# ####### api server tests
echo "Running api server tests"

python3 -m pytest \
    -p no:cacheprovider \
    -s "${current_dir}/../tests/test_api_predict_common.py"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

python3 -m pytest \
    -p no:cacheprovider \
    -s "${current_dir}/../tests/test_api_predict_gan.py" \
    --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi

python3 -m pytest \
    -p no:cacheprovider \
    -s "${current_dir}/../tests/test_api_predict_diffusion.py" \
    --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

if [ $OUT != 0 ]; then
    exit 1
fi
