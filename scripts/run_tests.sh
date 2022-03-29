DIR=$1

if [ $# -ne 1 ]
then
    echo "Usage: $0 [DIR]"
    exit
fi

echo "Specified [$DIR]"

current_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "Current dir is [$current_dir]"

export PYTHONDONTWRITEBYTECODE=1
export NCCL_P2P_DISABLE=1

####### no sem tests
echo "Running no semantics training tests"
URL=https://www.deepdetect.com/joligan/datasets/horse2zebra.zip
ZIP_FILE=$DIR/horse2zebra.zip
TARGET_NOSEM_DIR=$DIR/horse2zebra
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_NOSEM_DIR
unzip $ZIP_FILE -d $DIR
rm $ZIP_FILE

python3 -m pytest -p no:cacheprovider -s "${current_dir}/../tests/test_run_nosemantic.py" --dataroot "$TARGET_NOSEM_DIR"
OUT=$?

echo "Deleting target dir $DIR"
rm -rf $DIR/*

if [ $OUT != 0 ]; then
    exit 1
fi

####### mask semantics test
echo "Running mask semantics training tests"
URL=https://www.deepdetect.com/joligan/datasets/noglasses2glasses_ffhq_mini.zip
ZIP_FILE=$DIR/noglasses2glasses_ffhq_mini.zip
TARGET_MASK_SEM_DIR=$DIR/noglasses2glasses_ffhq_mini
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_MASK_SEM_DIR
unzip $ZIP_FILE -d $DIR
rm $ZIP_FILE

python3 -m pytest -p no:cacheprovider -s "${current_dir}/../tests/test_run_semantic_mask.py" --dataroot "$TARGET_MASK_SEM_DIR"
OUT=$?

echo "Deleting target dir $DIR"
rm -rf $DIR/*

if [ $OUT != 0 ]; then
    exit 1
else
    exit 0
fi
