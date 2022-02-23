DIR=$1

if [ $# -ne 1 ]
then
    echo "Usage: $0 [DIR]"
    exit
fi

echo "Specified [$DIR]"

current_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "Current dir is [$current_dir]"

URL=http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip
ZIP_FILE=$DIR/horse2zebra.zip
TARGET_DIR=$DIR/horse2zebra
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d $DIR
rm $ZIP_FILE

export NCCL_P2P_DISABLE=1

pytest -s "${current_dir}/../tests/" --dataroot "$TARGET_DIR"
