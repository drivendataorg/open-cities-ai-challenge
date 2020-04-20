RAW_DATA_DIR=data/raw
INTER_DATA_DIR=data/interim
PROCESSED_DATA_DIR=data/processed

mkdir -p $INTER_DATA_DIR
mkdir -p $PROCESSED_DATA_DIR

echo "Extracting test files..."
tar -xvzf $RAW_DATA_DIR/test.tgz -C $INTER_DATA_DIR

# stitch test small images into big tif files
echo "Stitching test files..."
python -m src.data.stitch_test \
    --df_path=$PROCESSED_DATA_DIR/test_mosaic.csv \
    --path_pattern=$INTER_DATA_DIR/test/*/*.tif \
    --dst_dir=$PROCESSED_DATA_DIR/test
