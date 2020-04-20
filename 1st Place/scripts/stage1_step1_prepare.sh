RAW_DATA_DIR=data/raw
INTER_DATA_DIR=data/interim
PROCESSED_DATA_DIR=data/processed

mkdir -p $INTER_DATA_DIR
mkdir -p $PROCESSED_DATA_DIR

# unpack data archives
echo "Extracting train tier 1 files..."
tar -xvzf $RAW_DATA_DIR/train_tier_1.tgz -C $INTER_DATA_DIR

# resample train data to 0.1 meters per pixel
echo "Resamping train data to 0.1 m/pixel"
python -m src.data.resample --path_pattern=$INTER_DATA_DIR/train_tier_1/*/*/*.tif --dst_res=0.1

# for each resampled tif file create raster mask from provided geometries in .geojson file
echo "Creating raster masks"
python -m src.data.generate_masks --path_pattern=$INTER_DATA_DIR/train_tier_1/*/*/res-*.tif

# slicing big tif files to small ones with size 1024x1024
echo "Slicing tif files..."
python -m src.data.cut_train \
    --path_pattern=$INTER_DATA_DIR/train_tier_1/*/*/res-*.tif \
    --dst_dir=$PROCESSED_DATA_DIR/train \
    --sample_size=1024
