import sys
# sys.path.append('/usr/local/lib64/python2.7/site-packages')
sys.path.append('/usr/local/lib')
import functools as f
import multiprocessing
import os
# rio env --gdal-data
os.environ['GDAL_DATA'] = '/home/brodt/.local/lib/python3.6/site-packages/rasterio/gdal_data'
import itertools as it
from pathlib import Path
import geopandas as gpd
from matplotlib import pyplot as plt
import numpy as np
from pystac import Catalog
from rasterio.transform import from_bounds
from rio_tiler import main as rt_main
from shapely.geometry import Polygon
import solaris as sol
import skimage
import tqdm

# os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

# We have to add this wrkaround for stackio:
# (https://pystac.readthedocs.io/en/latest/concepts.html#using-stac-io)
# from pystac import STAC_IO


# def my_read_method(uri):
#     parsed = urlparse(uri)
#     if parsed.scheme.startswith('http'):
#         return requests.get(uri).text
#     else:
#         return STAC_IO.default_read_text_method(uri)

    
# STAC_IO.read_text_method = my_read_method


def save_tile_img(tif_url, xyz, tile_size, save_path='', prefix='', display=False):
    x, y, z = xyz
    img_save_path = f'{save_path}/{prefix}{z}_{x}_{y}.png'
    if os.path.exists(img_save_path):
        return
    
    tile, mask = rt_main.tile(tif_url, x, y, z, tilesize=tile_size)

    if display: 
        plt.imshow(np.moveaxis(tile, 0, 2))
        plt.show()

    skimage.io.imsave(img_save_path, np.moveaxis(tile, 0, 2), check_contrast=False) 

    
def save_tile_mask(labels_poly, tile_poly, xyz, tile_size, save_path='', prefix='', display=False, greyscale=True):
    x, y, z = xyz
    img_save_path = f'{save_path}/{prefix}{z}_{x}_{y}.png'
    if os.path.exists(img_save_path):
        return
    
    # get affine transformation matrix for this tile using rasterio.transform.from_bounds: https://rasterio.readthedocs.io/en/stable/api/rasterio.transform.html#rasterio.transform.from_bounds
    tfm = from_bounds(*tile_poly.bounds, tile_size, tile_size) 

    # crop geometries to what overlaps our tile polygon bounds
    cropped_polys = [poly for poly in labels_poly if poly.intersects(tile_poly)]
    cropped_polys_gdf = gpd.GeoDataFrame(geometry=cropped_polys, crs='epsg:4326')

    # burn a footprint/boundary/contact 3-channel mask with solaris: https://solaris.readthedocs.io/en/latest/tutorials/notebooks/api_masks_tutorial.html
    fbc_mask = sol.vector.mask.df_to_px_mask(df=cropped_polys_gdf,
                                             channels=['footprint', 'boundary', 'contact'],
                                             affine_obj=tfm,
                                             shape=(tile_size, tile_size),
                                             boundary_width=5,
                                             boundary_type='inner',
                                             contact_spacing=5,
                                             meters=True)

    if display:
        plt.imshow(fbc_mask)
        plt.show()
  
    if greyscale:
        skimage.io.imsave(img_save_path, fbc_mask[:,:,0], check_contrast=False) 
    else:
        skimage.io.imsave(img_save_path, fbc_mask, check_contrast=False) 
        

def process_tile(row, tif_url, all_polys, tile_size, img_path, area, img_id, mask_path, display, greyscale):
    idx, item = row
    tile_poly = item['geometry']
    xyz = item['xyz']
    try:
        save_tile_img(tif_url,
                      xyz,
                      tile_size, 
                      save_path=img_path, prefix=f'{area}_{img_id}_{idx}_',
                      display=display)

        save_tile_mask(all_polys, 
                       tile_poly,
                       xyz,
                       tile_size,
                       save_path=mask_path, prefix=f'{area}_{img_id}_{idx}_',
                       display=display,
                       greyscale=greyscale)
    except:
        print(f'Cannot process tile {area} {img_id} {idx}')


def save_area_id_images(area, img_id, label_id, zoom_level=19, tile_size=1024, display=False, greyscale=False):
    # The item
    one_item = cols[area].get_item(id=img_id)

    # Load labels shapefile
    lab = cols[area].get_item(id=label_id)
    gdf = gpd.read_file(lab.make_asset_hrefs_absolute().assets['labels'].href)
    # get the geometries from the geodataframe
    all_polys = gdf.geometry

    # Get outlines as polygons
    polygon_geom = Polygon(one_item.to_dict()['geometry']['coordinates'][0])
    polygon = gpd.GeoDataFrame(index=[0], crs=gdf.crs, geometry=[polygon_geom])   

    # Tile at zoom_level
    polygon['geometry'].to_file(img_id + '.geojson', driver='GeoJSON')
    cmd = f'cat {img_id}.geojson | supermercado burn {zoom_level} | mercantile shapes | fio collect > {img_id}{zoom_level}tiles.geojson'
    os.system(cmd)

    # Load tiles
    tiles = gpd.read_file(f'{img_id}{zoom_level}tiles.geojson')

    # Add a convenience column
    tiles['xyz'] = tiles.id.apply(lambda x: x.lstrip('(,)').rstrip('(,)').split(','))
    tiles['xyz'] = [[int(q) for q in p] for p in tiles['xyz']]

    # IMG URL
    tif_url = one_item.assets['image'].href

    # Sometimes it's just ./id.tif - add full path (should maybe use make_asset_hrefs_absolute instead!!)
    if tif_url.startswith('./'):
        tif_url = '/'.join(one_item.to_dict()['links'][1]['href'].split('/')[:-1]) + tif_url[1:]

    print('TIF URL:', tif_url)
    print('Number of tiles:', len(tiles))

    # Loop through tiles, downloading and saving
    with multiprocessing.Pool(32) as p:
        _ = list(p.imap_unordered(func=f.partial(process_tile, 
                                                 tif_url=tif_url,
                                                 all_polys=all_polys,
                                                 tile_size=tile_size,
                                                 img_path=img_path,
                                                 area=area,
                                                 img_id=img_id,
                                                 mask_path=mask_path,
                                                 display=display,
                                                 greyscale=greyscale,
                                                ),
                                  iterable=tiles.iterrows()))


# Folder Setup
data_dir = Path('data/train_tier_1_tiles_1024/')
data_dir.mkdir(exist_ok=True, parents=True)

img_path = data_dir / 'images'
img_path.mkdir(exist_ok=True)

mask_path = data_dir / 'masks'
mask_path.mkdir(exist_ok=True)

# load our training and test catalogs
train1_cat = Catalog.from_file('data/train_tier_1/catalog.json')
# train2_cat = Catalog.from_file('https://drivendata-competition-building-segmentation.s3-us-west-1.amazonaws.com/train_tier_2/catalog.json')

# test_cat = Catalog.from_file('https://drivendata-competition-building-segmentation.s3-us-west-1.amazonaws.com/test/catalog.json')

cols = {cols.id:cols for cols in train1_cat.get_children()}

# Get a list of the possible areas ('scenes) and ids
areas = []
for c in cols:
    itms = [x for x in cols[c].get_all_items()]
    for i, id in enumerate(itms):
        if i % 2 == 0 and i + 1 < len(itms):
            areas.append((c, itms[i].id, itms[i + 1].id))

            
# You could fetch all the data with:
with tqdm.tqdm(areas) as pbar:
    for a in pbar:
        save_area_id_images(*a, zoom_level=19, tile_size=1024)
