import os
import pandas as pd

from multiprocessing import Pool

DEM_source = 'USGS_3DEP'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEM_DIR = os.path.join(BASE_DIR, f'source_data/dem_data/{DEM_source}/')

# ensure the folders exist
for p in [DEM_DIR]:
    if not os.path.exists(p):
        os.makedirs(p)

# to customize the study region, modify the array of lat/lon tuples [()]
url_df = pd.read_csv(BASE_DIR + '/setup_scripts/file_lists/merged-files.txt', header=None)
url_list = [e[0] for e in url_df.values if not os.path.exists(DEM_DIR + '/' + e[0].split('/')[-1])]


def download_file(url):
    command = f'wget {url} -P {DEM_DIR}'
    os.system(command)

p = Pool()
p.map(download_file, url_list)

output_folder = os.path.join(BASE_DIR, 'processed_data/processed_dem')

# # this command builds the dem mosaic "virtual raster"
vrt_command = f"gdalbuildvrt -resolution highest -a_srs epsg:4269 {output_folder}/USGS_3DEP_mosaic_4269.vrt {DEM_DIR}/*.tif"
os.system(vrt_command)