import os
import pandas as pd

from multiprocessing import Pool

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEM_DIR = os.path.join(BASE_DIR, 'source_data/dem_data')
EP_DIR = os.path.join(DEM_DIR, 'USGS_3DEP')

# ensure the folders exist
for p in [DEM_DIR, EP_DIR]:
    if not os.path.exists(p):
        os.mkdir(p)

# to customize the study region, modify the array of lat/lon tuples [()]
url_df = pd.read_csv(BASE_DIR + '/setup_scripts/file_lists/merged-files.txt', header=None)
url_list = [e[0] for e in url_df.values if not os.path.exists(EP_DIR + '/' + e[0].split('/')[-1])]

def download_file(url):
    command = f'wget {url} -P {EP_DIR}'
    os.system(command)

p = Pool()
p.map(download_file, url_list)

# # this command builds the dem mosaic "virtual raster"
vrt_command = f"gdalbuildvrt -resolution highest -a_srs epsg:4326 {DEM_DIR}/BC_DEM_mosaic_4326.vrt {EP_DIR}/*.tif"
os.system(vrt_command)