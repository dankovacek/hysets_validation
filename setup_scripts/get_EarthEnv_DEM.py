import os
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEM_DIR = os.path.join(BASE_DIR, 'source_data/dem_data')
# earth_env_folder = ''
EENV_DIR = os.path.join(DEM_DIR, 'EarthEnv_DEM90')

# ensure the folders exist
for p in [DEM_DIR, EENV_DIR]:
    if not os.path.exists(p):
        os.mkdir(p)

# to customize the study region, modify the array of lat/lon tuples [()]
coord_pairs = [('55', '145'), ('55', '140'), ('45', '120'), ('45', '125'), ('45', '130'), ('50', '115'), ('65', '135'), ('65', '140'), ('65', '145'), ('60', '140'), ('60', '145'), ('65', '125'), ('65', '130'), ('60', '120'), ('60', '125'), ('60', '130'), ('60', '135'), ('55', '120'), ('55', '125'), ('55', '130'), ('55', '135'), ('50', '120'), ('50', '125'), ('50', '130'), ('40', '120'), ('40', '115'), ('40', '125'), ('45', '115'), ('50', '135')]

# the download url format is the following:
# http://mirrors.iplantcollaborative.org/earthenv_dem_data/EarthEnv-DEM90/EarthEnv-DEM90_N55W110.tar.gz

for pair in coord_pairs:
    lat, lon = pair[0], pair[1]
    filename = f'EarthEnv-DEM90_N{lat}W{lon}.tar.gz'
    download_path = f'http://mirrors.iplantcollaborative.org/earthenv_dem_data/EarthEnv-DEM90/{filename}'
    command = f'wget {download_path} -P {EENV_DIR}'
    save_path = f'{EENV_DIR}/{filename}'
    if not os.path.exists(save_path):
        os.system(command)
    
        folder_name = filename.split('.')[0]
        os.system(f'tar -xf {EENV_DIR}/{filename} -C {EENV_DIR}')
        os.remove(f'{EENV_DIR}/{filename}')

# this command builds the dem mosaic "virtual raster"
vrt_command = f"gdalbuildvrt -resolution highest -a_srs epsg:4326 {DEM_DIR}/EENV_DEM_mosaic_4326.vrt {EENV_DIR}/EarthEnv-DEM90_*.bil"
os.system(vrt_command)