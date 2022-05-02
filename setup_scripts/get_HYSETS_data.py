import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HYSETS_DIR = os.path.join(BASE_DIR, 'source_data/HYSETS_data')
STN_DIR = os.path.join(HYSETS_DIR, 'USGS_station_locations')
BASIN_DIR = os.path.join(HYSETS_DIR, 'HYSETS_basin_polygons')

# ensure the folders exist
for p in [HYSETS_DIR, STN_DIR, BASIN_DIR]:
    if not os.path.exists(p):
        os.mkdir(p)

# filename = 'USGS_Streamgages-NHD_Locations_Shape.zip'
download_path = 'https://water.usgs.gov/GIS/dsdl/'

# base_name = filename.split('.')[0]
# download the file
# command = f'wget {download_path}{filename} -O {HYSETS_DIR}/{filename}'
# save_path = f'{STN_DIR}/{filename}'
# if not os.path.exists(save_path):
#     print(f'    ...downloading file {filename} from {download_path}')
#     os.system(command)

# unzip the archive
# os.system(f'unzip {HYSETS_DIR}/{filename} -d {STN_DIR}')
# os.remove(f'{HYSETS_DIR}/{filename}')

# get the very large file containing basin polygons
download_path = 'https://files.osf.io/v1/resources/rpc3w/providers/googledrive/?zip='
basin_filename = 'HYSETS_watershed_boundaries'
command = f'wget {download_path}{basin_filename} -O {HYSETS_DIR}/{basin_filename}'
save_path = f'{BASIN_DIR}/{basin_filename}'
if not os.path.exists(save_path):
    print(f'    ...downloading file {basin_filename} from {download_path}')
    os.system(command)

os.system(f'unzip {HYSETS_DIR}/{basin_filename} -d {BASIN_DIR}')
os.remove(f'{HYSETS_DIR}/{basin_filename}')

