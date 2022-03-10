import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WSC_DIR = os.path.join(BASE_DIR, 'source_data/WSC_data')
BASIN_DIR = os.path.join(WSC_DIR, 'WSC_basin_polygons')

# ensure the folders exist
for p in [WSC_DIR, BASIN_DIR]:
    if not os.path.exists(p):
        os.mkdir(p)

url = 'https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/HydrometricNetworkBasinPolygons/'
# these two-digit strings represent the set of WSC regions 
# that cover BC.  Modify to customize your study area.
# for zone in ['07', '08', '09', '10']:
#     filename = f'{zone}.zip'
#     command = f'wget {url}{filename} -O {BASIN_DIR}/{filename}'

#     # download the file
#     save_path = f'{BASIN_DIR}/{filename}'
#     if not os.path.exists(save_path):
#         print(f'    ...downloading file {filename} from {url}')
#         os.system(command)

#     # unzip the archive
#     os.system(f'unzip {BASIN_DIR}/{filename} -d {BASIN_DIR}')
#     os.remove(f'{BASIN_DIR}/{filename}')

# remove qgz files and leave the folders
for f in os.listdir(BASIN_DIR):
    if f.endswith('.qgz'):
        os.remove(os.path.join(BASIN_DIR,f))

# separate the files into separate folders: 
# basin (polygon), pour point (point), station (point)
all_stations = [e for e in os.listdir(BASIN_DIR)]
for station in all_stations:
    stn_path = os.path.join(BASIN_DIR, station)
    all_stn_files = os.listdir(stn_path)
    for f in ['DrainageBasin', 'PourPoint', 'Station']:
        stn_files = [e for e in all_stn_files if (f in e) & (os.path.isfile(os.path.join(stn_path, e)))]
        new_folder = os.path.join(BASIN_DIR, station + '/' + f)
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)

        for stn_file in stn_files:
            old_path = os.path.join(stn_path, stn_file)
            new_path = os.path.join(new_folder, stn_file)
            os.replace(old_path, new_path)
        
