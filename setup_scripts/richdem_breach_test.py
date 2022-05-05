import os

import richdem as rd

dem_source = 'EarthEnv_DEM90'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_data_dir = os.path.join(BASE_DIR, 'processed_data')
source_data_dir = os.path.join(BASE_DIR, 'source_data')
dem_folder = os.path.join(source_data_dir, f'dem_data/processed_dem/')

output_folder = os.path.join(source_data_dir, 'dem_data/breach_test/')



regions = sorted([e.split('_')[0] for e in os.listdir(dem_folder)])

for dem_file in os.listdir(dem_folder):
    region = dem_file.split('_')[0]
    if region != '08N':
        continue
    output_fname = f'{region}_breached.tiff'
    fpath = os.path.join(dem_folder, dem_file)
    dem = rd.LoadGDAL(fpath)
    dem_breached    = rd.BreachDepressions(dem, in_place=False)
    rd.SaveGDAL(os.path.join(output_folder, output_fname), dem_breached)

