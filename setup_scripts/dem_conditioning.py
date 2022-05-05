from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.pool import ThreadPool
import os
import time
import sys

import pandas as pd
import numpy as np

# from functools import partial

from whitebox.whitebox_tools import WhiteboxTools


import shapely
from shapely.geometry import Polygon, Point
import geopandas as gpd

# import xarray as xr
# import rioxarray as rxr
# import rasterio  as rio

wbt = WhiteboxTools()
wbt.verbose = True

t0 = time.time()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'source_data/')

def default_callback(msg):
    print(f'   ...completed {msg}.')
    print('')


def process_dem(source_dem_dir, dem_treatment, DEM_source):
        
    dir_method = 'D8' # D8, DINF
    # for region in code
    region_codes = sorted(list(set([e.split('_')[0] for e in os.listdir(source_dem_dir)])))

    dem_files = [f'{c}_{DEM_source}_3005_res1.tif' for c in region_codes]
    
    i = 0

    for file in dem_files:
        # get the covering region for the station
        i += 1
        t_start = time.time()
        region_code = file.split('_')[0]

        print(f'Starting dem pre-processing on region {region_code} {i}/{len(dem_files)}.')
        
        region_dem_path = os.path.join(source_dem_dir, file)

        if not os.path.exists(region_dem_path):
            print(region_dem_path)
            raise Exception; f'check region dem path.'

        t_path =  os.path.join(source_dem_dir, f'{region_code}_{DEM_source}_3005_res1_{dem_treatment}.tif')
        fdir_path = os.path.join(source_dem_dir, f'{region_code}_{DEM_source}_3005_res1_d8.tif')
        acc_path = os.path.join(source_dem_dir, f'{region_code}_{DEM_source}_3005_res1_acc.tif')

        processed_check = os.path.exists(acc_path) & os.path.exists(fdir_path)

        # if fdir and acc were already processed, skip creating them
        if processed_check:
            continue

        if dem_treatment == 'BCLC':         
            wbt.breach_depressions_least_cost(
                region_dem_path, 
                t_path, 
                100, 
                max_cost=None, 
                min_dist=True, 
                flat_increment=None, 
                fill=True, 
                callback=default_callback('breach depressions')
            )
        else:
            wbt.fill_depressions(
                region_dem_path, 
                t_path, 
                100, 
                callback=default_callback('fill depressions')
            )


        if not os.path.exists(fdir_path):
            wbt.d8_pointer(
                t_path, 
                fdir_path, 
                esri_pntr=False, 
                callback=default_callback('flow direction')
            )
    
        t_end_cond = time.time()
        t_cond = t_end_cond - t_start
        print(f'    ...completed conditioning, flow direction, and flow accumulation for {region_code} in {t_cond:.1f}s.')


        if not os.path.exists(acc_path):
            wbt.d8_flow_accumulation(
                t_path, 
                acc_path, 
                out_type="cells", 
                log=False, 
                clip=False,
                pntr=False, 
                esri_pntr=False, 
                callback=default_callback('flow accumulation')
            )
        os.remove(t_path)

    print('')
    print('Basin delineation script complete.')
    print('')
    print('__________________________')


if __name__ == '__main__':

    # specify the DEM source
    # either 'EarthEnv_DEM90' or 'USGS_3DEP'
    DEM_source = 'EarthEnv_DEM90'
    # DEM_source = 'PYSHEDS_burned_streams'
    DEM_source = 'USGS_3DEP'

    library = 'WBT'

    compute_loc = sys.argv[1]
    dem_treatment = sys.argv[2]

    if not dem_treatment:
        dem_treatment = 'FILL'

    if compute_loc == 'local':
        source_dem_dir = f'/media/danbot/Samsung_T5/geospatial_data/DEM_data/processed_dem/{DEM_source}/'
    else:
        source_dem_dir = os.path.join(BASE_DIR, f'processed_data/processed_dem/{DEM_source}/')

    process_dem(source_dem_dir, dem_treatment, DEM_source)
