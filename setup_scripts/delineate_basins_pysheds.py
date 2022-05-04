
import os
import time
import json
import psutil

from multiprocessing import Pool

import pandas as pd
import numpy as np

import shapely
from shapely.geometry import Polygon, Point
import geopandas as gpd

# import xarray as xr
# import rioxarray as rxr
import rasterio  as rio

from raster_handler import RasterObject

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

group_codes = ['07G', '07O', '07U', '08A', '08B', '08C', '08D', '08E', '08F', '08G', '08H', '08N', '08O', '08P', '09A', 'ERockies', 'Fraser', 'Liard', 'Peace']
group_codes = ['08P', '08O', '07G', '07U', '07O', '08G', '08H']

version = '20220504'

output_folder = os.path.join(BASE_DIR, f'processed_data/processed_basin_polygons_{version}/')

# remove groups that have already been completed
processed_groups = list(set([e.split('_')[0] for e in os.listdir(output_folder)]))
group_codes = [g for g in group_codes if g not in processed_groups]

for group_code in group_codes:
    raster = RasterObject(group_code=group_code)

    n_pixels = raster.dem.shape[0] * raster.dem.shape[1] 

    t0 = time.time()
    raster.preprocess_dem()
    t1 = time.time()
    print('')
    print(f'   ...{n_pixels:.1f} pixel DEM loaded and preprocessed in {t1-t0:.1f}s')
    
    # list of stations found in the give region
    stations = raster.stations

    # snap the pour point locations to a cell
    # with an area that corresponds to the expected drainage area    

    pour_pts = []
    for stn in stations:
        pour_pts.append(raster.find_pour_point(stn))
    
    t2 = time.time()
    print(f'    ...{len(pour_pts)} pour points adjusted in {t2-t1:.1f}s')

    basins = []
    for ppt in pour_pts:
        basins.append(raster.derive_basin(ppt))

        
    basins = pd.concat(basins)
    basin_gdf = gpd.GeoDataFrame(geometry=basins.geometry, crs='EPSG:3005')
    basin_gdf['station'] = stations
    basin_gdf['station_area'] = [raster.get_baseline_drainage_area(stn) for stn in stations]
    basin_gdf['pct_area_delineated'] = basin_gdf.geometry.area / (basin_gdf['station_area'] * 1E6)

    ppt_gdf = gpd.GeoDataFrame(geometry=pour_pts, crs='EPSG:3005')
    ppt_gdf['station'] = stations

    out_path = output_folder + f'{group_code}_basins.geojson'
    gdf_path = os.path.join(out_path)
    basin_gdf.to_file(gdf_path)
    ppt_path = output_folder + f'{group_code}_ppts.geojson'
    basin_gdf.to_file(ppt_path)
    print(f'  saved {out_path}')


print('')
print('Basin delineation script complete.')
print('')
print('__________________________')
