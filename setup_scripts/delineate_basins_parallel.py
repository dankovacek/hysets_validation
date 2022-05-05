
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

from setup_scripts.raster_handler import RasterObject

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

group_codes = ['07G', '07O', '07U', '08A', '08B', '08C', '08D', '08E', '08F', '08G', '08H', '08N', '08O', '08P', '09A', 'ERockies', 'Fraser', 'Liard', 'Peace']
group_codes = ['08P']

version = '20220504'

output_folder = os.path.join(BASE_DIR, f'processed_data/processed_basin_polygons_{version}/')

for group_code in group_codes:
    raster = RasterObject(group_code=group_code)
        
    # use the published station locations as a first
    # approximation of the basin pour point.
    stations = raster.stations
    station_locs = raster.stn_locs

    # snap the pour point locations to a cell
    # with an area that corresponds to the expected drainage area
    expected_areas = [raster.get_baseline_drainage_area(s) for s in stations]

    inputs = list(zip(station_locs, expected_areas))

    pour_pts = []
    for i in inputs:
        pour_pts.append(raster.find_pour_point(i))

    basins = []
    for ppt in pour_pts:
        basins.append(raster.derive_basin(ppt))

        
    basins = pd.concat(basins)
    basin_gdf = gpd.GeoDataFrame(geometry=basins.geometry, crs='EPSG:3005')
    basin_gdf['station'] = stations
    basin_gdf['station_area'] = expected_areas
    basin_gdf['pct_area_delineated'] = basin_gdf.geometry.area / basin_gdf['station_area'] * 1E6

    gdf_path = os.path.join(output_folder + f'{group_code}.geojson')
    basin_gdf.to_file(gdf_path)


print('')
print('Basin delineation script complete.')
print('')
print('__________________________')
