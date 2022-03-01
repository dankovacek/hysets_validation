
import os
import pickle

import numpy as np
import pandas as pd

import rioxarray as rxr
import geopandas as gpd


data_dir = '/media/danbot/Samsung_T5/geospatial_data/'
hysets_dir = os.path.join(data_dir, 'HYSETS_data/')
# hydat_dir = os.path.join(data_dir, 'hydat_db/')
dem_dir = os.path.join(data_dir, 'DEM_data/')

processed_dem_dir = os.path.join(dem_dir, 'processed_dem/')

derived_basin_path = '/home/danbot/Documents/code/hysets_validation/processed_data/derived_basins/'

# file containing derived watershed properties used in hysets
hysets_props_fname = 'HYSETS_watershed_properties.txt'
# import the set of derived watershed properties from hysets
hysets_props = pd.read_csv(hysets_dir + hysets_props_fname, delimiter=';')


with open('20220211_code_dict.pickle', 'rb') as handle:
    code_dict = pickle.load(handle)

def retrieve_raster(dem_path):
    # open dem and mask using basin polygon
    raster = rxr.open_rasterio(dem_path, masked=True)
    crs = raster.rio.crs.to_epsg()
    if not crs:
        crs = raster.rio.crs.to_wkt()
    return raster, crs

    
def clip_raster(raster, crs, basin_polygon):   
    basin_polygon = basin_polygon.to_crs(crs)
    bounds = tuple(basin_polygon.bounds.values[0])
    subset_raster = raster.rio.clip_box(*bounds)
    return subset_raster.rio.clip(basin_polygon.geometry, basin_polygon.crs)


def process_raster_stats(all_data):
    data = all_data.data[0].flatten()
    data = data[~np.isnan(data)]
    n_pixels = len(data)
    unique, counts = np.unique(data, return_counts=True)
    return unique, counts, n_pixels

region_codes = sorted(list(set(code_dict.values())))

dirtype = 'D8' # or DIRINF
resolution = 'low'
all_data = all_data = {}
anomalous_stations = {}
for region in region_codes:
    print(f'Starting assessment of {region} region.')
    stations = [e for e in code_dict.keys() if code_dict[e] == region]

    # retrieve the files once per region
    region_dem_file = processed_dem_dir + f'{region}_DEM_3005_{resolution}.tif'
    region_dir_file = processed_dem_dir + f'{region}_DIR_3005_{resolution}.tif'
    region_acc_file = processed_dem_dir + f'{region}_WBT_ACC_{dirtype}_FILL_3005_{resolution}.tif'

    region_raster, region_crs = retrieve_raster(region_acc_file)


    cell_size = region_raster.rio.resolution()
    raster_crs = region_raster.rio.crs.to_epsg()

    for station in stations:
        basin_path = derived_basin_path + f'{station}_basin_{resolution}.geojson'
        try:
            basin_polygon = gpd.read_file(basin_path, driver='GeoJSON')
        except Exception as e:
            print(f'   {station} has no file {basin_path}')
            continue

        try:
            basin_data = clip_raster(region_raster, region_crs, basin_polygon)
        except Exception as e:
            print(f'    {station} raster clip failed')
            print(e)
            continue
        
        unique, counts, pixels = process_raster_stats(basin_data)
        all_data[station] = {'unique': unique, 'counts': counts, 'pixels': pixels}

        pixel_area = abs(cell_size[0]) * abs(cell_size[1])
        total_area = pixel_area * pixels / 1E6
        
        geom_area = basin_polygon.geometry.area / 1E6
        hysets_info = hysets_props[hysets_props['Official_ID'] == station]
        hysets_area = hysets_info['Drainage_Area_km2'].values[0]
        area_diff = abs(hysets_area - geom_area)
        anomalous_stations[station] = [area_diff]

            

anomalies_df = pd.DataFrame(anomalous_stations).T
data_df = pd.DataFrame(all_data).T

anomalies_df.to_csv(f'results/anomalies_{dirtype}_{resolution}.csv')
data_df.to_csv(f'results/acc_values_{dirtype}_{resolution}.csv')

