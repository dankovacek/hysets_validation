from concurrent.futures import process
import os
import time
import pickle

import pandas as pd
import numpy as np

import shapely
from shapely.geometry import Polygon, Point
import geopandas as gpd

import xarray as xr
import rioxarray as rxr
import rasterio  as rio

from pysheds.grid import Grid

t0 = time.time()

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

source_data_dir = os.path.join(base_dir, 'source_data/')
processed_data_dir = os.path.join(base_dir, 'processed_data/')

data_dir = '/media/danbot/Samsung_T5/geospatial_data/'
dem_dir = data_dir + 'DEM_data/'
processed_dem_path = dem_dir + 'processed_dem/'

t7_media_path = '/media/danbot/T7 Touch/thesis_data/processed_stations/'

processed_basins_path = '/home/danbot/Documents/code/hysets_validation/processed_data/derived_basins/pysheds/'


code_dict_path = base_dir + '/validate_hysets/20220211_code_dict.pickle'
with open(code_dict_path, 'rb') as handle:
    code_dict = pickle.load(handle)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# now create polygons using the raster just generated
def retrieve_raster(fpath):
    rds = rxr.open_rasterio(fpath)
    crs = rds.rio.crs
    affine = rds.rio.transform(recalc=False)
    return rds, crs, affine


def get_acc_threshold(res, min_area):
    return int(min_area / (abs(res[0] * res[1])))


def fill_holes(data):
           
    interior_gaps = data.interiors.values.tolist()[0]
    
    # group_name = data.index.values[0]
    gap_list = []
    if interior_gaps is not None:
        # print(f'   ...{len(interior_gaps)} gaps found in {group_name} groupings.')
        for i in interior_gaps:
            gap_list.append(Polygon(i))
        data_gaps = gpd.GeoDataFrame(geometry=gap_list, crs=data.crs)
        
        appended_set = data.append(data_gaps)
        appended_set['group'] = 0
        merged_polygon = appended_set.dissolve(by='group')
        return merged_polygon.geometry.values[0]
    else:
        # print(f'  ...no gaps found in {group_name}')
        return data.geometry.values[0]


def pysheds_basin_polygon(stn, grid, catch, crs, affine, out_path):

    # Create view
    catch_view = grid.view(catch, dtype=np.uint8)

    # Create a vector representation of the catchment mask
    mask = catch_view > 0
    polygons = rio.features.shapes(catch_view, mask=mask, transform=affine)  
    geoms = [shapely.geometry.shape(shp) for shp, _ in polygons]    
    basin_polygon = shapely.ops.unary_union(geoms)

    if basin_polygon.is_empty:
        print(f'    ...basin polygon retrieval failed for {stn}')
        return False
    else:
        gdf = gpd.GeoDataFrame(geometry=[basin_polygon], crs=crs)
        gdf['geom_type'] = gdf.geometry.geom_type
        gdf['geometry'] = fill_holes(gdf)
        gdf.to_file(out_path, driver='GeoJSON')
        return True


def pysheds_delineation(grid, fdir, acc, snap_point_path, threshold):

    sp = gpd.read_file(snap_point_path, driver='GeoJSON')
    sp_pt = sp.geometry.values[0]
    x, y = sp_pt.x, sp_pt.y

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Snap pour point to high accumulation cell
    x_snap, y_snap = grid.snap_to_mask(acc > threshold, (x, y))

    distance = np.sqrt((x_snap - x)**2 + (y_snap - y)**2)
    # print(f'   ...difference btwn. nearest and PYSHEDS snap = {distance:1f}')

    # Delineate the catchment
    catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype='coordinate')
    
    return catch


def pysheds_condition_dem(grid, dem):
    pits = grid.detect_pits(dem)
    if pits.any():
        filled_dem = grid.fill_pits(dem)
    else:
        filled_dem = dem

    depressions = grid.detect_depressions(filled_dem)
    if depressions.any():
        flooded_dem = grid.fill_depressions(filled_dem)
    else:
        flooded_dem = filled_dem

    flats = grid.detect_flats(flooded_dem)
    if flats.any():
        inflated_dem = grid.resolve_flats(flooded_dem)
    else:
        inflated_dem = flooded_dem

    return inflated_dem
    

# bad stations on low
# ['07GE001', '07OB001', '07OB003', '09CA003', '08MG021', '08MH141', '08LE075', '09AF001', '10AB003', '10EA002', '08KC003']

# bad stations on med
# ['07GE001', '07OB001', '07OB003', '09CA003', '08MG021', '08MH141', '08LE075', '09AF001', '10AB003', '10EA002', '08KC003']

# bad stations on hi
# 

dir_method = 'D8' # D8, DINF
delineation_method = 'PYSHEDS'
# for region in code
region_codes = sorted(list(set(code_dict.values())))

bad_basins = []
i = 0
for resolution in ['low', 'med']:
    for region_code in ['Fraser']:#region_codes:
        # get the covering region for the station
        t_start = time.time()

        print(f'Starting analysis on region {region_code} {i}/{len(region_codes)}.')
        # load the region DEM once and iterate through all
        region_dem_path = os.path.join(processed_dem_path, f'{region_code}_DEM_3005_{resolution}.tif')

        grid = Grid.from_raster(region_dem_path)

        dem = grid.read_raster(region_dem_path)

        conditioned_dem = pysheds_condition_dem(grid, dem)

        fdir = grid.flowdir(conditioned_dem)
        acc = grid.accumulation(fdir)
        t_end_cond = time.time()
        t_cond = t_end_cond - t_start
        print(f'    ...completed conditioning, flow direction, and flow accumulation for {region_code} in {t_cond:.1f}.')
       
        dem_raster, dem_crs, dem_affine = retrieve_raster(region_dem_path)
        acc_threshold = get_acc_threshold(dem_raster.rio.resolution(), min_area=1E6)
        
        stations = sorted([s for s in list(set(code_dict.keys())) if code_dict[s] == region_code])

        j = 0
        for station in ['08MG026']:#stations:
            if j % 10 == 0:
                print(f'   Deriving basin for {station} {j}/{len(stations)}')
            # stations in the region to trim the number of redundant file loadings
            basin_out_path = processed_basins_path + f'{station}_{delineation_method}_basin_derived_{resolution}.geojson'

            if not os.path.exists(basin_out_path):

                snap_point_path = f'/media/danbot/Samsung_T5/geospatial_data/WSC_data/reprojected_data/{station}/{station}_pour_point.geojson'

                catch = pysheds_delineation(grid, fdir, acc, snap_point_path, acc_threshold)
                
                basin_created = pysheds_basin_polygon(station, grid, catch, dem_crs, dem_affine, basin_out_path)
                
                if not basin_created:
                    bad_basins.append(station)
                    print(f'    ...basin creation failed for {station}.')
                else:
                    if j % 10 == 0:
                        print(f'   ...completed basin polygon creation for {station}')
                
            j += 1

    i += 1

print('The following basin delineations failed:')
print(bad_basins)