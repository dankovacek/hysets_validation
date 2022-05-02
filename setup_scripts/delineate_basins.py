from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.pool import ThreadPool
import os
import time
import json

import queue
import threading

import pandas as pd
import numpy as np

# from functools import partial

# import multiprocessing

from multiprocessing import Pool

# from numba import prange, jit

# import richdem as rd

import shapely
from shapely.geometry import Polygon, Point
import geopandas as gpd

import xarray as xr
import rioxarray as rxr
import rasterio  as rio

from pysheds.grid import Grid

import warnings
warnings.filterwarnings('ignore')


t0 = time.time()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'source_data/')

# specify the DEM source
# either 'EarthEnv_DEM90' or 'USGS_3DEP'
DEM_source = 'EarthEnv_DEM90'
# DEM_source = 'PYSHEDS_burned_streams'
DEM_source = 'USGS_3DEP'

# data_dir = '/media/danbot/Samsung_T5/geospatial_data/'
processed_dem_dir = os.path.join(BASE_DIR, f'processed_data/processed_dem/{DEM_source}')

# processed_dem_dir = f'/media/danbot/Samsung_T5/geospatial_data/DEM_data/processed_dem/{DEM_source}'

dem_treatment = 'FILLED' #
# if 'burned' in DEM_source:
# dem_treatment = 'BURNED'

# if dem_treatment == 'BURNED':
# processed_dem_dir = f'/media/danbot/Samsung_T5/geospatial_data/DEM_data/processed_dem/{DEM_source}/'


snap_method = 'SNAPMIN' # snap to minimum area (no prior knowledge)
snap_method = 'SNAPMINEX' # snap to expected minimum area (add min prior)
snap_method = 'SNAPMINMAX' # snap to expected area range (add min and max prior)

output_basin_polygon_path = os.path.join(BASE_DIR, f'processed_data/processed_basin_polygons_20220502/{DEM_source}_{dem_treatment}_{snap_method}')

if not os.path.exists(output_basin_polygon_path):
    os.makedirs(output_basin_polygon_path)

hysets_data_path = os.path.join(BASE_DIR, 'source_data/HYSETS_data/')
hysets_df = pd.read_csv(hysets_data_path + '/HYSETS_watershed_properties.txt', sep=';')

USGS_stn_locs_path = hysets_data_path + 'USGS_station_locations/'
usgs_df = gpd.read_file(USGS_stn_locs_path, layer='USGS_Streamgages-NHD_Locations')

usgs_df = usgs_df.to_crs(3005)
usgs_df['Official_ID'] = usgs_df['SITE_NO']

wsc_path = os.path.join(BASE_DIR, 'source_data/WSC_data/WSC_basin_polygons')
wsc_stns = os.listdir(wsc_path)

hysets_df = pd.read_csv(os.path.join(BASE_DIR, 'source_data/HYSETS_data/HYSETS_watershed_properties.txt'), sep=';')

hysets_locs = [Point(x, y) for x, y in zip(hysets_df['Centroid_Lon_deg_E'].values, hysets_df['Centroid_Lat_deg_N'])]
hysets_df = gpd.GeoDataFrame(hysets_df, geometry=hysets_locs, crs='EPSG:4269')
hysets_df = hysets_df.to_crs(3005)

# load the SEAK basins
# SEAK_file = 'USFS_Southeast_Alaska_Drainage_Basin__SEAKDB__Watersheds.geojson'
# seak_path = os.path.join(BASE_DIR, f'source_data/AK_Coastline/{SEAK_file}')
# seak_gdf = gpd.read_file(seak_path)
# print(seak_gdf)
# print(asdf)

def get_grp_polygon(basin_polygons_path, polygon_fnames, grp_code):
    grp_polygon_path = [e for e in polygon_fnames if grp_code in e][0]
    grp_polygon = gpd.read_file(basin_polygons_path + grp_polygon_path)
    grp_polygon = grp_polygon.to_crs(3005)
    grp_polygon['grp'] = grp_code
    grp_polygon = grp_polygon.dissolve(by='grp', aggfunc='sum')
    return grp_polygon


# create a dictionary where the key: value pairs 
# map stations to the regional group name
stn_mapper_path = os.path.join(BASE_DIR, 'processed_data/')
mapper_dict_file = 'station_to_region_mapper.json'


if not os.path.exists(os.path.join(stn_mapper_path, mapper_dict_file)):
    raise Exception; '  Mapper file not found.  You need to first run process_complete_basin_groups.py'
   

# with open(stn_mapper_path + mapper_dict_file, 'rb') as handle:
#     code_dict = np.load(handle)
json_filepath = stn_mapper_path + mapper_dict_file
with open(json_filepath, 'r') as json_file:
    json_str = json.load(json_file)
    code_dict = json.loads(json_str)

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
    # determine the number of cells to use as the threshold
    # for snapping the pour point. 
    return int(min_area / (abs(res[0] * res[1])))


def fill_holes(data):
    interior_holes = data.interiors.dropna().tolist()
    interior_holes = [e for sublist in interior_holes for e in sublist]
    gap_list = []
    if len(interior_holes) > 0:
        print(f'   ...{len(interior_holes)} holes found to be filled.')
        for hole in interior_holes:
            gap_list.append(Polygon(hole))

        data_gaps = gpd.GeoDataFrame(geometry=gap_list, crs=data.crs)
        # appended_set = pd.concat([data, data_gaps])
        appended_set = data.append(data_gaps)
        appended_set['group'] = 0
        merged_polygon = appended_set.dissolve(by='group', aggfunc='sum')
        print(f'     merged geometry has {len(merged_polygon)} polygon (s)')
        if len(merged_polygon) == 1:
            return merged_polygon.geometry.values[0]
        else:
            raise Exception; 'Unmerged polygons in fill_holes()'
    else:
        if len(data) == 1:
            return data.geometry.values[0]
        else:
            raise Exception; "Original data needs to be dissolved into single polygon"


def pysheds_basin_polygon(stn, grid, fdir, crs, baseline_area, pour_point):
    
    # snapped_loc_gdf = gpd.GeoDataFrame(geometry=[pour_point], crs=hysets_df.crs)
    # snapped_loc_gdf.to_file(os.path.join(stn_mapper_path, 'snapped_locations_hires'))

    # Delineate the catchment
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    catch = grid.catchment(x=pour_point.x, y=pour_point.y, fdir=fdir, dirmap=dirmap, xytype='coordinate')

    # Create view
    catch_view = grid.view(catch, dtype=np.uint8)

    # # Create a vector representation of the catchment mask
    mask = catch_view > 0
    polygons = rio.features.shapes(catch_view, mask=mask, transform=catch.affine)  
    geoms = [shapely.geometry.shape(shp) for shp, _ in polygons]    
    basin_polygon = shapely.ops.unary_union(geoms)

    if basin_polygon.is_empty:
        print(f'    ...basin delineation failed for {stn}')
        return None
    else:
        gdf = gpd.GeoDataFrame(geometry=[basin_polygon], crs=crs)
        gdf['geom_type'] = gdf.geometry.geom_type
        gdf['geometry'] = fill_holes(gdf)
        derived_area = gdf.geometry.area.values[0] / 1E6
        area_r = 100 * derived_area / baseline_area
        print(f'    ...delineated basin is {area_r:.0f}% baseline area for {stn}')
        print('')
        return gdf


# @jit(nopython=True)
def affine_map_vec_numba(affine, x, y):
    a, b, c, d, e, f, _, _, _ = affine
    n = x.size
    new_x = np.zeros(n, dtype=np.float32)
    new_y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        new_x[i] = x[i] * a + y[i] * b + c
        new_y[i] = x[i] * d + y[i] * e + f
    return new_x, new_y


def get_nearest_point(stn_loc, acc, area_threshold, expected_area, bounds_type='minmax'):    
    
    # convert to BC albers for distance calc
    # if stream_raster.rio.crs.to_epsg() != 3005:
    #     stream_raster = stream_raster.rio.reproject('EPSG:3005')

    # assert stn_loc.crs == 'epsg:3005'

    stn_loc = stn_loc.geometry.values[0]

    # mask for just the cells in the expected range
    max_cells = int((1 + area_threshold) * expected_area)
    min_cells = int((1 - area_threshold) * expected_area)

    if bounds_type == 'min':
        yi, xi = np.where((acc > min_cells))
    else:
        yi, xi = np.where((acc >= min_cells) & (acc <= max_cells))

    affine_tup = tuple(acc.affine)
    
    # convert to coordinates
    x, y = affine_map_vec_numba(affine_tup, xi, yi)
    # print('how many returned from affine map?')
    # print(len(x))

    if len(x) == 0:
        print('   No point found.')
        print('   ')
        return False, (None, None)
        

    # mask_indices = np.c_[xi, yi]

    # convert to array of tuples
    mask_coords = np.c_[x, y]
    # print('how many mask coords returned?')
    # print(len(mask_coords))
    # calculate distances from station to flow accumulation points
    stn_coords = (stn_loc.x, stn_loc.y)

    diffs = np.array([np.subtract(stn_coords, c) for c in mask_coords])
    
    dists = [np.linalg.norm(d, ord=2) for d in diffs]
    
    min_xy = mask_coords[np.argmin(dists)]

    # min_indices = mask_indices[np.argmin(dists)]

    # closest_target_acc = acc[min_indices[0]-4:min_indices[0]+4, min_indices[1]-4:min_indices[1]+4]

    min_dist = np.linalg.norm(np.subtract(min_xy, stn_coords))
    print(f'    Min. distance from pour point to threshold flow acc cell = {min_dist:.0f} m')

    return True, min_xy



def find_pour_point(acc, station, baseline_DA, raster_res, snap_method):

    expected_area_m = baseline_DA * 1E6

    if station in hysets_df['Official_ID'].values:
        # use the baseline station location
        # print(f'     {station} location found in HYSETS, using published location.')
        location = hysets_df[hysets_df['Official_ID'] == station]
        hysets_loc = location.geometry.values[0]
        x, y = hysets_loc.x, hysets_loc.y

    else:
        # try retrieving the pour point from WSC
        # print(f'   {station} found in WSC directory, using published pour point.')
        pp_loc = os.path.join(wsc_path, f'{station}/PourPoint/')
        layer_name = f'{station}_PourPoint_PointExutoire'
        location = gpd.read_file(pp_loc, layer=layer_name)
        location = location.to_crs(3005)
        ppt = location.geometry.values[0]
        x, y = ppt.x, ppt.y
   
    # Snap pour point to threshold accumulation cell
    distance = 0
    x_snap, y_snap = x, y
    try:
        # as a first try, use a minimum of 1km^2 accumulation threshold
        threshold_cells = get_acc_threshold(raster_res, 1E6)
        d1 = True
        x_snap, y_snap = grid.snap_to_mask(acc > threshold_cells, (x, y))
        shift_distance = np.sqrt((x_snap - x)**2 + (y_snap - y)**2)
        # print(f'   pour point adjusted by {shift_distance:.0f}m for min threshold.')
    except Exception as e:
        print(f'   ...Snap to mask failed {threshold_cells:.0f} cells.')
        d1 = False
        x_snap, y_snap = x, y

    if d1:
        distance = shift_distance

    max_shift_distance_m = 350 # max. allowable distance to move a station (m)
    area_thresh = 0.23
    if snap_method == 'SNAPMINEX':
        threshold_cells = get_acc_threshold(raster_res, expected_area_m)
        nr_found, (x_nr, y_nr) = get_nearest_point(location, acc, area_thresh, threshold_cells, 'min')
        try:
            distance = np.sqrt((x_nr - x)**2 + (y_nr - y)**2)
            distance_diff = abs(distance - original_distance)
            print(f'   pour point adjusted by {shift_distance:.0f}m for min threshold.')
            if distance_diff <= max_shift_distance_m:
                x_snap, y_snap = x_nr, y_nr
        except Exception as e:
            print(f'   ...Min threshold failed {threshold_cells:.0f} cells.')
            print(e)
            d1 = False

    elif snap_method == 'SNAPMINMAX':
        original_distance = distance
        expected_area_m = baseline_DA * 1E6
        threshold_cells = get_acc_threshold(raster_res, expected_area_m)
        nr_found, (x_nr, y_nr) = get_nearest_point(location, acc, area_thresh, threshold_cells, 'range')
        while (area_thresh > 0.0):
            if not nr_found:
                break
            try:                
                distance = np.sqrt((x_nr - x)**2 + (y_nr - y)**2)
                distance_diff = abs(distance - original_distance)
                if (distance_diff > max_shift_distance_m):
                    break
                else:
                    x_snap, y_snap = x_nr, y_nr
                
                if area_thresh <= 0.06:
                    area_thresh -= 0.02
                else:
                    area_thresh -= 0.04
            except Exception as e:
                print(f'Error delineating basin for {station}:')
                print(e)
                print(asdf)
                # print(f'   Area threshold range failed.')
                break      

    pour_point = Point(x_snap, y_snap)

    return distance, pour_point


def check_for_baseline_drainage_area(stn):
    data = hysets_df[hysets_df['Official_ID'] == stn]
    area = data['Drainage_Area_km2'].values[0]
    gsim_flag = data['Flag_GSIM_boundaries'] = 1
    if gsim_flag:
        gsim_area = data['Drainage_Area_GSIM_km2'].values[0]
        if (not np.isnan(area)) & (not np.isnan(gsim_area)):
            area_ratio = area / gsim_area
        if gsim_area > area:
            area = gsim_area
            print(f'    Using GSIM area for {stn}. A/gsim_A: {area_ratio}')
    return area


def derive_basin(path):
    station = path.split('/')[-1].split('_')[0]
        
    # stations in the region to trim the number of redundant file loadings
    baseline_DA = check_for_baseline_drainage_area(station)
    # print(f'    ...{station} baseline DA: {baseline_DA:.1f}km^2')

    # if not os.path.exists(path):            
    distance, pour_point = find_pour_point(acc, station, baseline_DA, raster_res, snap_method)
    
    basin_gdf = pysheds_basin_polygon(station, grid, fdir, 'EPSG:3005', baseline_DA, pour_point)
    
    if basin_gdf is not None:
        print(f'    ...completed basin polygon creation for {station} {distance:.0f}m from reported location.')
        return (basin_gdf, path)
    else:
        return None
        
dir_method = 'D8' # D8, DINF
delineation_method = 'PYSHEDS'
# for region in code
region_codes = sorted(list(set(code_dict.values())))

bad_basins = []
i = 0

dem_files = os.listdir(processed_dem_dir)

# print(len(dem_files))
# print(asdfsd)

resolution = 'res1'

# '09A', '08F'
#
for region_code in ['08P']:# region_codes:
    # get the covering region for the station
    i += 1
    t_start = time.time()

    print(f'Starting analysis on region {region_code} {i}/{len(region_codes)}.')
    
    # load the region DEM once and iterate through all
    # region_dem_path = os.path.join(processed_dem_dir, f'{region_code}_EarthEnv_DEM90_3005_{resolution}.tif')
    # region_dem_path = os.path.join(processed_dem_dir, f'{region_code}_{DEM_source}.tif')
    # region_dem_path = os.path.join(processed_dem_dir, f'{region_code}_{DEM_source}_burned_flattened2_streams_{resolution}.tif')
    # nd_val = -32768
    # load the region DEM once and iterate through all
    # if region_code in ['08A', '07O']:
    #     DEM_source = 'USGS_3DEP'      
    # else:
    #     DEM_source = 'EarthEnv_DEM90'
    region_dem_path = os.path.join(processed_dem_dir, f'{region_code}_{DEM_source}_3005_{resolution}.tif')
    if not os.path.exists(region_dem_path):
        print(region_dem_path)
        raise Exception; f'check region dem path.'
    if dem_treatment == 'BURNED':
        region_dem_path = os.path.join(processed_dem_dir, f'{region_code}_{DEM_source}_burned_streams_{resolution}.tif')
        print(region_dem_path)

    
    file = region_dem_path.split('/')[-1]
    print(f'    Opening DEM {file}')

    grid = Grid.from_raster(region_dem_path, data_name='dem')
    dem = grid.read_raster(region_dem_path, data_name='dem')

    # dem = dem.astype(np.uint32)

    raster_res = dem.dy_dx[::-1]
    
    # conditioned_dem = pysheds_condition_dem(grid, dem)
    filled_dem = grid.fill_pits(dem)
    del dem

    flooded_dem = grid.fill_depressions(filled_dem)
    del filled_dem

    conditioned_dem = grid.resolve_flats(flooded_dem, max_iter=1E12, eps=1E-12)
    del flooded_dem
    
    # clear conditioned dem before creating flow direction
    # del dem
    
    # if the DEM conditioning is ineffective, we can't do anything
    # beyond this point to fix it...
    fdir = grid.flowdir(conditioned_dem, out_name='fdir')
    
    # clear conditioned dem from memory before creating acc
    del conditioned_dem
    acc = grid.accumulation(fdir, out_name='acc')

    
    t_end_cond = time.time()
    t_cond = t_end_cond - t_start
    print(f'    ...completed conditioning, flow direction, and flow accumulation for {region_code} in {t_cond:.1f}s.')
    
    # dem_raster, dem_crs, dem_affine = retrieve_raster(region_dem_path)

    # raster_res = dem_raster.rio.resolution()
    # min_threshold = get_acc_threshold(raster_res, min_area=1E6)
    
    stations = sorted([s for s in list(set(code_dict.keys())) if code_dict[s] == region_code])

    output_paths = [os.path.join(output_basin_polygon_path, f'{s}_{DEM_source}_basin.geojson') for s in stations]

    futures = []
    basin_results = []
    for p in output_paths:
        with ThreadPoolExecutor(max_workers=10) as executor:
            for op in output_paths:
                futures.append(executor.submit(derive_basin, op))

            for future in as_completed(futures):
                if future.result() is not None:
                    basin_results.append(future.result())
    # clear fdir and acc from memory
    del fdir
    del acc
    
    basin_results = [e for e in basin_results if e is not None]
    if len(basin_results) > 0:
        for g, path in basin_results:
            g.to_file(path, driver='GeoJSON')

    t_end = time.time()
    print(f'    Completed {len(stations)} stations for {region_code} in {t_end - t_start:.1f}s')
    
    
print('')
print('Basin delineation script complete.')
print('')
print('__________________________')
