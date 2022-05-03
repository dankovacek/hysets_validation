from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.pool import ThreadPool
import os
import time
import json

# import queue
# import threading

import pandas as pd
import numpy as np

# from functools import partial

import ray
import psutil

# from numba import prange, jit

# import richdem as rd

import shapely
from shapely.geometry import Polygon, Point
import geopandas as gpd

# import xarray as xr
# import rioxarray as rxr
import rasterio  as rio

from pysheds.grid import Grid

num_cpus = psutil.cpu_count(logical=False)

ray.init(num_cpus=num_cpus)

# import warnings
# warnings.filterwarnings('ignore')

class MainData:
    def __init__(
        self, 
        DEM_source,
        revision, 
        compute_loc='local',
        DEM_treatment='FILLED', 
        snap_method='SNAMMINMAX',
        crs = 3005,
        ):

        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'source_data/')
        self.DEM_source = DEM_source

        if compute_loc == 'local':
            # if dem_treatment == 'BURNED':
            self.processed_dem_dir = f'/media/danbot/Samsung_T5/geospatial_data/DEM_data/processed_dem/{DEM_source}/'
        else:
            self.processed_dem_dir = os.path.join(self.BASE_DIR, f'processed_data/processed_dem/{DEM_source}')

        self.DEM_treatment = DEM_treatment
        self.snap_method = snap_method

        self.code_dict = self.set_mapper_dict()
        self.group_codes = sorted(list(set(self.code_dict.values())))
        self.dem_files = os.listdir(self.processed_dem_dir)

        self.output_folder = os.path.join(self.BASE_DIR, f'processed_data/processed_basin_polygons_{revision}/{DEM_source}_{DEM_treatment}_{snap_method}')

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        hysets_data_path = os.path.join(self.BASE_DIR, 'source_data/HYSETS_data/')
        hysets_df = pd.read_csv(hysets_data_path + '/HYSETS_watershed_properties.txt', sep=';')

        USGS_stn_locs_path = hysets_data_path + 'USGS_station_locations/'
        usgs_df = gpd.read_file(USGS_stn_locs_path, layer='USGS_Streamgages-NHD_Locations')

        usgs_df = usgs_df.to_crs(crs)
        usgs_df['Official_ID'] = usgs_df['SITE_NO']

        wsc_path = os.path.join(self.BASE_DIR, 'source_data/WSC_data/WSC_basin_polygons')
        wsc_stns = os.listdir(wsc_path)

        hysets_df = pd.read_csv(os.path.join(self.BASE_DIR, 'source_data/HYSETS_data/HYSETS_watershed_properties.txt'), sep=';')

        hysets_locs = [Point(x, y) for x, y in zip(hysets_df['Centroid_Lon_deg_E'].values, hysets_df['Centroid_Lat_deg_N'])]
        hysets_df = gpd.GeoDataFrame(hysets_df, geometry=hysets_locs, crs='EPSG:4269')
        self.hysets_df = hysets_df.to_crs(crs)

        # load the SEAK basins
        # SEAK_file = 'USFS_Southeast_Alaska_Drainage_Basin__SEAKDB__Watersheds.geojson'
        # seak_path = os.path.join(BASE_DIR, f'source_data/AK_Coastline/{SEAK_file}')
        # seak_gdf = gpd.read_file(seak_path)
        # print(seak_gdf)
        # print(asdf)


    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def set_mapper_dict(self):
        # create a dictionary where the key: value pairs 
        # map stations to the regional group name
        stn_mapper_path = os.path.join(self.BASE_DIR, 'processed_data/')
        mapper_dict_file = 'station_to_region_mapper.json'

        if not os.path.exists(os.path.join(stn_mapper_path, mapper_dict_file)):
            raise Exception; '  Mapper file not found.  You need to first run process_complete_basin_groups.py'
   
        # with open(stn_mapper_path + mapper_dict_file, 'rb') as handle:
        #     code_dict = np.load(handle)
        json_filepath = stn_mapper_path + mapper_dict_file
        with open(json_filepath, 'r') as json_file:
            json_str = json.load(json_file)
            return json.loads(json_str)


    def get_baseline_drainage_area(self, stn):
        data = self.hysets_df[self.hysets_df['Official_ID'] == stn]
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


    def get_hysets_ppt(self, stn):
        # use the baseline station location
        # print(f'     {station} location found in HYSETS, using published location.')
        location = self.hysets_df[self.hysets_df['Official_ID'] == stn]
        hysets_loc = location.geometry.values[0]
        return (hysets_loc.x, hysets_loc.y)


class RasterData:
    def __init__(self, group_code, res_code, main_data):
        group_code = group_code
        main_data = main_data
        DEM_source = main_data.DEM_source

        dem_fname = f'{group_code}_{DEM_source}_3005_{res_code}.tif'
        region_dem_path = os.path.join(main_data.processed_dem_dir, dem_fname)
        if not os.path.exists(region_dem_path):
            print(region_dem_path)
            raise Exception; f'check region dem path.'

        self.stations = sorted([s for s in list(set(main_data.code_dict.keys())) if main_data.code_dict[s] == group_code])
        print(f'   ...{len(self.stations)} stations found in {group_code}.')
        
        self.grid = Grid.from_raster(region_dem_path)
        dem = self.grid.read_raster(region_dem_path, data_name='dem')
        print('   ...Dem loaded.')
        self.raster_res = dem.dy_dx[::-1]
            # conditioned_dem = pysheds_condition_dem(grid, dem)
        filled_dem = self.grid.fill_pits(dem)
        print('   ...Fill pits completed.')
        del dem

        flooded_dem = self.grid.fill_depressions(filled_dem)
        print('   ...Fill depressions completed.')
        del filled_dem

        conditioned_dem = self.grid.resolve_flats(flooded_dem, max_iter=1E12, 
        eps=1E-12)
        print('   ...Resolve flats completed.')
        del flooded_dem
        
        # clear conditioned dem before creating flow direction
        # del dem
        
        # if the DEM conditioning is ineffective, we can't do anything
        # beyond this point to fix it...
        self.fdir = self.grid.flowdir(conditioned_dem, out_name='fdir')
        print('   ...Flow direction completed.')
        
        # clear conditioned dem from memory before creating acc
        del conditioned_dem
        self.acc = self.grid.accumulation(self.fdir, out_name='acc')
        print('   ...Flow accumulation completed.')
        self.affine = self.acc.affine


class Basin:
    def __init__(
        self, 
        station_id,
        snap_method,
        baseline_DA,
        output_folder,
        DEM_source,
        raster_res,
        stn_loc):

        self.stn = station_id
        self.snap_method = snap_method
        self.baseline_DA = baseline_DA
        self.expected_area_m = self.baseline_DA * 1E6
        self.stn_loc = stn_loc
        self.min_threshold = 1E6
        self.raster_res = raster_res
        self.expected_threshold = self.baseline_DA * 1E6
        self.output_folder = output_folder
        self.output_path = os.path.join(output_folder, f'{self.stn}_{DEM_source}_basin.geojson')

    
    def get_acc_threshold(self, min_area_m):
        # determine the number of cells to use as the threshold
        # for snapping the pour point. 
        return int(min_area_m / (abs(self.raster_res[0] * self.raster_res[1])))


    def pysheds_basin_polygon(self, crs):
        
        # snapped_loc_gdf = gpd.GeoDataFrame(geometry=[pour_point], crs=hysets_df.crs)
        # snapped_loc_gdf.to_file(os.path.join(stn_mapper_path, 'snapped_locations_hires'))

        # Delineate the catchment
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        catch = rdat.grid.catchment(x=self.pour_point.x, y=self.pour_point.y, fdir=rdat.fdir, dirmap=dirmap, xytype='coordinate')

        # Create view
        catch_view = rdat.grid.view(catch, dtype=np.uint8)

        # # Create a vector representation of the catchment mask
        mask = catch_view > 0
        polygons = rio.features.shapes(catch_view, mask=mask, transform=catch.affine)  
        geoms = [shapely.geometry.shape(shp) for shp, _ in polygons]    
        basin_polygon = shapely.ops.unary_union(geoms)

        if basin_polygon.is_empty:
            print(f'    ...basin delineation failed for {self.stn}')
            return None
        else:
            gdf = gpd.GeoDataFrame(geometry=[basin_polygon], crs=crs)
            geom = self.fill_holes(gdf)
            derived_area = geom.area / 1E6
            area_r = 100 * derived_area / self.baseline_DA
            data = {
                'out_path': self.output_path,
                'derived_area_km': derived_area,
                'area_r': area_r,
            }
            df = pd.DataFrame(data)
            gdf = gpd.GeoDataFrame(df, geometry=[geom])
            print('')
            print('')
            print('constructed gdf')
            print(gdf)
            print(f'    ...delineated basin is {area_r:.0f}% baseline area for {self.stn}')
            print('')
            print('')
            return gdf


    # @jit(nopython=True)
    def affine_map_vec_numba(self, affine, x, y):
        a, b, c, d, e, f, _, _, _ = affine
        n = x.size
        new_x = np.zeros(n, dtype=np.float32)
        new_y = np.zeros(n, dtype=np.float32)
        for i in range(n):
            new_x[i] = x[i] * a + y[i] * b + c
            new_y[i] = x[i] * d + y[i] * e + f
        return new_x, new_y


    def get_nearest_point(self, area_threshold, threshold_cells, bounds_type='minmax'):
        # mask for just the cells in the expected range
        max_cells = int((1 + area_threshold) * threshold_cells)
        min_cells = int((1 - area_threshold) * threshold_cells)

        if bounds_type == 'min':
            yi, xi = np.where((rdat.acc > min_cells))
        else:
            yi, xi = np.where((rdat.acc >= min_cells) & (rdat.acc <= max_cells))

        affine_tup = tuple(rdat.affine)
        
        # convert to coordinates
        x, y = self.affine_map_vec_numba(affine_tup, xi, yi)
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
        stn_coords = (self.stn_loc[0], self.stn_loc[1])

        diffs = np.array([np.subtract(stn_coords, c) for c in mask_coords])
        
        dists = [np.linalg.norm(d, ord=2) for d in diffs]
        
        min_xy = mask_coords[np.argmin(dists)]

        min_dist = np.linalg.norm(np.subtract(min_xy, stn_coords))
        print(f'    Min. distance from pour point to threshold flow acc cell = {min_dist:.0f} m')

        return True, min_xy


    def find_pour_point(self):    
        # Snap pour point to threshold accumulation cell
        distance = 0
        x, y = self.stn_loc[0], self.stn_loc[1]
        x_snap, y_snap = x, y
        # try:
            # as a first try, use a minimum of 1km^2 accumulation threshold
        d1 = False
        threshold_cells = self.get_acc_threshold(self.min_threshold)
        # x_snap, y_snap = rdat.grid.snap_to_mask(rdat.acc > threshold_cells, (x, y))
        # shift_distance = np.sqrt((x_snap - x)**2 + (y_snap - y)**2)
        # except Exception as e:
        #     print(f'   ...Snap to mask failed {threshold_cells:.0f} cells.')
        #     d1 = False
        #     x_snap, y_snap = x, y

        # if d1:
        #     distance = shift_distance
        
        # max. allowable distance to move a station (m)
        max_shift_distance_m = 350 
        # set +/- bounds for finding snap points with a given area
        area_thresh = 0.23
        if self.snap_method == 'SNAPMINEX':
            threshold_cells = self.get_acc_threshold(self.expected_area_m)
            nr_found, (x_nr, y_nr) = self.get_nearest_point(area_thresh, threshold_cells, 'min')
            try:
                distance = np.sqrt((x_nr - x)**2 + (y_nr - y)**2)
                distance_diff = abs(distance - original_distance)
                print(f'   pour point adjusted by {distance:.0f}m for min threshold.')
                if distance_diff <= max_shift_distance_m:
                    x_snap, y_snap = x_nr, y_nr
            except Exception as e:
                print(f'   ...Min threshold failed {threshold_cells:.0f} cells.')
                print(e)
                d1 = False

        elif snap_method == 'SNAPMINMAX':
            original_distance = distance
            threshold_cells = self.get_acc_threshold(self.expected_area_m)
            nr_found, (x_nr, y_nr) = self.get_nearest_point(area_thresh, threshold_cells, 'range')
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
                    print(f'Error delineating basin for {self.stn}:')
                    print(e)
                    print(asdf)
                    # print(f'   Area threshold range failed.')
                    break      

        return Point(x_snap, y_snap)


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
            appended_set.loc[:, 'group'] = 0
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


@ray.remote
def derive_basin(basin):
    # stations in the region to trim the number of redundant file loadings
    # baseline_DA = get_baseline_drainage_area(station)
    # print(f'    ...{station} baseline DA: {baseline_DA:.1f}km^2')
    # find the pour point
    basin.pour_point = basin.find_pour_point()
    basin.basin_gdf = basin.pysheds_basin_polygon('EPSG:3005')
    if basin.basin_gdf is not None:
        print(f'    ...completed basin polygon creation for {basin.stn}.')
        return basin.basin_gdf
    else: 
        return None
        
dir_method = 'D8' # D8, DINF

resolution_code = 'res1'
compute_loc = 'local'
DEM_source = 'USGS_3DEP' 
DEM_treatment = 'FILLED'
snap_method = 'SNAPMINMAX'

revision = '20220503'

main_data = MainData(        
    DEM_source,
    revision, 
    compute_loc,
    DEM_treatment, 
    snap_method
    )

ray.put(main_data)


i = 0
for group_code in ['08P']:# main_data.group_codes:# region_codes:
    # get the covering region for the station
    i += 1
    t_start = time.time()

    print(f'Starting analysis on region {group_code} {i}/{len(main_data.group_codes)}.')

    rdat = RasterData(group_code, resolution_code, main_data)
    
    t_end_cond = time.time()
    t_cond = t_end_cond - t_start
    print(f'  Completed hydraulic enforcing for {group_code} in {t_cond:.1f}s.')
    print('##################################################')
    
    ray.put(rdat)

    stations = rdat.stations

    basins = []
    for stn in stations:
        
        baseline_DA = main_data.get_baseline_drainage_area(stn)
        stn_loc = main_data.get_hysets_ppt(stn)
        output_folder = main_data.output_folder
        raster_res = rdat.raster_res

        basin = Basin(
            stn,
            snap_method,
            baseline_DA,
            output_folder,
            DEM_source,
            raster_res,
            stn_loc
            )

        basins.append(basin)

    # print(asdfsd)

    basin_results = ray.get([derive_basin.remote(b) for b in basins])

    print(basin_results)
    print(asdfasdfasd)
    
    # basin_results = [e for e in basin_results if e is not None]
    if len(basin_results) > 0:
        for g, path in basin_results:
            g.to_file(path, driver='GeoJSON')

    t_end = time.time()
    print(f'    Completed {len(rdat.stations)} stations for {group_code} in {t_end - t_start:.1f}s')
    
    
print('')
print('Basin delineation script complete.')
print('')
print('__________________________')
