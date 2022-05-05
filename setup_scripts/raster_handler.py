
import os
import time
import json
from tokenize import group

import pandas as pd
import numpy as np

import shapely
from shapely.geometry import Polygon, Point
import geopandas as gpd

import rasterio as rio

from pysheds.grid import Grid

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'source_data/')


class RasterObject:
    def __init__(
        self,
        group_code,
        dir_method='D8',
        resolution_code='res1',
        compute_loc='local',
        DEM_source='USGS_3DEP', 
        DEM_treatment='FILL',
        snap_method = 'SNAPMINMAX',
        crs = 3005):

        self.group_code = group_code
        self.dir_method = dir_method
        self.DEM_source = DEM_source
        self.resoluton_code = resolution_code
        self.snap_method = snap_method
        self.crs = crs
        self.DEM_treatment = DEM_treatment

        if compute_loc == 'local':
            # if dem_treatment == 'BURNED':
            self.processed_dem_dir = f'/media/danbot/Samsung_T5/geospatial_data/DEM_data/processed_dem/{DEM_source}/'
        else:
            self.processed_dem_dir = os.path.join(BASE_DIR, f'processed_data/processed_dem/{DEM_source}')

        self.retrieve_station_data()
        self.code_dict = self.set_mapper_dict()
        self.stations = self.get_stations_by_group_code(group_code)

        dem_fname = f'{group_code}_{DEM_source}_3005_res1.tif'
        self.grid, self.dem = self.load_raster(dem_fname)


    def ensure_dir(self, file_path):
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)


    def set_mapper_dict(self):
        # create a dictionary where the key: value pairs 
        # map stations to the regional group name
        stn_mapper_path = os.path.join(BASE_DIR, 'processed_data/')
        mapper_dict_file = 'station_to_region_mapper.json'
        mapper_path = os.path.join(stn_mapper_path, mapper_dict_file)
        if not os.path.exists(mapper_path):
            print(mapper_path)
            raise Exception; '  Mapper file not found.  You need to first run process_complete_basin_groups.py'

        # with open(stn_mapper_path + mapper_dict_file, 'rb') as handle:
        #     code_dict = np.load(handle)
        json_filepath = stn_mapper_path + mapper_dict_file
        with open(json_filepath, 'r') as json_file:
            json_str = json.load(json_file)
            return json.loads(json_str)


    def retrieve_station_data(self):
        hysets_data_path = os.path.join(BASE_DIR, 'source_data/HYSETS_data/')
        hysets_df = pd.read_csv(hysets_data_path + 'HYSETS_watershed_properties.txt', sep=';')

        # USGS_stn_locs_path = hysets_data_path + 'USGS_station_locations/'
        # usgs_df = gpd.read_file(USGS_stn_locs_path, layer='USGS_Streamgages-NHD_Locations')

        # usgs_df = usgs_df.to_crs(self.crs)
        # usgs_df['Official_ID'] = usgs_df['SITE_NO']

        # wsc_path = os.path.join(BASE_DIR, 'source_data/WSC_data/WSC_basin_polygons')
        # wsc_stns = os.listdir(wsc_path)

        hysets_df = pd.read_csv(os.path.join(BASE_DIR, 'source_data/HYSETS_data/HYSETS_watershed_properties.txt'), sep=';')

        hysets_locs = [Point(x, y) for x, y in zip(hysets_df['Centroid_Lon_deg_E'].values, hysets_df['Centroid_Lat_deg_N'])]
        hysets_df = gpd.GeoDataFrame(hysets_df.copy(), geometry=hysets_locs, crs='EPSG:4269')
        hysets_df = hysets_df.to_crs(self.crs)

        self.hysets_df = hysets_df.copy()
        # self.wsc_stns = wsc_stns.copy()
        # self.usgs_df = usgs_df.copy()

        # load the SEAK basins
        # SEAK_file = 'USFS_Southeast_Alaska_Drainage_Basin__SEAKDB__Watersheds.geojson'
        # seak_path = os.path.join(BASE_DIR, f'source_data/AK_Coastline/{SEAK_file}')
        # seak_gdf = gpd.read_file(seak_path)
        # print(seak_gdf)
        # print(asdf)


    def get_baseline_drainage_area(self, stn):
        """Retrieve the published drainage area for a HYSETS station.

        Args:
            stn (str): Official ID of a streamflow monitoring station.

        Returns:
            float: drainage area in km^2 
        """
        data = self.hysets_df[self.hysets_df['Official_ID'] == stn].copy()
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


    def get_hysets_loc(self, stn):
        # use the baseline station location
        location = self.hysets_df[self.hysets_df['Official_ID'] == stn]
        if location.empty:
            print(f'     {stn} location not found in HYSETS.')
        hysets_loc = location.geometry.values[0]
        return (hysets_loc.x, hysets_loc.y)


    def check_if_file_exists(self, file):
        path = os.path.join(self.processed_dem_dir, file)

        if not os.path.exists(path):
            print(f' Path does not exist: {path}')
            raise Exception
        else:
            return path


    def load_raster(self, file):
        path = self.check_if_file_exists(file)
        grid = Grid.from_raster(path, dtype=np.uint8)
        data = grid.read_raster(path)
        print(f'   ...File loaded: {file}')
                
        self.raster_res = data.dy_dx[::-1]
        self.affine = data.affine
        return grid, data

    
    def preprocess_dem(self):
        filled = self.grid.fill_depressions(self.dem)
        inflated = self.grid.resolve_flats(filled, max_iter=1E12, eps=1E-12)
        self.fdir = self.grid.flowdir(inflated)
        self.acc = self.grid.accumulation(self.fdir)
        del filled
        del inflated

    def load_acc(self, group_code):
        file = f'{group_code}_{self.DEM_source}_3005_{self.resoluton_code}_acc.tif'
        path = os.path.join(self.processed_dem_dir, file)

        if not os.path.exists(path):
            print(f' Path does not exist: {path}')
            raise Exception

        region_acc_path = os.path.join(file, path)

        grid = Grid.from_raster(region_acc_path)
        acc = grid.read_raster(region_acc_path)
        print('   ...Flow accumulation loaded.')

        
        self.grid = grid
        self.acc = acc
    

    def get_acc_threshold(self, min_area_m):
        # determine the number of cells to use as the threshold
        # for snapping the pour point. 
        return int(min_area_m / (abs(self.raster_res[0] * self.raster_res[1])))

    
    def get_stations_by_group_code(self, group_code):
        all_stations = list(set(self.code_dict.keys()))
        return sorted([s for s in all_stations if self.code_dict[s] == group_code])


    def affine_map_vec_numba(self, affine, x, y):
        a, b, c, d, e, f, _, _, _ = affine
        n = x.size
        new_x = np.zeros(n, dtype=np.float32)
        new_y = np.zeros(n, dtype=np.float32)
        for i in range(n):
            new_x[i] = x[i] * a + y[i] * b + c
            new_y[i] = x[i] * d + y[i] * e + f
        return new_x, new_y


    def get_nearest_point(self, stn_loc, area_threshold, threshold_cells):
        # mask for just the cells in the expected range
        t0 = time.time()
        max_cells = int((1 + area_threshold) * threshold_cells)
        min_cells = int((1 - area_threshold) * threshold_cells)

        yi, xi = np.where((self.acc >= min_cells) & (self.acc <= max_cells))

        affine_tup = tuple(self.affine)
        
        # convert to coordinates
        x, y = self.affine_map_vec_numba(affine_tup, xi, yi)
        t1 = time.time()
        
        # print('how many returned from affine map?')
        # print(len(x))

        if len(x) == 0:
            print('   No point found.')
            print('   ')
            return False, (None, None)
        
        # convert to array of tuples
        mask_coords = np.c_[x, y]
        # print('how many mask coords returned?')
        # print(len(mask_coords))
        # calculate distances from station to flow accumulation points
        stn_coords = (stn_loc[0], stn_loc[1])

        diffs = np.array([np.subtract(stn_coords, c) for c in mask_coords])
        
        dists = [np.linalg.norm(d, ord=2) for d in diffs]
        
        min_xy = mask_coords[np.argmin(dists)]

        min_dist = np.linalg.norm(np.subtract(min_xy, stn_coords))
        print(f'    Min. distance from pour point to threshold flow acc cell = {min_dist:.0f} m')

        return True, min_xy

    
    def get_cell_index(self, pt):
        x, y = pt.x, pt.y
        return self.grid.nearest_cell
        


    def find_pour_point(self, stn):
        stn_loc = self.get_hysets_loc(stn)
        expected_area_m = self.get_baseline_drainage_area(stn) * 1E6
        # Snap pour point to threshold accumulation cell
        distance = 0
        min_threshold = 1E6
        x, y = stn_loc[0], stn_loc[1]
        x_snap, y_snap = x, y
        # try:
            # as a first try, use a minimum of 1km^2 accumulation threshold
        d1 = False
        self.threshold_cells = self.get_acc_threshold(min_threshold)
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


        original_distance = distance
        threshold_cells = self.get_acc_threshold(expected_area_m)
        nr_found, (x_nr, y_nr) = self.get_nearest_point(stn_loc, area_thresh, threshold_cells)
        nn = 0
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
                    print(f'   new area threshold: {area_thresh}')
                else:
                    area_thresh -= 0.04
            except Exception as e:
                print(f'Error delineating basin for {self.stn}:')
                print(e)
                print(asdf)
                # print(f'   Area threshold range failed.')
                break      

        return Point(x_snap, y_snap)


    def fill_holes(self, data):
        geoms = data.explode(index_parts=False)            
        interior_holes = geoms.interiors.dropna().tolist()
        interior_holes = [e for sublist in interior_holes for e in sublist]
        gap_list = []
        if len(interior_holes) > 0:
            # print(f'   ...{len(interior_holes)} holes found to be filled.')
            for hole in interior_holes:
                gap_list.append(Polygon(hole))

            data_gaps = gpd.GeoDataFrame(geometry=gap_list, crs=data.crs)
            # appended_set = pd.concat([data, data_gaps])
            appended_set = data.append(data_gaps)
            appended_set.loc[:, 'group'] = 0
            merged_polygon = appended_set.dissolve(by='group', aggfunc='sum')
            # print(f'     merged geometry has {len(merged_polygon)} polygon (s)')
            if len(merged_polygon) == 1:
                return merged_polygon.geometry.values[0]
            else:
                raise Exception; 'Unmerged polygons in fill_holes()'
        else:
            if len(data) == 1:
                return data.geometry.values[0]
            else:
                raise Exception; "Original data needs to be dissolved into single polygon"


    def derive_basin(self, pour_point):

        # Delineate the catchment
        catch = self.grid.catchment(x=pour_point.x, y=pour_point.y, fdir=self.fdir)

        # Create view
        catch_view = self.grid.view(catch, dtype=np.uint8)

        # # Create a vector representation of the catchment mask
        mask = catch_view > 0
        polygons = rio.features.shapes(catch_view, mask=mask, transform=self.affine)  
        geoms = [shapely.geometry.shape(shp) for shp, _ in polygons]    
        basin_polygon = shapely.ops.unary_union(geoms)

        if basin_polygon.is_empty:
            print(f'    ...basin delineation failed.')
            return None
        else:
            gdf = gpd.GeoDataFrame(geometry=[basin_polygon], crs='EPSG:3005')
            gdf['geometry'] = self.fill_holes(gdf)
            
        if gdf is not None:
            # print(f'    ...basin polygon created.')
            return gdf
        else: 
            return None


