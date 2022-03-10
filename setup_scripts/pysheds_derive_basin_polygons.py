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

import warnings
warnings.filterwarnings('ignore')

t0 = time.time()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'source_data/')

# data_dir = '/media/danbot/Samsung_T5/geospatial_data/'
dem_dir = DATA_DIR + 'dem_data/'
processed_dem_dir = dem_dir + 'processed_dem/'

# specify the DEM source
# either 'EarthEnv_DEM90' or 'USGS_3DEP'
DEM_source = 'EarthEnv_DEM90'

# t7_media_path = '/media/danbot/T7 Touch/thesis_data/processed_stations/'


output_basin_polygon_path = BASE_DIR + '/processed_data/processed_basin_polygons/'

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
mapper_dict_file = 'station_to_region_mapper.pickle'

if not os.path.exists(os.path.join(stn_mapper_path, mapper_dict_file)):
    raise Exception; '  Mapper file doesnt exist.  You need to first run process_hydrologic_regions.py'
    
with open(stn_mapper_path + mapper_dict_file, 'rb') as handle:
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
        appended_set = pd.concat([data, data_gaps])
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


def pysheds_delineation(grid, fdir, acc, station, threshold):

    if station not in os.listdir(wsc_path):
        print(f'    {station} not found in WSC directory.  Revert to HYSETS location.')
        location = hysets_df[hysets_df['Official_ID'] == station]
        sp_pt = location.geometry.values[0]
        x, y = sp_pt.x, sp_pt.y
    else:
        # retrieve the pour point
        print(f'   {station} found in WSC directory, using published pour point.')
        pp_loc = os.path.join(wsc_path, f'{station}/PourPoint/')
        layer_name = f'{station}_PourPoint_PointExutoire'
        pp_gdf = gpd.read_file(pp_loc, layer=layer_name)
        pp_gdf = pp_gdf.to_crs(3005)
        ppt = pp_gdf.geometry.values[0]
        x, y = ppt.x, ppt.y

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Snap pour point to high accumulation cell
    x_snap, y_snap = grid.snap_to_mask(acc > threshold, (x, y))

    distance = np.sqrt((x_snap - x)**2 + (y_snap - y)**2)

    print(f'   ...difference btwn. nearest and PYSHEDS snap = {distance:1f}')
    snapped_loc = Point(x_snap, y_snap)
    snapped_loc_gdf = gpd.GeoDataFrame(geometry=[snapped_loc], crs=hysets_df.crs)
    snapped_loc_gdf.to_file(os.path.join(stn_mapper_path, 'snapped_locations'))

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
    

dir_method = 'D8' # D8, DINF
delineation_method = 'PYSHEDS'
# for region in code
region_codes = sorted(list(set(code_dict.values())))

bad_basins = []
i = 0

dem_files = os.listdir(processed_dem_dir)

resolutions = sorted(list(set([e.split('.')[0].split('_')[-1] for e in dem_files])))[::-1]

print(f'  The following DEM resolutions were found and will be used to process basins: {resolutions}')

for resolution in resolutions:
    for region_code in region_codes:
        # get the covering region for the station
        t_start = time.time()

        print(f'Starting analysis on region {region_code} {i}/{len(region_codes)}.')
        
        # load the region DEM once and iterate through all
        region_dem_path = os.path.join(processed_dem_dir, f'{region_code}_{DEM_source}_3005_{resolution}.tif')

        # foo = '/media/danbot/Samsung_T5/geospatial_data/DEM_data/processed_dem'
        # region_dem_path = os.path.join(foo, f'{region_code}_DEM_3005_{resolution}.tif')
        assert os.path.exists(region_dem_path)

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
        for station in stations:
            if j % 10 == 0:
                print(f'   Deriving basin for {station} {j}/{len(stations)}')
            # stations in the region to trim the number of redundant file loadings
            ensure_dir(output_basin_polygon_path)
            basin_out_path = output_basin_polygon_path + f'{station}_{delineation_method}_basin_derived_{resolution}.geojson'

            if not os.path.exists(basin_out_path):

                # snap_point_path = f'/media/danbot/Samsung_T5/geospatial_data/WSC_data/reprojected_data/{station}/{station}_pour_point.geojson'

                catch = pysheds_delineation(grid, fdir, acc, station, acc_threshold)
                
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