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

# t7_media_path = '/media/danbot/T7 Touch/thesis_data/processed_stations/'

basin_polygons_path = BASE_DIR + '/processed_data/merged_basin_groups/final_polygons/'

output_basin_polygon_path = BASE_DIR + '/processed_data/processed_basin_polygons/'

hysets_data_path = os.path.join(BASE_DIR, 'source_data/HYSETS_data/')
hysets_df = pd.read_csv(hysets_data_path + '/HYSETS_watershed_properties.txt', sep=';')

USGS_stn_locs_path = hysets_data_path + '/USGS_station_locations/'
usgs_df = gpd.read_file(USGS_stn_locs_path, layer='USGS_Streamgages-NHD_Locations')

usgs_df = usgs_df.to_crs(3005)
usgs_df['Official_ID'] = usgs_df['SITE_NO']

# wsc_path = os.path.join(BASE_DIR, 'source_data/WSC_data')
# wsc_pp_geom_base = wsc_path + 'all/'
# print(wsc_pp_geom_base)

wsc_df = pd.read_csv(os.path.join(BASE_DIR, 'source_data/WSC_Stations_2020.csv'))
wsc_locs = [Point(x, y) for x, y in zip(wsc_df['Longitude'].values, wsc_df['Latitude'])]
wsc_df = gpd.GeoDataFrame(wsc_df, geometry=wsc_locs, crs='EPSG:4269')
wsc_df = wsc_df.to_crs(3005)
wsc_stations = wsc_df['Station Number'].values
wsc_df['Official_ID'] = wsc_df['Station Number']


hysets_df = pd.read_csv(os.path.join(BASE_DIR, 'source_data/HYSETS_data/HYSETS_watershed_properties.txt'), sep=';')

hysets_locs = [Point(x, y) for x, y in zip(hysets_df['Centroid_Lon_deg_E'].values, hysets_df['Centroid_Lat_deg_N'])]
hysets_df = gpd.GeoDataFrame(hysets_df, geometry=hysets_locs, crs='EPSG:4269')
hysets_df = hysets_df.to_crs(3005)

combined_df = pd.concat([wsc_df[['Official_ID', 'geometry']], hysets_df[['Official_ID', 'geometry']]], join='outer', ignore_index=True)

# drop duplicates, but keep the first station polygon
combined_df.drop_duplicates(subset='Official_ID', keep='first', inplace=True, ignore_index=True)

stn_loc_df = pd.concat([wsc_df[['Official_ID', 'geometry']], usgs_df[['Official_ID', 'geometry']]], join='outer', ignore_index=True)

stn_loc_df = stn_loc_df[stn_loc_df['Official_ID'].isin(hysets_df['Official_ID'].values)]


def get_grp_polygon(basin_polygons_path, polygon_fnames, grp_code):
    grp_polygon_path = [e for e in polygon_fnames if grp_code in e][0]
    grp_polygon = gpd.read_file(basin_polygons_path + grp_polygon_path)
    grp_polygon = grp_polygon.to_crs(3005)
    grp_polygon['grp'] = grp_code
    grp_polygon = grp_polygon.dissolve(by='grp', aggfunc='sum')
    return grp_polygon

# get all the stations that fall within the study area
def map_stations_to_basin_groups(nhn_group_polygon_path):    

    # create a dict to organize stations by their location in the regional groupings
    region_dict = {}
    regional_group_polygon_fnames = os.listdir(nhn_group_polygon_path)
    code_dict = {}
    for fname in regional_group_polygon_fnames:
        grp_code = fname.split('_')[0]
        print(f'  Finding HYSETS stations within {grp_code}.')
        grp_polygon = get_grp_polygon(basin_polygons_path, regional_group_polygon_fnames, grp_code)

        intersecting_pts = gpd.sjoin(stn_loc_df, grp_polygon, how='inner', predicate='intersects')
        station_IDs = list(intersecting_pts['Official_ID'].values)
        region_dict[grp_code] = station_IDs
        # stations must also be in wsc basin list
        missing_stns = [e for e in station_IDs if e not in wsc_stations]
        if len(missing_stns) > 0:
            print(f'   {missing_stns} are included in the HYSETS database but are not found in the WSC station list.')

        for stn in station_IDs:
            if stn in code_dict.keys():
                print(f'  for some reason, {stn} is already in code_dict...')
                print(f'    under {grp_code}, code_dict[stn] = {code_dict[stn]}')
            code_dict[stn] = grp_code
            
    return code_dict

stn_mapper_path = os.path.join(BASE_DIR, 'processed_data/')
mapper_dict_file = 'station_to_region_mapper.pickle'
if not os.path.exists(stn_mapper_path):
    os.mkdir(stn_mapper_path)

existing_mappers = os.listdir(stn_mapper_path)
if mapper_dict_file not in existing_mappers:
    code_dict = map_stations_to_basin_groups(basin_polygons_path)
    with open(stn_mapper_path + mapper_dict_file, 'wb') as handle:
        pickle.dump(code_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
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

    location = stn_loc_df[stn_loc_df['Official_ID'] == station]

    sp_pt = location.geometry.values[0]
    x, y = sp_pt.x, sp_pt.y

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Snap pour point to high accumulation cell
    x_snap, y_snap = grid.snap_to_mask(acc > threshold, (x, y))

    distance = np.sqrt((x_snap - x)**2 + (y_snap - y)**2)

    print(f'   ...difference btwn. nearest and PYSHEDS snap = {distance:1f}')
    snapped_loc = Point(x_snap, y_snap)
    snapped_loc_gdf = gpd.GeoDataFrame(geometry=[snapped_loc], crs=stn_loc_df.crs)
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

for resolution in ['low']:#resolutions:
    for region_code in region_codes:
        # get the covering region for the station
        t_start = time.time()

        print(f'Starting analysis on region {region_code} {i}/{len(region_codes)}.')
        
        # load the region DEM once and iterate through all
        region_dem_path = os.path.join(processed_dem_dir, f'{region_code}_DEM_3005_{resolution}.tif')

        foo = '/media/danbot/Samsung_T5/geospatial_data/DEM_data/processed_dem'
        region_dem_path = os.path.join(foo, f'{region_code}_DEM_3005_{resolution}.tif')
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