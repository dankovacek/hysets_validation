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

import fiona

from pysheds.grid import Grid

from whitebox.whitebox_tools import WhiteboxTools

wbt = WhiteboxTools()
wbt.verbose = False

t0 = time.time()

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

source_data_dir = os.path.join(base_dir, 'source_data/')
processed_data_dir = os.path.join(base_dir, 'processed_data/')

data_dir = '/media/danbot/Samsung_T5/geospatial_data/'
dem_dir = data_dir + 'DEM_data/'

t7_media_path = '/media/danbot/T7 Touch/thesis_data/processed_stations/'

code_dict_path = base_dir + '/validate_hysets/20220211_code_dict.pickle'
with open(code_dict_path, 'rb') as handle:
    code_dict = pickle.load(handle)


# bc_regional_basins_file = processed_data_dir + 'merged_basin_groups/BC_basin_region_groups_EPSG4326.geojson'

# mapped_groups = gpd.read_file(bc_regional_basins_file)

# wsc_dir = data_dir + 'WSC_data/WSC_Catchments/'
# # other data sources

# bc_regional_basins = gpd.read_file(bc_regional_basins_file)
# basins_crs = bc_regional_basins.crs

# dem_mosaic_file = dem_dir + 'BC_DEM_mosaic_4326.vrt'

processed_dem_path = dem_dir + 'processed_dem/'

# wsc_basins = gpd.read_file(wsc_dir + 'HYDZ_HYD_WATERSHED_BND_POLY.geojson')
# wsc_basins = wsc_basins.to_crs(3005)

hysets_dir = data_dir + 'HYSETS_data/'
# file containing derived watershed properties used in hysets
hysets_props_fname = 'HYSETS_watershed_properties.txt'
# import the set of derived watershed properties from hysets
hysets_props = pd.read_csv(hysets_dir + hysets_props_fname, delimiter=';')
# create a dataframe of station locations from hysets
hy_geom = [Point(e['Centroid_Lon_deg_E'], e['Centroid_Lat_deg_N']) for _, e in hysets_props.iterrows()]
hysets_gdf = gpd.GeoDataFrame(hysets_props, 
                              geometry=hy_geom)
# convert to projected CRS NOTE--USE BC ALBERS: EPSG 3005
hysets_gdf = hysets_gdf.set_crs('EPSG:4326')
# hysets_gdf = hysets_gdf.to_crs('EPSG:3857')
hysets_gdf = hysets_gdf.to_crs('EPSG:3005')


hysets_basins = gpd.read_file(hysets_dir + 'HYSETS_watershed_boundaries.zip')
hysets_basins = hysets_basins.set_crs('EPSG:4326')
hysets_basins = hysets_basins.to_crs('EPSG:3005')
hysets_basins.head()

regional_groups = gpd.read_file(processed_data_dir + 'merged_basin_groups/BC_basin_region_groups_EPSG4326.geojson')
regional_groups = regional_groups.to_crs(hysets_basins.crs)


# get all hysets basins that are in the BC set
target_stations = gpd.sjoin(hysets_basins, regional_groups, how='inner', predicate='intersects')
target_stations.reset_index(inplace=True)
print(f'   {len(target_stations)} basins intersect with the study region.')

target_stations = target_stations.sort_values('Area')
target_station_IDs = target_stations['OfficialID'].values

source_data_mapping_fname =  processed_data_dir + 'catchment_validation_mapping.pickle'

# basin_dict_path = processed_data_dir + f'basin_network_dict.pickle'
# with open(source_data_mapping_fname, 'rb') as handle:
#         stn_data_mapping_dict = pickle.load(handle)
# print(stn_data_mapping_dict)
# print(asfds)
# with open(basin_dict_path, 'rb') as handle:
#         stn_data_mapping_dict = pickle.load(handle)


# def create_station_mapping_dict(stations):
#     stn_data = {}
#     excluded_stns = []
#     for stn in stations:
#         # get station coordinates
#         loc = hysets_gdf.loc[hysets_gdf['Official_ID'] == stn, 'geometry']

#         # check that crs are the same between data sources before proceeding
#         assert loc.crs == mapped_groups.crs, 'Object crs are not the same.'

#         # find the basin group that contains the location of interest
#         # the basin group will be used to reference the correct DEM and processed spatial files
#         # according to the station / basin location
#         bounding_polygon = mapped_groups.loc[mapped_groups.geometry.contains(loc.values[0]), 'WSCSSDA']
#         if bounding_polygon.empty:
#             print(f'  {stn} has no bounding polygon {loc.values[0]}')
#             excluded_stns.append(stn)
#         else:
#             grp_code = bounding_polygon.values[0]
#             stn_data[stn] = {
#                 'loc': loc,
#                 'code': grp_code,
#             }

#     with open('../processed_data/excluded_basins.pickle', 'wb') as handle:
#         pickle.dump(excluded_stns, handle)
    
#     return stn_data

# if not os.path.exists(source_data_mapping_fname):
#     stn_data = create_station_mapping_dict(bc_station_IDs)
#     with open(source_data_mapping_fname, 'wb') as handle:
#         pickle.dump(stn_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         stn_data_mapping_dict = stn_data
# else:

with open(source_data_mapping_fname, 'rb') as handle:
    stn_data_mapping_dict = pickle.load(handle)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_nearest_point(stn, stn_loc, stream_raster):
    # grp_code = stn_data_mapping_dict[stn]['code']
    # loc = stn_data_mapping_dict[stn]['loc']
    
    # convert to BC albers for distance calc
    if stream_raster.rio.crs.to_epsg() != 3005:
        stream_raster = stream_raster.rio.reproject('EPSG:3005')

    assert stn_loc.crs == 'epsg:3005'

    stn_loc = stn_loc.geometry.values[0]
    
    # calculate distances from station to flow accumulation points
    dists = np.sqrt((stn_loc.x-stream_raster.x)**2 + (stn_loc.y-stream_raster.y)**2)
    
    stream_raster.name = 'stream'
    dists.name = 'distances'
    ds = xr.merge([dists, stream_raster])

    # check that 0 values (non stream but inside basin) 
    # are set to nan before delineating basin
    # ds = ds.where(ds.stream == 0, np.nan)
    # one_count = ds.distances.where(ds.distances == 1).count()
    # nil_count = ds.distances.where(ds.distances == 0).count()
    # nan_count = ds.distances.where(ds.distances == np.nan).count()
    # tot_count = ds.sizes
    min_distance = ds.distances.min()
    min_distance_loc = ds.where(ds.distances == min_distance, drop=True).squeeze()
    
    if min_distance.size == 0:        
        print('   ')
        print('   ')
        raise Exception('No minimum point found for {}')
    else:
        print(f'    Minimum distance to stream network is {min_distance.item():.1f}m')
        snapped_point = Point(min_distance_loc.x.item(), min_distance_loc.y.item())
        return snapped_point
    

def set_snapped_point(stn, stn_loc, stream_raster, resolution):
    snapped_point_path = t7_media_path + f'{stn}/{stn}_snapped_pour_point_{resolution}.shp'
    if os.path.exists(snapped_point_path):
        gdf = gpd.read_file(snapped_point_path)
    else:
        pt = get_nearest_point(stn, stn_loc, stream_raster)
        gdf = gpd.GeoDataFrame(geometry=[pt], crs=stn_loc.crs)
        print(f'    new snap point created.  crs={stn_loc.crs}')
        gdf.to_file(snapped_point_path)
        
    return snapped_point_path


# now create polygons using the raster just generated
def retrieve_raster(fpath):
    rds = rxr.open_rasterio(fpath)
    crs = rds.rio.crs
    affine = rds.rio.transform(recalc=False)
    return rds, crs, affine


def watershed_raster_to_polygon(stn, resolution):    
    fpath = t7_media_path + f'{stn}/{stn}_basin_derived_{resolution}.tif'
    rds, crs, affine = retrieve_raster(fpath)
    mask = rds.data > 0    
    polygons = rio.features.shapes(rds.data, mask=mask, transform=affine)  
    geoms = [shapely.geometry.shape(shp) for shp, _ in polygons]    
    catchment_polygon = shapely.ops.unary_union(geoms)
    return catchment_polygon, fpath, crs


def get_or_derive_shape_polygon(stn, resolution, delineation_method):
    polygon_out_path = t7_media_path + f'{stn}/{stn}_WBT_derived_catchment_{resolution}.shp'
    # if not os.path.exists(polygon_out_path):
    if delineation_method == 'WBT':
        basin_polygon, raster_fpath, crs = watershed_raster_to_polygon(stn, resolution)

        if basin_polygon.is_empty:
            print(f'    ...basin polygon retrieval failed for {stn}')
            return False
        else:
            gdf = gpd.GeoDataFrame(geometry=[basin_polygon], crs=crs)
            gdf.to_file(polygon_out_path)
            os.remove(raster_fpath)
            return True


def pysheds_delineation(dir_raster, snap_point_path, acc_grid, threshold):

    fname = snap_point_path.split('/')[-1]
    layer = fname.split('.')[0]
    path = '/'.join([e for e in snap_point_path.split('/') if e != fname])

    sp = gpd.read_file(path, layer=layer)
    sp_pt = sp.geometry.values[0]
    x, y = sp_pt.x, sp_pt.y

    
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    # if 'D8' in out_path:
    #     dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    # elif 'DINF' in out_path:
    #     dirmap = None
    #     raise Exception; "Need to figure out and set dirmap for dinf."

    # Snap pour point to high accumulation cell
    # x_snap, y_snap = acc_grid.snap_to_mask(acc_grid > threshold, (x, y))

    # distance = np.sqrt((x_snap - x)**2 + (y_snap - y)**2)
    # print(f'   ...difference btwn. nearest and PYSHEDS snap = {distance:1f}'
    foo = f'{processed_dem_path}{region_code}_DIR_3005_{resolution}.tif'
    fdir_grid = Grid.from_raster(foo)
    fdir = fdir_grid.read_raster(foo)
    # fdir = retrieve_raster(f'{processed_dem_path}{region_code}_DIR_3005_{resolution}.tif')

    # Delineate the catchment
    catch = acc_grid.catchment(x=x, y=y, fdir=fdir, dirmap=dirmap, xytype='coordinate')
    
    return catch


def pysheds_basin_polygon(stn, acc_grid, catch, out_path):

    fname = out_path.split('/')[-1]
    # Create view
    catch_view = acc_grid.view(catch, dtype=np.uint8)

    # Create a vector representation of the catchment mask
    shapes = acc_grid.polygonize(catch_view)

    # Specify schema
    schema = {
            'geometry': 'Polygon',
            'properties': {'LABEL': 'float:16'}
    }

    # Write shapefile
    with fiona.open(out_path, 'w',
                    driver='ESRI Shapefile',
                    crs=acc_grid.crs.srs,
                    schema=schema) as c:
        i = 0
        for shape, value in shapes:
            rec = {}
            rec['geometry'] = shape
            # rec['nodata'] = dir_raster.rio.nodata,
            rec['properties'] = {'LABEL' : str(value)}
            rec['id'] = str(i)
            c.write(rec)
            i += 1
    print(f'   ..Completed {fname}')


def get_acc_threshold(res, min_area):
    return int(min_area / (abs(res[0] * res[1])))



def derive_watershed(stn, flow_direction_path, dir_raster, method, stream_raster, acc_grid, resolution, threshold):
    t0 = time.time()
       
    # make sure processed data folder exists
    ensure_dir(t7_media_path+ f'{stn}/')
    
    station_info_hysets = hysets_gdf[hysets_gdf['Official_ID'] == stn]
    hysets_stn_loc = station_info_hysets.geometry.values[0]
    stn_loc = gpd.GeoDataFrame(geometry=[hysets_stn_loc], crs=station_info_hysets.crs)
    stn_loc = stn_loc.to_crs(3005)
        
    # need to define a pour point in order for wbt to derive watershed     
    snapped_pourpoint_path = set_snapped_point(stn, stn_loc, stream_raster, resolution)

    # output filename
    basin_out_path = t7_media_path + f'{stn}/{stn}_{method}_basin_derived_{resolution}.tif'

    if method == 'WBT':
        # if not os.path.exists(basin_out_path):
        wbt.watershed(
            d8_pntr=flow_direction_path,
            pour_pts=snapped_pourpoint_path, 
            output=basin_out_path, 
            esri_pntr=False, 
        )   
    elif method == 'PYSHEDS':
        pysheds_catch = pysheds_delineation(dir_raster, snapped_pourpoint_path, acc_grid, threshold)
        pysheds_basin_polygon(stn, acc_grid, pysheds_catch, basin_out_path)

    t1 = time.time()
    print(f'   processed {stn} in {t1-t0:.1f}s')




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
for resolution in ['low', 'med', 'hi']:
    for region_code in region_codes:
        # get the covering region for the station

        print(f'Starting analysis on {region_code} region {i}/{len(region_codes)}.')
        # load the region DEM once and iterate through all
        region_dem_path = os.path.join(processed_dem_path, f'{region_code}_DEM_3005_{resolution}.tif')
        # source direction and pour point path
        flow_direction_path = f'{processed_dem_path}{region_code}_DIR_3005_{resolution}.tif'
        stream_path = processed_dem_path + f'{region_code}_STREAMS_{dir_method}_FILL_3005_{resolution}.tif'

        acc_path = processed_dem_path + f'{region_code}_WBT_ACC_{dir_method}_FILL_3005_{resolution}.tif'
        
        acc_raster, acc_crs, acc_affine = retrieve_raster(acc_path)
        acc_grid = Grid.from_raster(acc_path, window_crs=acc_crs, nodata=acc_raster.rio.nodata)

        raster_res = acc_raster.rio.resolution()
        # set default minimum area for accumulation to 1km^2
        acc_threshold = get_acc_threshold(raster_res, min_area=1E6)

        stream_raster, stream_crs, stream_affine = retrieve_raster(stream_path)
        dir_raster, dir_crs, dir_affine = retrieve_raster(flow_direction_path)
        dir_nodata = dir_raster.rio.nodata
        # direction_grid = Grid.from_raster(flow_direction_path, window_crs=dir_crs, nodata=dir_raster.rio.nodata)
        
        stations = sorted([s for s in list(set(code_dict.keys())) if code_dict[s] == region_code])

        j = 0
        for station in stations:
            print(f'   Deriving basin for {station} {j}/{len(stations)}')
            # stations in the region to trim the number of redundant file loadings
            derive_watershed(station, flow_direction_path, dir_raster, delineation_method, stream_raster, acc_grid, resolution, acc_threshold)
            print(asfsd)
            basin_created = get_or_derive_shape_polygon(station, resolution, delineation_method)
            if not basin_created:
                bad_basins.append(station)
                print(f'    ...basin creation failed for {station}.')
            else:
                print(f'   ...completed basin polygon creation for {station}')

print('The following basin delineations failed:')
print(bad_basins)