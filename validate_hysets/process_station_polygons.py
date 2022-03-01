import os
import time
import pickle

import pandas as pd
import numpy as np

import shapely
from shapely.geometry import Point, Polygon
from shapely import ops

import geopandas as gpd

import xarray as xr
import rioxarray as rxr
import rasterio as rio


import warnings
warnings.filterwarnings('ignore')

from whitebox.whitebox_tools import WhiteboxTools

wbt = WhiteboxTools()

data_dir = '/media/danbot/Samsung_T5/geospatial_data/'
hysets_dir = os.path.join(data_dir, 'HYSETS_data/')
# hydat_dir = os.path.join(data_dir, 'hydat_db/')
dem_dir = os.path.join(data_dir, 'DEM_data/')
processed_dem_dir = os.path.join(dem_dir, 'processed_dem/')
# # dem_fpath_1km = os.path.join(dem_dir, 'watershed_group_tiles_merged_1km/')
# # dem_fpath_10km = os.path.join(dem_dir, 'watershed_group_tiles_merged_10km/')

# # root_path = '/home/danbot/Documents/code/thesis_code/validate_hysets/'
# # updated basins from December 2021 (dave Hutchinson) have path formats like: 
# # WSC_data/2021-12-basins/all/07AA001/basin/07AA001_DrainageBasin_BassinDeDrainage.shp
wsc_basin_dir = data_dir + 'WSC_data/2021-12-basins/all/'

# # other data sources
# glhymps_fpath = data_dir + 'GLHYMPS/GLHYMPS.gdb'

# # snow and land use / land cover
# nalcms_fpath = data_dir + 'NALCMS_NA_2010/NA_NALCMS_2010_v2_land_cover_30m/' + 'NA_NALCMS_2010_v2_land_cover_30m.tif'

# # where to save results of validation
processed_raster_output_path = '/media/danbot/T7 Touch/thesis_data/processed_stations/'
processed_polygon_output_path = '/home/danbot/Documents/code/hysets_validation/processed_data/derived_basins/'

with open('20220211_code_dict.pickle', 'rb') as handle:
    code_dict = pickle.load(handle)


# file containing derived watershed properties used in hysets
hysets_props_fname = 'HYSETS_watershed_properties.txt'
# import the set of derived watershed properties from hysets
hysets_props = pd.read_csv(hysets_dir + hysets_props_fname, delimiter=';')
hy_geom = [shapely.geometry.Point(e['Centroid_Lon_deg_E'], e['Centroid_Lat_deg_N']) for _, e in hysets_props.iterrows()]
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

# find stations in common between HYSETS and WSC within the study region
all_stations = list(code_dict.keys())
print(f'There are {len(all_stations)} in HYSETS falling within the study region')
# common_stations = [stn for stn in all_stations if stn in wsc_stations]
# print(f'   and {len(common_stations)} of these are also in common with the WSC polygon set.')


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def wb_callback(msg):
    print(msg)
    

def get_nearest_point(stn, grp_code, loc, resolution):
    
    fpath = processed_dem_dir + f'{grp_code}_STREAMS_D8_FILL_3005_{resolution}.tif'
    flow_acc = rxr.open_rasterio(fpath, masked=True)
    
    # convert to BC albers for distance calc
    acc = flow_acc.rio.reproject('EPSG:3005')
    temp_loc = loc.to_crs(3005)

    # calculate distances from station to flow accumulation points
    dists = np.sqrt((temp_loc.geometry.x.values[0]-acc.x)**2 + (temp_loc.geometry.y.values[0]-acc.y)**2)
    
    acc.name = 'stream'
    dists.name = 'distances'
    ds = xr.merge([dists, acc])
    
    stream = ds.where(~np.isnan(ds.stream))
    
    min_dist = stream.distances.min().item()
    
    # print(f'   min distance = {min_dist:.0f}m')

    dists = stream.where(stream.distances == min_dist, drop=True)
    
    if dists.distances.size == 0:        
        print('   ')
        print('   ')
        raise Exception('No minimum point found')
    else:
        return Point(dists.x, dists.y), dists.rio.crs
    

def set_snapped_point(stn, snap_point_path, grp_code, stn_loc, resolution):
    if os.path.exists(snap_point_path):
        gdf = gpd.read_file(snap_point_path)
        crs = gdf.crs
        # print(f'    snap point path exists, crs={crs}')
    else:
        pt, crs = get_nearest_point(stn, grp_code, stn_loc, resolution)
        gdf = gpd.GeoDataFrame(geometry=[pt], crs=crs)
        print(f'    new snap point created.  crs={crs}')
        gdf.to_file(snap_point_path)
        
    return gdf
    

def derive_watershed_raster(stn, resolution='low'):
    t0 = time.time()
    # output filename
    basin_out_path = processed_raster_output_path + f'{stn}/{stn}_basin_derived_{resolution}.tif'
    if not os.path.exists(basin_out_path):
        hysets_basin_info = hysets_basins[hysets_basins['OfficialID'] == stn]
        hysets_basin_da = hysets_basin_info['Area'].values[0]
        
        hysets_stn_properties = hysets_gdf[hysets_gdf['Official_ID'] == stn]
        hysets_stn_properties = hysets_stn_properties.to_crs(3005)
        hysets_stn_loc = hysets_stn_properties.geometry.values[0]
        
        # WSC_data/2021-12-basins/all/07AA001/basin/07AA001_DrainageBasin_BassinDeDrainage.shp
        wsc_basin_fpath = wsc_basin_dir + f'{stn}/basin/'
        wsc_basin = gpd.read_file(wsc_basin_fpath)
        wsc_basin = wsc_basin.to_crs(3005)
        new_wsc_basin_area = wsc_basin.geometry.area.values[0] / 1E6
        assert wsc_basin.crs == hysets_basin_info.geometry.crs
        
        wsc_stn_loc_fpath = wsc_basin_dir + f'{stn}/pour_point/'
        wsc_stn_loc = gpd.read_file(wsc_stn_loc_fpath)
        wsc_stn_loc = wsc_stn_loc.to_crs(3005)
        
        
        grp_code = code_dict[stn]
        
        loc_distance = wsc_stn_loc.geometry.distance(hysets_stn_loc).values[0] / 1E3
            
        info_string = f'{stn}: bound by {grp_code}: WSC DA = {new_wsc_basin_area:.1f}km^2, HYSETS DA= {hysets_basin_da:.1f}km^2.'
        info_string += f'  {loc_distance:.1f}km btwn. reported locations.'
        
        # make sure processed data folder exists
        ensure_dir(processed_raster_output_path + f'{stn}/')
        
        # dem, d8 flow direction, and pour point paths
        dem_path = processed_dem_dir + f'{grp_code}_DEM_3005_{resolution}.tif'
        d8_pointer_path = processed_dem_dir + f'{grp_code}_DIR_3005_{resolution}.tif'
        xarray_snapped_pp_fpath = processed_raster_output_path + f'{stn}/{stn}_xarraysnapped_pour_point.shp'
            
        # need to define a pour point in order for wbt to derive watershed
        set_snapped_point(stn, xarray_snapped_pp_fpath, grp_code, wsc_stn_loc, resolution)
            
        assert os.path.exists(dem_path)
        assert os.path.exists(d8_pointer_path)
        assert os.path.exists(xarray_snapped_pp_fpath)


        # print(f'   ... delineating basin for {stn}')
        wbt.watershed(
            d8_pntr=d8_pointer_path, 
            pour_pts=xarray_snapped_pp_fpath, 
            output=basin_out_path, 
            esri_pntr=False, 
        )        
    
    t1 = time.time()
    # print(f'   processed {stn} in {t1-t0:.1f}s')
    return basin_out_path


def retrieve_raster(fpath):
    rds = rxr.open_rasterio(fpath)
    crs = rds.rio.crs
    affine = rds.rio.transform(recalc=False)
    return rds, crs, affine


def watershed_raster_to_polygon(stn, raster_path):
    rds, crs, affine = retrieve_raster(raster_path)
    mask = rds.data == 1    
    results = rio.features.shapes(rds.data, mask=mask, transform=affine)

    geoms = [shapely.geometry.shape(shp) for shp, value in list(results)]
    
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)

    if len(gdf) > 1:
        # assert 0 == 1
        gdf['station'] = stn
        gdf = gdf.dissolve(by='station', aggfunc='sum')

    basin_area = gdf.geometry.area / 1E6
    gdf['area'] = basin_area

    return gdf


resolution = 'hi'  # low, med, hi
i = 0
for stn in all_stations[1:]:
    if i % 10 == 0:
        print(f'   ...starting {stn} {i}/{len(all_stations)}')
        
    polygon_out_path = processed_polygon_output_path + f'{stn}_basin_{resolution}.geojson'
    raster_path = processed_raster_output_path + f'{stn}/{stn}_basin_derived_{resolution}.tif'
    if not os.path.exists(polygon_out_path):
        basin_raster_path = derive_watershed_raster(stn, resolution)
    
        gdf = watershed_raster_to_polygon(stn, raster_path)
        if not gdf.empty:
            gdf.to_file(polygon_out_path, driver='GeoJSON')
        else:
            print(f'{stn} returned an empty polygon.')
    
        if os.path.exists(polygon_out_path):
            os.remove(raster_path)
    
    i += 1
    # print(adsfsad)