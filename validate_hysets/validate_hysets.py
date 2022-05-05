from distutils.log import error
import os
import time
import pickle

import pandas as pd
import numpy as np

import shapely
from shapely.geometry import Point, Polygon, box
from shapely import ops

import geopandas as gpd

import xarray as xr
import rioxarray as rxr
import rasterio as rio

import richdem as rd

# from whitebox.whitebox_tools import WhiteboxTools


import logging
logging.getLogger('richdem').setLevel(logging.ERROR)

# wbt = WhiteboxTools()

# cwd = os.getcwd()
# wbt.verbose = False
# wbt.set_working_dir(cwd)

import warnings
warnings.filterwarnings('ignore')

t0 = time.time()
print('Starting HYSETS validation script.  Loading resources...')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'source_data/')

ext_data_dir = '/media/danbot/Samsung_T5/geospatial_data/'
hysets_dir = os.path.join(DATA_DIR, 'HYSETS_data/')
hysets_dir = os.path.join(ext_data_dir, 'HYSETS_data')

hydat_dir = os.path.join(DATA_DIR, 'hydat_db/')
dem_dir = os.path.join(DATA_DIR, 'dem_data/')

processed_dem_dir = os.path.join(dem_dir, 'processed_dem/')

processed_data_dir = os.path.join(BASE_DIR, 'processed_data/')
# dem_fpath_1km = os.path.join(dem_dir, 'watershed_group_tiles_merged_1km/')
# dem_fpath_10km = os.path.join(dem_dir, 'watershed_group_tiles_merged_10km/')

validation_results_path = os.path.join(BASE_DIR, 'validate_hysets')
# root_path = '/home/danbot/Documents/code/thesis_code/validate_hysets/'
# updated basins from December 2021 (dave Hutchinson) have path formats like: 
# WSC_data/2021-12-basins/all/07AA001/basin/07AA001_DrainageBasin_BassinDeDrainage.shp
# wsc_basin_dir = data_dir + 'WSC_data/2021-12-basins/all/'

# compile the list of stations with a basin polygon
# wsc_stations = os.listdir(wsc_basin_dir)

# other data sources
glhymps_fpath = DATA_DIR + 'GLHYMPS_data/GLHYMPS.gdb'

# snow and land use / land cover
nalcms_fpath = ext_data_dir + 'NALCMS_NA_2010/NA_NALCMS_2010_v2_land_cover_30m/' + 'NA_NALCMS_2010_v2_land_cover_30m.tif'

# # where to save results of validation
# processed_data_output_path = '/media/danbot/T7 Touch/thesis_data/processed_stations/'
# # processed_polygons_out_path = '/home/danbot/Documents/code/hysets_validation/processed_data/derived_basins/'
# processed_polygon_path = '/media/danbot/T7 Touch/thesis_data/processed_stations/'

# wsc_df = pd.read_csv(hydat_dir + 'hydrometric_StationList_2021-07.csv')
# wsc_df.columns = [e.strip() for e in wsc_df.columns]

# wsc_bc = wsc_df[wsc_df['Prov/Terr'] == 'BC']
# wsc_bc.columns = [e.strip() for e in wsc_bc.columns]
# wsc_bc.columns

# t_wsc = time.time()
# print(f"    ..WSC data loaded in {t_wsc - t0:.1f}")

# file containing derived watershed properties used in hysets
hysets_props_fname = '/HYSETS_watershed_properties.txt'

# import the set of derived watershed properties from hysets
hysets_props = pd.read_csv(hysets_dir + hysets_props_fname, delimiter=';')

# create a dataframe of station locations from hysets
# hy_geom = [shapely.geometry.Point(e['Centroid_Lon_deg_E'], e['Centroid_Lat_deg_N']) for _, e in hysets_props.iterrows()]
# hysets_gdf = gpd.GeoDataFrame(hysets_props, 
                            #   geometry=hy_geom)
# convert to projected CRS NOTE--USE BC ALBERS: EPSG 3005
# hysets_gdf = hysets_gdf.set_crs('EPSG:4326')
# # hysets_gdf = hysets_gdf.to_crs('EPSG:3857')
# hysets_gdf = hysets_gdf.to_crs('EPSG:3005')

hysets_basins = gpd.read_file(hysets_dir + '/HYSETS_watershed_boundaries.zip')
hysets_basins = hysets_basins.set_crs('EPSG:4326')
hysets_basins = hysets_basins.to_crs('EPSG:3005')
hysets_basins.head()

t_hs = time.time()
print(f"    ....HYSETS data loaded in {t_hs - t0:.1f}")

def retrieve_raster(fpath):
    rds = rxr.open_rasterio(fpath, masked=True, mask_and_scale=True)
    crs = rds.rio.crs
    affine = rds.rio.transform(recalc=False)
    return rds, crs, affine


# load resources once!
nalcms_raster, nalcms_crs, nalcms_affine = retrieve_raster(nalcms_fpath)
t_nal = time.time()
print(f"    ......NALCMS data loaded in {t_nal - t_hs:.1f}")

# data_folder = processed_data_output_path
region_mapper_fpath = os.path.join(processed_data_dir, 'station_to_region_mapper.pickle')
with open(region_mapper_fpath, 'rb') as handle:
    code_dict = pickle.load(handle)

print('    Resources loaded.')


# find stations in common between HYSETS and WSC within the study region
processed_polygon_fpath = os.path.join(processed_data_dir, 'processed_basin_polygons/')
all_station_polygons = os.listdir(processed_polygon_fpath)
all_stations = list(set([e.split('_')[0] for e in all_station_polygons]))
print(f'    {len(all_stations)} stations to process.')

wsc_path = os.path.join(BASE_DIR, 'source_data/WSC_data/WSC_basin_polygons')
wsc_stns = os.listdir(wsc_path)
common_stations = [e for e in all_stations if e in wsc_stns]

print(f'    of which {len(common_stations)} are also in the WSC basin set.')

def calculate_gravelius_and_perim(polygon):
    
    p = polygon.to_crs('EPSG:3005')
    perimeter = p.geometry.length.values[0]
    area = p.geometry.area.values[0] 
    if area == 0:
        return np.nan, perimeter
    else:
        perimeter_equivalent_circle = np.sqrt(4 * np.pi * area)
        gravelius = perimeter / perimeter_equivalent_circle

    return gravelius, perimeter


def clip_raster_to_basin(station, basin_polygon, raster):
    
    crs = raster.rio.crs.to_epsg()
    if not crs:
        crs = raster.rio.crs.to_wkt()
    
    basin_polygon = basin_polygon.to_crs(crs)
    bounds = tuple(basin_polygon.bounds.values[0])

    # trimmed_box = box(*bounds).bounds
    try:
        subset_raster = raster.rio.clip_box(*bounds)
        # bounds_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / 1E6
        # poly_area = basin_polygon.geometry.area.values[0] / 1E6
        clipped_raster = subset_raster.rio.clip(basin_polygon.geometry, basin_polygon.crs, all_touched=True)
        return clipped_raster, True
    except Exception as e:
        print(e)
        return None, False
    

def process_basin_elevation(dem_raster):
    # evaluate masked raster data
    vals = dem_raster.data.flatten()    
    mean_val, min_val, max_val = np.nanmean(vals), np.nanmin(vals), np.nanmax(vals)
    return mean_val, min_val, max_val


def recategorize_lulc(data):    
    forest = ('Land_Use_Forest_frac', [1, 2, 3, 4, 5, 6])
    shrub = ('Land_Use_Shrubs_frac', [7, 8, 11])
    grass = ('Land_Use_Grass_frac', [9, 10, 12, 13])
    wetland = ('Land_Use_Wetland_frac', [14])
    crop = ('Land_Use_Crops_frac', [15])
    urban = ('Land_Use_Urban_frac', [16, 17])
    water = ('Land_Use_Water_frac', [18])
    snow_ice = ('Land_Use_Snow_Ice_frac', [19])
    lulc_dict = {}
    for label, p in [forest, shrub, grass, wetland, crop, urban, water, snow_ice]:
        prop_vals = round(sum([data[e] if e in data.keys() else 0 for e in p]), 2)
        lulc_dict[label] = [prop_vals]
    return lulc_dict


def get_value_proportions(data):
    # vals = data.data.flatten()
    all_vals = data.data.flatten()
    vals = all_vals[~np.isnan(all_vals)]
    n_pts = len(vals)
    unique, counts = np.unique(vals, return_counts=True)

    # create a dictionary of land cover values by coverage proportion
    prop_dict = {k: v/n_pts for k, v in zip(unique, counts)}
    prop_dict = recategorize_lulc(prop_dict)
    return prop_dict


def check_lulc_sum(stn, data):
    checksum = sum(list([e[0] for e in data.values()])) 
    # print(f'checksum = {checksum}')
    if abs(1-checksum) >= 0.02:
        print(f'   ...{stn} failed checksum: {checksum}')        


def process_lulc(stn, basin_polygon):
    print('lulc')
    clipped_raster, success = clip_raster_to_basin(stn, basin_polygon, nalcms_raster)
    if not success:
        return pd.DataFrame()
    # checksum verifies proportions sum to 1
    prop_dict = get_value_proportions(clipped_raster)
    check_lulc_sum(stn, prop_dict)
    return pd.DataFrame(prop_dict, index=[stn])


def get_perm_and_porosity(merged):
    merged['area_frac'] = merged['Shape_Area'] / merged['Shape_Area'].sum()
    weighted_permeability = round((merged['area_frac'] * merged['Permeability_no_permafrost']).sum(), 2)
    weighted_porosity = round((merged['area_frac'] * merged['Porosity']).sum(), 2)
    # print('permeability, porosity: ')
    # print(weighted_permeability, weighted_porosity)
    return weighted_permeability, weighted_porosity
    

def process_glhymps(basin_polygon, glhymps_fpath):
    # pulled from checking the crs proj4 string using fiona
    # not sure what the equivalent EPSG code is...
    proj = 'proj=cea +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    # convert  polygon crs to match GLHYMPS
    poly = basin_polygon.to_crs(proj)
    gdf = gpd.read_file(glhymps_fpath, mask=poly)
    
    # convert back to 3857 for plotting
    poly_gdf = gpd.GeoDataFrame(poly, geometry=[poly.geometry.values[0]], crs=proj)
    # glhymps_gdf = glhymps_gdf.to_crs('EPSG:3857')
    merged = gpd.overlay(gdf, poly_gdf, how='intersection')
    merged = merged.to_crs('EPSG:3857')
    return merged


def calculate_slope_and_aspect(clipped_raster):   
    
    wkt = clipped_raster.rio.crs.to_wkt()
    affine = clipped_raster.rio.transform()
    
    rdem_clipped = rd.rdarray(
        clipped_raster.data[0], 
        no_data=clipped_raster.rio.nodata, 
        # projection=wkt, 
    )
    rdem_clipped.geotransform = affine.to_gdal()
    rdem_clipped.projection = wkt

    slope = rd.TerrainAttribute(rdem_clipped, attrib='slope_degrees')
    aspect = rd.TerrainAttribute(rdem_clipped, attrib='aspect')
    # print('slope, aspect')
    # print(slope, aspect)
    return np.nanmean(slope), np.nanmean(aspect)


def retrieve_basin_polygon(basin_polygon_source, station, resolution):
    if basin_polygon_source == 'manual':
        # basin_polygon_path = processed_polygons_out_path + 'pysheds/' + f'{station}_PYSHEDS_basin_derived_{resolution}.geojson'
        basin_polygon_path = processed_polygon_fpath + f'{station}_PYSHEDS_basin_derived_{resolution}.geojson'
        if not os.path.exists(basin_polygon_path):
            raise Exception; f'{basin_polygon_path} does not exist.'
        else:
            basin_polygon = gpd.read_file(basin_polygon_path, driver='GeoJSON')
    else:
        basin_polygon = hysets_basins[hysets_basins['OfficialID'] == station]
        
    assert basin_polygon.crs == 'epsg:3005'
    
    return basin_polygon


def process_basin_characteristics(grp_code, station, dem_raster, basin_polygon):

    data = {}
    data['OfficialID'] = [station]
    data['group_code'] = [grp_code]
    
    mean_el, min_el, max_el = process_basin_elevation(dem_raster)

    data['Drainage_Area_km2'] = basin_polygon.geometry.area.values[0] / 1E6

    data['Elevation_m'] = [mean_el]
    data['min_el'] = [min_el]
    data['max_el'] = [max_el]

    gravelius, perimeter = calculate_gravelius_and_perim(basin_polygon)
    data['Perimeter'] = [perimeter]
    data['Gravelius'] = [gravelius]
        
    slope, aspect = calculate_slope_and_aspect(dem_raster)
    data['Slope_deg'] = [slope]
    data['Aspect_deg'] = [aspect]
    
    glhymps_df = process_glhymps(basin_polygon, glhymps_fpath)
    weighted_permeability, weighted_porosity = get_perm_and_porosity(glhymps_df)
    data['Permeability_logk_m2'] = [weighted_permeability]
    data['Porosity_frac'] = [weighted_porosity]
    
    lulc_df = process_lulc(station, basin_polygon)
    lulc_data = lulc_df.to_dict('records')[0]
    data.update(lulc_data)
    return data
    
       

def write_error(stn, d, msg):
    if stn in list(d.keys()):
        d[stn] += [msg]
    else:
        d[stn] = [msg]
    return d

# for region in code
region_codes = sorted(list(set(code_dict.values())))

DEM_source = 'EarthEnv_DEM90'

errors = {}
t_start = time.time()
for res in ['res1']:#, 'med', 'hi']:
    for polygon_source in ['manual']: #['manual', '']
        all_dfs = []
        for region in region_codes:
            
            stations = sorted([s for s in list(set(code_dict.keys())) if code_dict[s] == region])

            # load region code-specific resources once
            dem_path = processed_dem_dir + f'{region}_{DEM_source}_3005_{res}.tif'
            dem_raster, raster_crs, affine = retrieve_raster(dem_path)

            for station in stations:
                print(f'  Starting validation of {station}.')
                try:
                    basin_polygon = retrieve_basin_polygon(polygon_source, station, res)
                except Exception as e:
                    errors = write_error(station, errors, f'No basin polygon returned for {station}')
                    continue

                clipped_dem_raster, success = clip_raster_to_basin(station, basin_polygon, dem_raster)

                if (basin_polygon.geometry.area.values[0] / 1E6 < 1.0) | (not success):
                    print(f'    ....error in {station} raster at {res} resolution')
                    errors = write_error(station, errors, f'Dem clipping failed for {station}')
                    continue
                basin_characteristics_dict = process_basin_characteristics(region, station, clipped_dem_raster, basin_polygon)
                basin_characteristics = pd.DataFrame(basin_characteristics_dict)
                
                if not basin_characteristics.empty:
                    all_dfs.append(basin_characteristics)
                else:
                    errors = write_error(errors, f'Failed to generate basin characteristcs for {station}.')
            t_s = time.time()
            t_source = t_s - t0
            print(f'    ...completed {polygon_source} for {region} in {t_source:.0f}s')

        if len(all_dfs) > 0:
            results = pd.concat(all_dfs, axis=0, ignore_index=True)
            results = results.sort_values(by='OfficialID')
            out_path = os.path.join(validation_results_path, f'results/{polygon_source}_{DEM_source}_basin_characteristics_{res}.csv')
            results.to_csv(out_path)
        else:
            print('    No results returned.  Something real bad happened.')
        print(results.head())
        t_r = time.time()
        t_resolution = t_r - t0
        print(f'    ...completed {res} resolution cycle in {t_resolution:.0f}s')
        if len(list(errors.keys())) > 0:
            error_path = os.path.join(validation_results_path, f'results/{polygon_source}_errors_{res}res.pickle')
            with open(error_path, 'wb') as handle:
                pickle.dump(errors, handle, protocol=pickle.HIGHEST_PROTOCOL)

t_finish = time.time()
t_tot = t_finish - t0
print(f'Analysis completed in {t_tot:.0f} seconds.')
