
import os
import numpy as np

from multiprocessing import Pool
import json
import urllib.request as urq

import pandas as pd

import rioxarray as rxr
import geopandas as gpd 
import shapely
from shapely.geometry import Polygon, Point

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'source_data/')
file_list_dir = os.path.join(BASE_DIR, 'setup_scripts/file_lists/')
# DATA_DIR = '/media/danbot/Samsung_T5/geospatial_data/'

DEM_source = 'USGS_3DEP'

dem_mosaic_file = 'EENV_DEM_mosaic_4326.vrt'
if DEM_source == 'USGS_3DEP':
    dem_mosaic_file = 'USGS_3DEP_mosaic_4269.vrt'

nhn_path = os.path.join(DATA_DIR, 'NHN_data/')
nhn_fname = 'NHN_INDEX_WORKUNIT_LIMIT_2.zip'
if not os.path.exists(nhn_path + nhn_fname):
    print(f'   ...downloading national hydrologic network index file.')
    file_url = f'https://ftp.maps.canada.ca/pub/nrcan_rncan/vector/geobase_nhn_rhn/index/{nhn_fname}'
    if not os.path.exists(nhn_path):
        os.mkdir(nhn_path)
    urq.urlretrieve(file_url, nhn_path + nhn_fname)
    print(f'   ...{nhn_fname} downloaded.')


nhn = gpd.read_file(nhn_path + nhn_fname)
nhn = nhn.to_crs(3005)

# import BC border polygon
bc_border_file = os.path.join(DATA_DIR, 'BC_border/BC_PROV_BOUNDARIES_LINES_500M.geojson')
bc_border_lines = gpd.read_file(bc_border_file)
bc_border_crs = bc_border_lines.crs
bc_border_polygon = Polygon(shapely.ops.linemerge(bc_border_lines.geometry.values))
bc_border = gpd.GeoDataFrame(geometry=[bc_border_polygon], crs=bc_border_crs)
bc_border = bc_border.to_crs(3005)

# get all NHN polygons intersecting with the BC polygon
bc_polygons = nhn[nhn.intersects(bc_border.geometry.values[0])]

# drop the 05 prefix (these bound the east side of the rockies)
# consider including these if comparing sets of in-basin/out-basin correlations
# bc_polygons = bc_polygons[bc_polygons['WSCMDA'] != '05']
# also drop 07A -- Upper Athabasca
# bc_polygons = bc_polygons[bc_polygons['WSCSDA'] != '07A']

# collect the polygons on the eastern side of the rockies
e_rockies_ssdas = [
    '05AA', '05AB', '05AD', '05AE', 
    '05BF', '05BJ', '05BL', '05BK', '05BL',
    '05BA', '05BB', '05BC', '05BD', '05BE', '05BG',
    '05CA', '05DA', '05DB', '05DC', '05DD', 
    '07AA', '07AB', '07AC', '07AD', '07AF']
e_rockies = nhn[nhn['WSCSSDA'].isin(e_rockies_ssdas)]
bc_polygons = bc_polygons.append(e_rockies)

alsek = ['08A']
alsek_polygons = nhn[nhn['WSCSDA'].isin(alsek)]
bc_polygons = bc_polygons.append(alsek_polygons)

bc_polygons.sort_values(by='WSCSSDA', inplace=True)
bc_polygons.drop_duplicates(subset='geometry', inplace=True, keep='first')

# merge wscssda polygons into more general wscsda groups
dissolved_regions = bc_polygons.dissolve(by='WSCSDA')

def download_files(input):
    (url, save_path) = input
    urq.urlretrieve(url, save_path)


def download_NHN_hydrologic_feature_data(pgs):
    save_directory = os.path.join(DATA_DIR, 'NHN_feature_data/BC_NHN_features/')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    all_links = []
    for _, row in pgs.iterrows():     
        dataset_name = row['DATASETNAM'].lower()
        region_no = row['WSCMDA']
        fname = f'nhn_rhn_{dataset_name}_shp_en.zip'
        base_url = f'https://ftp.maps.canada.ca/pub/nrcan_rncan/vector/geobase_nhn_rhn/shp_en/{region_no}/{fname}'

        if not os.path.exists(save_directory + fname):
            print(f'   ...{fname} will be downloaded.')
            print(base_url)
            all_links.append((base_url, save_directory+fname))
    print(f'   ...{len(all_links)} hydrologic feature files to be downloaded.')
    with Pool() as p:
        p.map(download_files, all_links)
        

def download_SEAK_streams_feature_data():
    save_path = os.path.join(DATA_DIR, 'NHN_feature_data/SEAK_features/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    seak_df = pd.read_csv(file_list_dir+'USGS_hydrography_data.txt', header=None)
    seak_df.columns = ['links']
    seak_links = seak_df['links'].values

    existing_files = os.listdir(save_path)
    seak_links = [l for l in seak_links if l.split('/')[-1] not in existing_files]

    dl_input = [(l, save_path+l.split('/')[-1]) for l in seak_links]

    with Pool() as p:
        p.map(download_files, dl_input)


download_SEAK_streams_feature_data()
print(f'   ...South East Alaska hydrologic feature files downloaded successfully.')

download_NHN_hydrologic_feature_data(bc_polygons)
print(f'   ...BC hydrologic feature files downloaded successfully.')

# def fill_holes(data):           
#     interior_gaps = data.interiors.values.tolist()[0]
#     group_name = data.index.values[0]
#     gap_list = []
#     if interior_gaps is not None:
#         print(f'   ...{len(interior_gaps)} gaps found in {group_name} groupings.')
#         for i in interior_gaps:
#             gap_list.append(Polygon(i))
#         data_gaps = gpd.GeoDataFrame(geometry=gap_list, crs=data.crs)
#         data_gaps['WSCSDA'] = [f'{i}' for i in data_gaps.index]
        
#         appended_set = data.append(data_gaps)
#         appended_set['group'] = 0
#         merged_polygon = appended_set.dissolve(by='group')
#         return merged_polygon.geometry.values[0]
#     else:
#         print(f'  ...no gaps found in {group_name}')
#         return data
    
# # fill holes and gaps in merged polygons
# bc_filled = dissolved_regions.copy()
# for i, grp in dissolved_regions.iterrows():
#     data = dissolved_regions[dissolved_regions.index == i]
#     bc_filled.loc[i, 'geometry'] = fill_holes(data)

# # further group the Liard, Fraser, and Peace sub-basins
# groups = {
#     'Fraser': ['08J', '08K', '08M', '08L'],
#     'Liard': ['10A', '10B', '10C', '10D'],
#     'Peace': ['07E', '07F'],
#     'ERockies': ['05A', '05B', '05C', '05D', '07A']
# }

# # rename groups in order to merge shapes in next step
# bc_filled['group_name'] = bc_filled.index.values
# for k, g in groups.items():
#     bc_filled.loc[bc_filled.index.isin(g), 'group_name'] = k

# merged_regions = bc_filled.dissolve(by='group_name')
# # # find and fill holes in the dissolved polygons
# bc_merged_filled = merged_regions.copy()

# for i, grp in bc_merged_filled.iterrows():
#     data = bc_merged_filled[bc_merged_filled.index == i]
#     filled_geom = fill_holes(data)
    
#     # to add a buffer, the crs must be in epsg 3005
#     assert bc_merged_filled.crs == 3005
#     buffer = 500
#     if i in ['07O', '07U', '08P', '07G', '08E', '09A', 'Peace', 'Liard']:
#         buffer = 1000
#     buffered_geom = filled_geom.buffer(1000, resolution=50)
#     bc_merged_filled.loc[i, 'geometry'] = buffered_geom

# # add in the WSCSDA
# bc_merged_filled['WSCSDAs'] = bc_merged_filled.index.values
# for k, v in groups.items():
#     bc_merged_filled.loc[bc_merged_filled.index == k, 'WSCSDAs'] = ','.join(v)

# dem_dir = os.path.join(DATA_DIR, 'dem_data/')

# dem = rxr.open_rasterio(dem_dir + dem_mosaic_file)
# dem_crs = dem.rio.crs.to_epsg()

# # save the output file
# bc_merged_filled = bc_merged_filled.to_crs(dem_crs)

# output_folder = os.path.join(BASE_DIR, 'processed_data/merged_basin_groups/')
# if not os.path.exists(output_folder):
#     os.mkdir(output_folder)

# bc_merged_filled.to_file(output_folder + f'BC_basin_region_groups_EPSG{dem_crs}.geojson', driver='GeoJSON')
# print(f'   ...BC_basin_region_groups.geojson file created successfully.')

# # save each row as a separate shape file
# split_out_dir = output_folder + 'split_groups/'
# if not os.path.exists(split_out_dir):
#     os.mkdir(split_out_dir)
# for code in bc_merged_filled.index.values:
#     gdf = bc_merged_filled[bc_merged_filled.index == code].copy()
#     gdf.to_file(split_out_dir + f'{code}_{dem_crs}.geojson', driver='GeoJSON')
#     print(f'   ...saved individual polygons separately at {split_out_dir}')

# create a dictionary where the key: value pairs 
# map stations to the regional group name
hysets_df = pd.read_csv(os.path.join(BASE_DIR, 'source_data/HYSETS_data/HYSETS_watershed_properties.txt'), sep=';')
hysets_locs = [Point(x, y) for x, y in zip(hysets_df['Centroid_Lon_deg_E'].values, hysets_df['Centroid_Lat_deg_N'])]
hysets_df = gpd.GeoDataFrame(hysets_df, geometry=hysets_locs, crs='EPSG:4269')
hysets_df = hysets_df.to_crs(3005)

basin_polygons_path = BASE_DIR + '/processed_data/merged_basin_groups/final_polygons/'

if not os.path.exists(basin_polygons_path):
    os.mkdir(basin_polygons_path)


def get_grp_polygon(basin_polygons_path, polygon_fnames, grp_code):
    grp_polygon_path = [e for e in polygon_fnames if grp_code in e][0]
    grp_polygon = gpd.read_file(basin_polygons_path + grp_polygon_path)
    grp_polygon = grp_polygon.to_crs(3005)
    grp_polygon['grp'] = grp_code
    grp_polygon = grp_polygon.dissolve(by='grp', aggfunc='sum')
    return grp_polygon


def map_stations_to_basin_groups(nhn_group_polygon_path):    
    # get all the stations that fall within the study area
    # create a dict to organize stations by their 
    # location in the regional groupings
    region_dict = {}
    regional_group_polygon_fnames = os.listdir(nhn_group_polygon_path)
    code_dict = {}
    for fname in regional_group_polygon_fnames:
        grp_code = fname.split('_')[0]
        print(f'  Finding HYSETS stations within {grp_code}.')
        grp_polygon = get_grp_polygon(basin_polygons_path, regional_group_polygon_fnames, grp_code)

        intersecting_pts = gpd.sjoin(hysets_df, grp_polygon, how='inner', predicate='intersects')
        station_IDs = list(intersecting_pts['Official_ID'].values)
        region_dict[grp_code] = station_IDs

        for stn in station_IDs:
            if stn in code_dict.keys():
                print(f'  for some reason, {stn} is already in code_dict...')
                print(f'    under {grp_code}, code_dict[stn] = {code_dict[stn]}')
            code_dict[stn] = grp_code
            
    return code_dict

stn_mapper_path = os.path.join(BASE_DIR, 'processed_data/')
mapper_dict_file = 'station_to_region_mapper.json'
if not os.path.exists(stn_mapper_path):
    os.mkdir(stn_mapper_path)


existing_mappers = os.listdir(stn_mapper_path)
if not os.path.exists(os.path.join(stn_mapper_path, mapper_dict_file)):
    print('Creating station to region mapping object.')
    code_dict = map_stations_to_basin_groups(basin_polygons_path)
    filepath = stn_mapper_path + mapper_dict_file
    with open(filepath, 'w') as fp:
        json.dumps(code_dict, fp)
else:
    print('  Mapper dict already created.')
    # for reg, stns in code_dict.items():
    #     print(f'{reg}: {stns}')