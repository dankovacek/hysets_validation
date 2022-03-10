
import os
import pickle

from multiprocessing import Pool

import urllib.request as urq

import pandas as pd

import rioxarray as rxr
import geopandas as gpd 
import shapely
from shapely.geometry import Polygon, Point

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'source_data/')
# DATA_DIR = '/media/danbot/Samsung_T5/geospatial_data/'

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
bc_polygons = bc_polygons[bc_polygons['WSCMDA'] != '05']
# also drop 07A -- Upper Athabasca
bc_polygons = bc_polygons[bc_polygons['WSCSDA'] != '07A']

# merge wscssda polygons into more general wscsda groups
dissolved_regions = bc_polygons.dissolve(by='WSCSDA')


def download_files(input):
    (url, save_path) = input
    # print(url)
    # print(save_path)
    urq.urlretrieve(url, save_path)


def download_hydrologic_feature_data(pgs):
    save_directory = os.path.join(DATA_DIR, 'NHN_feature_data/')
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    
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

download_hydrologic_feature_data(bc_polygons)
print(f'   ...hydrologic feature files downloaded successfully.')

def fill_holes(data):           
    interior_gaps = data.interiors.values.tolist()[0]
    group_name = data.index.values[0]
    gap_list = []
    if interior_gaps is not None:
        print(f'   ...{len(interior_gaps)} gaps found in {group_name} groupings.')
        for i in interior_gaps:
            gap_list.append(Polygon(i))
        data_gaps = gpd.GeoDataFrame(geometry=gap_list, crs=data.crs)
        data_gaps['WSCSDA'] = [f'{i}' for i in data_gaps.index]
        
        appended_set = data.append(data_gaps)
        appended_set['group'] = 0
        merged_polygon = appended_set.dissolve(by='group')
        return merged_polygon.geometry.values[0]
    else:
        print(f'  ...no gaps found in {group_name}')
        return data
    
# fill holes and gaps in merged polygons
bc_filled = dissolved_regions.copy()
for i, grp in dissolved_regions.iterrows():
    data = dissolved_regions[dissolved_regions.index == i]
    bc_filled.loc[i, 'geometry'] = fill_holes(data)

# further group the Liard, Fraser, and Peace sub-basins
groups = {
    'Fraser': ['08J', '08K', '08M', '08L'],
    'Liard': ['10A', '10B', '10C', '10D'],
    'Peace': ['07E', '07F'],
}

# rename groups in order to merge shapes in next step
bc_filled['group_name'] = bc_filled.index.values
for k, g in groups.items():
    bc_filled.loc[bc_filled.index.isin(g), 'group_name'] = k

merged_regions = bc_filled.dissolve(by='group_name')
# find and fill holes in the dissolved polygons
bc_merged_filled = merged_regions.copy()
for i, grp in bc_merged_filled.iterrows():
    data = bc_merged_filled[bc_merged_filled.index == i]
    bc_merged_filled.loc[i, 'geometry'] = fill_holes(data)

# add in the WSCSDA
bc_merged_filled['WSCSDAs'] = bc_merged_filled.index.values
for k, v in groups.items():
    bc_merged_filled.loc[bc_merged_filled.index == k, 'WSCSDAs'] = ','.join(v)


dem_dir = os.path.join(DATA_DIR, 'dem_data/')

dem = rxr.open_rasterio(dem_dir + 'BC_DEM_mosaic_4326.vrt')
dem_crs = dem.rio.crs.to_epsg()

# save the output file
bc_merged_filled = bc_merged_filled.to_crs(dem_crs)
output_folder = os.path.join(BASE_DIR, 'processed_data/merged_basin_groups/')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
bc_merged_filled.to_file(output_folder + f'BC_basin_region_groups_EPSG{dem_crs}.geojson', driver='GeoJSON')
print(f'   ...BC_basin_region_groups.geojson file created successfully.')

# save each row as a separate shape file
split_out_dir = output_folder + 'split_groups/'
if not os.path.exists(split_out_dir):
    os.mkdir(split_out_dir)
for code in bc_merged_filled.index.values:
    gdf = bc_merged_filled[bc_merged_filled.index == code].copy()
    gdf.to_file(split_out_dir + f'{code}_{dem_crs}.geojson', driver='GeoJSON')
print(f'   ...saved individual polygons seperately at {split_out_dir}')


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
mapper_dict_file = 'station_to_region_mapper.pickle'
if not os.path.exists(stn_mapper_path):
    os.mkdir(stn_mapper_path)


existing_mappers = os.listdir(stn_mapper_path)
if not os.path.exists(os.path.join(stn_mapper_path, mapper_dict_file)):
    code_dict = map_stations_to_basin_groups(basin_polygons_path)
    with open(stn_mapper_path + mapper_dict_file, 'wb') as handle:
        pickle.dump(code_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    print('  Mapper dict already created.')
    for reg, stns in code_dict.items():
        print(f'{reg}: {stns}')