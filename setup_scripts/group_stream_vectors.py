
import os

from multiprocessing import Pool

import pandas as pd
# import rioxarray as rxr
import geopandas as gpd 
import fiona

from shapely.geometry import Polygon
from shapely.ops import linemerge

import zipfile


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_data_dir = os.path.join(BASE_DIR, 'processed_data')

vector_save_path = os.path.join(processed_data_dir, 'grouped_hydrographic_features')
if not os.path.exists(vector_save_path):
    os.mkdir(vector_save_path)

# 
# Using the regrouped hydrologic regions, (process_hydrologic_regions.py),
# group the stream vectors for dem processing
# 

def fill_holes(data):           
    interior_gaps = data.interiors.values.tolist()[0]
    group_name = data.index.values[0]
    gap_list = []
    if interior_gaps is not None:
        print(f'   ...{len(interior_gaps)} gaps found in {group_name} groupings.')
        for i in interior_gaps:
            gap_list.append(Polygon(i))
        data_gaps = gpd.GeoDataFrame(geometry=gap_list, crs=data.crs)
        
        appended_set = data.append(data_gaps)
        appended_set['group'] = 0
        merged_polygon = appended_set.dissolve(by='group')
        return merged_polygon.geometry.values[0]
    else:
        print(f'  ...no gaps found in {group_name}')
        return data.geometry.values[0]

# nhn_path = '/media/danbot/Samsung_T5/geospatial_data/WSC_data/NHN_feature_data/'
nhn_path = '/home/danbot/Documents/code/hysets_validation/source_data/NHN_feature_data/'

nhn_feature_path = os.path.join(nhn_path, 'BC_NHN_features/')

seak_path = os.path.join(nhn_path, 'SEAK_features')

bc_groups_path = os.path.join(processed_data_dir, 'merged_basin_groups/')

bc_groups = gpd.read_file(bc_groups_path + 'BC_transborder_final_regions_4326.geojson')
bc_groups = bc_groups.to_crs(3005)

# 1. get the list of coastal + island regions
coast_groups = [
    '08A', '08B', '08C', '08D', 
    '08E', '08F', '08G', '08M', 
    '09M'
    ]
coast_islands = ['08O', '08H']
seak_groups = ['08A', '08B', '08C', '08D']

seak_dict = {
    '08A': [19010405, 19010404, 19010403, 19010406],
    '08B': [19010301, 19010302, 19010303, 19010304,
    19010206, 19010204, 19010212, 19010211],
    '08C': [19010210, 19010208, 19010207, 19010205],
    '08D': [19010103, 19010209, 19010104, 19010102],
}

# 2. retrieve the polygons associated with the 'region' boundary.
# 3. retrieve littoral / shoreline layers and merge them
# 4. split the region polygon using the line created in step 3.
# 5. discard the sea surface polygon
# 6. save new polygon and use to trim DEM in dem_basin_mapper.py

# collection of individual linestrings for splitting in a 
# list and add the polygon lines to it.
# line_split_collection.append(polygon.boundary) 
# merged_lines = shapely.ops.linemerge(line_split_collection)
# border_lines = shapely.ops.unary_union(merged_lines)
# decomposition = shapely.ops.polygonize(border_lines)

# load and merge the SEAK files into one gdf
seak_streams_path = os.path.join(nhn_path, 'SEAK_WBDHU8_polygons.geojson')
SEAK_polygons = gpd.read_file(seak_streams_path)
SEAK_polygons = SEAK_polygons.to_crs(3005)

SEAK_files = os.listdir(seak_path)

def retrieve_and_group_layers(feature_path, files, target_crs, target_layer):
    dfs = []
    all_crs = []
    print(f'    ...checking features at {feature_path} for layer {target_layer}.')
    for file in files:
        file_layers = zipfile.ZipFile(os.path.join(feature_path, file)).namelist()
        layers = [e for e in file_layers if (target_layer in e) & (e.endswith('.shp'))]
        if layers:
            for layer in layers:
                layer_path = os.path.join(feature_path, file) + f'!{layer}'
                df = gpd.read_file(layer_path)
                crs = df.crs
                print(f'    crs={crs}')
                if crs not in all_crs:
                    all_crs.append(crs)
                    print(f' new crs found: {crs}')
                df = df.to_crs(target_crs)
                # append the dataframe to the group list
                dfs.append(df)
        else:
            print(f'no target layers found in {file}')    
    return dfs


all_crs = []
# bc_groups = bc_groups[bc_groups['group_name'] == '08H'].copy()
# print(bc_groups)
target_crs = 3005
bc_groups = bc_groups.to_crs(target_crs)

bc_groups = bc_groups[bc_groups['group_name'].isin(['08B', '08C', '08D'])]

for i, row in bc_groups.iterrows():
    grp_code = row['group_name']
    sda_codes = row['WSCSDAs']
    if sda_codes == None:
        sda_codes = [row['group_code'].lower()]
        grp_code = row['group_code']
    else:
        sda_codes = [e.lower() for e in row['WSCSDAs'].split(',')]
    print(f'Starting stream vector merge on {grp_code}: {sda_codes}')
    nhn_files = [e for e in os.listdir(nhn_feature_path) if e.split('_')[2][:3] in sda_codes]

    # there is one sub-sub basin region polygon that has 
    # a corrupt archive and needs to be filtered out
    bad_zip_file_link = 'https://ftp.maps.canada.ca/pub/nrcan_rncan/vector/geobase_nhn_rhn/shp_en/08/nhn_rhn_08nec00_shp_en.zip'
    bad_zip_file = bad_zip_file_link.split('/')[-1]
    
    # skip the bad file:
    nhn_files_trimmed = [f for f in nhn_files if f != bad_zip_file]

    seak_included = False

    for target_layer in ['WATERBODY', 'ISLAND', 'NLFLOW', 'LITTORAL',]: 
        df_list = []
        group_stream_layers = []
        print(f'    Starting merge of {target_layer} features.')
        output_folder = os.path.join(vector_save_path, f'{grp_code}/{target_layer}/')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # use geojson for littoral and island (polygons)
        # use .shp for stream network (NLFLOW layer)
        output_filename = f'{grp_code}_{target_layer}_{target_crs}.geojson'
        if target_layer in ['NLFLOW']:
            output_filename = f'{grp_code}_{target_layer}_{target_crs}.shp'

        output_filepath = os.path.join(output_folder, output_filename)
        if not os.path.exists(output_filepath):
            
            nhn_dfs = retrieve_and_group_layers(nhn_feature_path, nhn_files_trimmed, target_crs, target_layer)

            if len(nhn_dfs) == 0:
                continue
            else:
                nhn_gdf = gpd.GeoDataFrame(pd.concat(nhn_dfs, ignore_index=True), crs=target_crs)
                print(f'    {len(nhn_gdf)} NHN items found.')
            
                    
            # nhn_gdf['intersects_group_polygon'] = gpd.sjoin(gdf, row, how='inner', predicate='contains')
            # gdf = gdf[gdf['intersects_group_polygon']].copy()
            # print(nhn_gdf.head())
            
            if nhn_gdf.empty:
                continue
            else:
                df_list.append(nhn_gdf)
            
            if (target_layer == 'NLFLOW') & (grp_code in seak_dict.keys()):
                huc_codes = [str(e) for e in seak_dict[grp_code]]
                print('')
                print(f'    ...searching for USGS vector info for {grp_code}.')
                
                group_seak_files = []
                for h in huc_codes:
                    files = [f for f in SEAK_files if h in f]
                    if len(files) > 0:
                        group_seak_files += files
                
                # there should be as many files as there are codes,
                # otherwise a file is missing.
                assert len(group_seak_files) == len(seak_dict[grp_code])

                # get the southeast alaska hydrographic feature files
                seak_dfs = retrieve_and_group_layers(seak_path, group_seak_files, target_crs, 'NHDFlowline')
                seak_gdf = gpd.GeoDataFrame(pd.concat(seak_dfs, ignore_index=True), crs=target_crs)
                # seak_gdf = seak_gdf.iloc[:5000]
                # seak_gdf = gpd.GeoDataFrame(pd.concat([gdf,seak_layer], ignore_index=True), crs=target_crs)
                print(f'    {len(seak_gdf)} SEAK items found.')
                if not seak_gdf.empty:
                    df_list.append(seak_gdf)
            
            if len(df_list) > 0:
                gdf = gpd.GeoDataFrame(pd.concat(df_list, ignore_index=True), crs=target_crs)
                
                # filter out geometries that lie outside of the group polygon
                # n_objects_before = len(gdf)
                # if target_layer == 'NLFLOW':
                #     print('   ...finding lines intersecting region polygon.')
                    
                # n_objects_after = len(gdf)
                # print(f'    {n_objects_before - n_objects_after} objects removed as non-intersecting.')

                gdf['geom_type'] = gdf.geometry.geom_type
                gdf['group_name'] = grp_code

                if target_layer == 'LITTORAL':
                    # cut out very small polygons (< 1km^2)
                    min_area = 1E6
                    gdf = gdf.to_crs(3005)

                    merged = linemerge(gdf.geometry.values)
                    merged_gdf = gpd.GeoDataFrame(geometry=[merged], crs=gdf.crs).explode(index_parts=False)
                    merged_gdf['is_ring'] = merged_gdf.geometry.is_ring
                    islands = merged_gdf[merged_gdf['is_ring']]
                    islands.geometry = [Polygon(e) for e in islands.geometry]
                    islands['area'] = islands.geometry.area
                    islands = islands[islands['area'] >= min_area]
                
                # file extension must be .shp for whiteboxtools StreamFill function.
                if target_layer in ['ISLAND', 'LITTORAL']:
                    print(f'   ...dissolving {target_layer} in {grp_code}.')
                    dissolved_regions = gdf.dissolve(by='group_name', dropna=True, aggfunc='sum')
                    # fill holes and gaps in merged polygons
                    dissolved_regions.to_file(output_filepath, driver='GeoJSON')
                else:
                    gdf.to_file(output_filepath)

                fname = output_filepath.split('/')[-1]
                print(f'   ...saved {fname}')
        else:
            fpath = output_filepath.split('/')[-1]
            print(f'file {fpath} exists')
