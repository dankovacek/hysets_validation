
import os

from multiprocessing import Pool

import pandas as pd
# import rioxarray as rxr
import geopandas as gpd 
import fiona

from shapely.geometry import Polygon
from shapely.ops import linemerge

import zipfile

#
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

nhn_path = '/media/danbot/Samsung_T5/geospatial_data/WSC_data/NHN_feature_data/'

bc_groups_path = '../processed_data/merged_basin_groups/'

bc_groups = gpd.read_file(bc_groups_path + 'BC_basin_region_groups_EPSG4326.geojson')

group_vector_save_path = bc_groups_path + 'group_stream_vectors/'

# 1. get the list of coastal + island regions
coast_groups = [
    '08A', '08B', '08C', '08D', 
    '08E', '08F', '08G', '08M', 
    '09M']
coast_islands = ['08O', '08H']
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


all_crs = []
bc_groups = bc_groups[bc_groups['group_name'] == '08H'].copy()
print(bc_groups)

for i, row in bc_groups.iterrows():
    grp_code = row['group_name']

    sda_codes = [e.lower() for e in row['WSCSDAs'].split(',')]
    
    print(f'Starting stream vector merge on {grp_code}: {sda_codes}')
    nhn_files = [e for e in os.listdir(nhn_path) if e.split('_')[2][:3] in sda_codes]

    df_list = []
    group_stream_layers = []
    # there is one sub-sub basin region polygon that has 
    # a corrupt archive and needs to be filtered out
    bad_zip_file_link = 'https://ftp.maps.canada.ca/pub/nrcan_rncan/vector/geobase_nhn_rhn/shp_en/08/nhn_rhn_08nec00_shp_en.zip'
    bad_zip_file = bad_zip_file_link.split('/')[-1]
    # skip the bad file:
    # print(bad_zip_file)
    nhn_files_trimmed = [f for f in nhn_files if f != bad_zip_file]

    for target_layer in ['ISLAND', 'NLFLOW', 'LITTORAL']:
        # if (target_layer == 'LITTORAL') & (grp_code not in coast_groups):
        #     continue
        # polygons are used against mosaic dem vrt which is in 4326
        # streams are used to clip the DEM which is in 3005
        target_crs = 3005
        output_filepath = group_vector_save_path + f'{target_layer}/{grp_code}_{target_layer}_{target_crs}.shp'
        if target_layer in ['LITTORAL', 'ISLAND']:
            target_crs = 4326
            output_filepath = group_vector_save_path + f'{target_layer}/{grp_code}_{target_layer}_{target_crs}.geojson'

        if not os.path.exists(output_filepath):
            for file in nhn_files_trimmed:
                try:
                    layers = zipfile.ZipFile(nhn_path + file).namelist()
                
                except Exception as e:
                    print(f'failed opening {nhn_path + file}')
                    break
                
                target_layers = [e for e in layers if (target_layer in e) & (e.endswith('.shp'))]
                
                if target_layers:
                    for sl in target_layers:
                        layer_path = nhn_path + file + f'!{sl}'
                        df = gpd.read_file(layer_path)
                        crs = df.crs
                        if crs not in all_crs:
                            all_crs.append(crs)
                            print(f' new crs found: {crs}')
                        df = df.to_crs(target_crs)
                        # append the dataframe to the group list
                        df_list.append(df)
                else:
                    print(f'no target layers found in {file}')

            # convert to geopandas dataframe
            gdf = gpd.GeoDataFrame(pd.concat(df_list, ignore_index=True), crs=target_crs) 
            

            gdf['geom_type'] = gdf.geometry.geom_type
            gdf['group_name'] = grp_code

            if target_layer == 'LITTORAL':
                # cut out very small polygons (< 0.5km^2)
                min_area = 1E6
                gdf = gdf.to_crs(3005)

                merged = linemerge(gdf.geometry.values)
                merged_gdf = gpd.GeoDataFrame(geometry=[merged], crs=gdf.crs).explode(index_parts=False)
                merged_gdf['is_ring'] = merged_gdf.geometry.is_ring
                islands = merged_gdf[merged_gdf['is_ring']]
                islands.geometry = [Polygon(e) for e in islands.geometry]
                islands['area'] = islands.geometry.area
                islands = islands[islands['area'] >= min_area]
  
            print(f'   ...dissolving {target_layer} in {grp_code}.')
            dissolved_regions = gdf.dissolve(by='group_name', dropna=True, aggfunc='sum')
            
            # file extension must be .shp for whiteboxtools StreamFill function.
            if target_layer in ['ISLAND', 'LITTORAL']:
                # fill holes and gaps in merged polygons
                dissolved_regions.to_file(output_filepath, driver='GeoJSON')

            else:
                dissolved_regions.to_file(output_filepath)

            fname = output_filepath.split('/')[-1]
            print(f'   ...saved {fname}')
    else:
        foo = output_filepath.split('/')[-1]
        print(f'file {foo} exists')
