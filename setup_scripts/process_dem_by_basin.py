import os
import time

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import geopandas as gpd
import rioxarray as rxr

import logging

import richdem as rd

from whitebox.whitebox_tools import WhiteboxTools

wbt = WhiteboxTools()

cwd = os.getcwd()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'processed_data/merged_basin_groups/')


wbt.verbose = True
wbt.set_working_dir(cwd)

dem_dir = '../source_data/dem_data/processed_dem/'
dem_dir = '/media/danbot/Samsung_T5/geospatial_data/DEM_data/processed_dem/'

if not os.path.exists(dem_dir):
    os.mkdir(dem_dir)

# BC basin region groups file is in EPSG 4326
bc_basins_file = 'BC_basin_region_groups_EPSG4326.geojson'

bc_basin_groups = gpd.read_file(DATA_DIR + bc_basins_file)
bc_basin_groups = bc_basin_groups.to_crs(3005)
bc_basin_groups['Area_km2'] = bc_basin_groups.geometry.area / 1E6
bc_basin_groups.sort_values(by='Area_km2', inplace=True)
# bc_basin_groups = bc_basin_groups.to_crs(4326)
bc_basin_groups.reset_index(inplace=True, drop=True)


def richdem_functions(out_path, data, analysis_type):

    t0 = time.time()

    dem = rd.rdarray(data.data[0], no_data=data.rio.nodata)
    dem.projection = data.rio.crs.to_wkt()
    dem.geotransform = data.rio.transform()
    
    #Fill depressions in the DEM. The data is modified in-place to avoid making
    #an unnecessary copy. This saves both time and RAM. Note that the function
    #has no return value when `in_place=True`.
    if analysis_type == 'breach_depressions':
        result = rd.BreachDepressions(dem, in_place=False)
    elif analysis_type == 'dem_fill':
        result = rd.FillDepressions(dem, in_place=False)
    elif analysis_type == 'flow_accumulation':
        result = rd.FlowAccumulation(dem, method='D8')
    else:
        raise Exception; "No analysis_type specified."

    #Save the DEM    

    data.data[0] = result.data
    data.rio.to_raster(out_path)

    t1 = time.time()
    tot_t = t1 - t0

    return tot_t


def retrieve_stream_vector_path(grp):
    stream_layer_fname = f'{grp}_NLFLOW_3005.shp'
    fpath = DATA_DIR + f'group_stream_vectors/NLFLOW/{stream_layer_fname}'
    # stream_vectors = gpd.read_file(fpath)
    return fpath


def process_dem_fill_single_pits(row, res):
    t0 = time.time()
    grp = row['group_name']
    in_path = dem_dir + f'{grp}_DEM_3005_{res}.tif'
    out_path = dem_dir + f'{grp}_FILLPITS_3005_{res}.tif'
    wbt.fill_single_cell_pits(
        in_path, 
        out_path, 
    )
    t1 = time.time()
    return t1 - t0


def process_dem_fill(row, res, dem, which_tool):
    # which tool should be RDEM or WBT
    grp = row['group_name']
    tot_time = np.nan
    # in_path = dem_dir + f'{grp}_DEM_3005_{res}.tif'
    in_path = dem_dir + f'{grp}_FILLPITS_3005_{res}.tif'
    out_path = dem_dir + f'{grp}_{which_tool}_FILL_3005_{res}.tif'

    if not os.path.exists(out_path):
        t0 = time.time()
        wbt.fill_depressions(
            in_path,
            out_path,
            fix_flats=True,
            flat_increment=None,
            max_depth=None,
        )
        t1 = time.time()
        tot_time = t1 - t0
    return tot_time


def process_dem_fillburn(row, res):
    grp = row['group_name']
    in_path = dem_dir + f'{grp}_FILLPITS_3005_{res}.tif'

    out_path = dem_dir + f'{grp}_WBT_FILLBURN_3005_{res}.tif'
    t0 = time.time()
    streams = retrieve_stream_vector_path(grp)

    if not os.path.exists(out_path):
        wbt.fill_burn(
            in_path, 
            streams, 
            out_path, 
        )    
    t1 = time.time()
    return t1 - t0


def process_flow_direction(row, res):
    grp = row['group_name']
    in_path = dem_dir + f'{grp}_WBT_FILL_3005_{res}.tif'
    out_path = dem_dir + f'{grp}_DIR_3005_{res}.tif'
    if not os.path.exists(out_path):
        wbt.d8_pointer(
            in_path, 
            out_path, 
            esri_pntr=False, 
        )


def process_flow_accumulation(row, res, in_path, dem_fill_method, dem_dir_method, cell_threshold=None):
    # DEM fill methods can be D8, DINF, 
    grp = row['group_name']
    out_path = dem_dir + f'{grp}_{tool}_ACC_{dem_dir_method}_{dem_fill_method}_3005_{res}.tif'
    t0 = time.time()
    in_path = in_path
    t_total = np.nan
    
    if not os.path.exists(out_path):
        # methods of flow accumulation with WBT are:
        # D8, DINF, or MDINF
        if dem_dir_method == 'D8':

            wbt.d8_flow_accumulation(
                in_path, 
                out_path, 
                out_type="cells", 
                log=False, 
                clip=False, 
                pntr=False, 
                esri_pntr=False, 
            )
        elif dem_dir_method == 'DINF':

            wbt.d_inf_flow_accumulation(
                in_path, 
                out_path, 
                out_type='cells', 
                threshold=None, 
                log=False, 
                clip=False, 
                pntr=False, 
            )
        elif dem_dir_method == 'MDINF':

            wbt.md_inf_flow_accumulation(
                in_path, 
                out_path, 
                out_type="cells", 
                exponent=1.1, 
                threshold=None, 
                log=False, 
                clip=False, 
            )
        t1 = time.time()
        t_total = t1 - t0
    
    return t_total


def extract_streams(in_path, out_path, stream_threshold):
    t0 = time.time()
    if not os.path.exists(out_path):
        wbt.extract_streams(
            in_path,
            out_path,
            threshold=stream_threshold,
            zero_background=False,
        )
    t1 = time.time()
    return t1 - t0    

min_basin_area = 1E6 # 1E6 m^2 is 1 km^2

results = [] 
# basin_sample = bc_basin_groups.copy()[:3]
bc_basin_groups = bc_basin_groups
for res in ['low', 'med', 'hi']:
    # for i, row in basin_sample.iterrows():
    for i, row in bc_basin_groups.iterrows():

        print(row)
        print(asfds)
        
        values = []
        if i == 0:
            time_cols = ['group_name', 'n_pixels']
        grp = row['group_name']
        values.append(grp)
        # row = bc_basin_groups[bc_basin_groups['group_name'] == '08O'] 
        print('')       
        print('')       
        print('')       
        print('')       
        print('######################################')       
        print(f'Starting DEM processing for {grp}.  {i+1}/{len(bc_basin_groups)}.')

        # open DEM and retrieve raster characteristics
        original_dem_path = dem_dir + f'{grp}_DEM_3005_{res}.tif'

        dem = rxr.open_rasterio(original_dem_path)
        projection = dem.rio.crs.to_wkt()
        cell_size = dem.rio.resolution()        
        num_pixels = dem.shape[0] * dem.shape[1]
        no_data = dem.rio.nodata
        values.append(num_pixels)
        print(f'   ...raster has {num_pixels:.2e} pixels')
        print(f'   ...raster resolution is {abs(cell_size[0]):.0f}m x {abs(cell_size[1]):.0f}m')
        

        # fill single pits outputs a file with the format:
        # dem_dir + f'{grp}_FILLPITS_3005_{res}.tif'
        # t_fill_pits = process_dem_fill_single_pits(row, res)
        # if i == 0:
        #     time_cols.append('fill_single_pits')
        # values.append(t_fill_pits)

        process_flow_direction(row, res)

        # for tool in ['RDEM', 'WBT']:
        # tool = 'WBT'
        # t_fill = process_dem_fill(row, res, dem, tool)
        # values.append(t_fill)
        # if i == 0:
        #     # time_cols += ['fill_rdem', 'fill_wbt']  
        #     time_cols += ['fill_wbt'] 
            
        # process the fillburn
        # t_fillburn = process_dem_fillburn(row, res)
        # values.append(t_fillburn)
        # if i == 0:
        #     time_cols.append('fill_burn')

        # process flow accumulation using WBT (takes DIR pointer raster)
        # tool = 'WBT'
        # for fill_method in ['FILL', 'FILLBURN']:
        #     for dir_method in ['D8', 'DINF']:
        #         input_path = dem_dir + f'{grp}_{tool}_{fill_method}_3005_{res}.tif'
        #         # minimum number of cells representing stream convergence
        #         cell_threshold = int( min_basin_area / ( abs(cell_size[0]) * abs(cell_size[1]) ))
                
        #         acc_t = process_flow_accumulation(row, res, input_path, fill_method, dir_method, cell_threshold)
                                
        #         values.append(acc_t)
        #         if i == 0:
        #             time_cols.append(f'acc_{tool}_{dir_method}_{fill_method}')

        #         # in_path = dem_dir + f'{grp}_{tool}_ACC_{dir_method}_{fill_method}_3005_{res}.tif'      
        #         out_path = dem_dir + f'{grp}_STREAMS_{dir_method}_{fill_method}_3005_{res}.tif'
        #         in_path = dem_dir + f'{grp}_{tool}_ACC_{dir_method}_{fill_method}_3005_{res}.tif'
        #         stream_extract_t = extract_streams(in_path, out_path, cell_threshold)

        #         values.append(stream_extract_t)
        #         if i == 0:
        #             time_cols.append(f'streams_{dir_method}_{fill_method}')


        # results.append(values)

    
# df = pd.DataFrame(results, columns=time_cols)
# df = df.T
# test_path = os.path.join(BASE_DIR, 'setup_scripts/performance_tests/')
# df.to_csv(test_path + 'DEM_Processing_Performance_Comparison.csv')
