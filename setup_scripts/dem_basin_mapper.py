import os
import sys

import time

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import geopandas as gpd
import rioxarray as rxr
import rasterio

import shapely
from shapely.ops import linemerge, unary_union, polygonize, split
from shapely.geometry import Polygon

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'processed_data/')

mask_dir = DATA_DIR + f'merged_basin_groups/split_groups/'

t0 = time.time()

dem_dir = os.path.join(BASE_DIR, 'source_data/dem_data/')
# dem_dir = '/media/danbot/Samsung_T5/geospatial_data/DEM_data/'

bc_region_final_polygons_folder = DATA_DIR + 'merged_basin_groups/final_polygons/'
# bc_basins = gpd.read_file(bc_basins_file)
# bc_basins['Area_km2'] = bc_basins.geometry.area / 1E6
# bc_basins.reset_index(inplace=True)
# bc_basins = bc_basins.sort_values('Area_km2')
# bc_basins.reset_index(inplace=True, drop=True)
# basins_crs = bc_basins.crs

dem_mosaic_file = dem_dir + 'BC_DEM_mosaic_4326.vrt'

# coastline_path = '/home/danbot/Documents/code/hysets_validation/source_data/'
# bc_coast = gpd.read_file(coastline_path + 'BC_Coastline/FWA_COASTLINES_SP.geojson')
# create boolean variable if the linestring is a closed loop or not
# bc_coast['is_ring'] = bc_coast.geometry.is_ring
coast_groups = [
    '08A', '08B', '08C', '08D',
    '08F', '08G', '08H', '08O', 
    ]

save_path = dem_dir + 'processed_dem/'


if not os.path.exists(save_path):
    os.mkdir(save_path)


def get_crs_and_resolution(fname):
    raster = rxr.open_rasterio(fname)
    crs = raster.rio.crs.to_epsg()
    res = raster.rio.resolution()    
    return crs, res


def filter_polygons_by_elevation(grp_code, area_polygons):
    dem_file = dem_dir + f'processed_dem/{grp_code}_DEM_3005_hi.tif'
    dem = rxr.open_rasterio(dem_file)
    raster_crs = dem.rio.crs.to_epsg()
    raster_nodata = dem.rio.nodata

    area_polygons = area_polygons.to_crs(raster_crs)
    
    min_elevation = -1E6
    for p in area_polygons.values:
        with rasterio.open(dem_file) as src:
            out_image, out_transform = rasterio.mask.mask(src, [p], crop=True)
            out_image = out_image[out_image != raster_nodata]
            mean_elevation = np.nanmean(out_image[0])
            if mean_elevation > min_elevation:
                polygon_to_keep = p
    return polygon_to_keep, raster_crs
                

# def trim_raster_mask_polygon(grp_code):

#     group_polygon = bc_basins[bc_basins['group_name'] == grp_code].copy()


#     # figure out which regions are covered instead by alaska
#     # create two separate functions: one for BC coast, one for AK
#     # because AK coast is polygon, not line...
#     # alaska polygon goes through 08A, 08B, 08C, 08D.
#     # if grp_code in ['08A','08B','08C','08D']:
#     #     coast_geometry = ak_coast.copy()
#     # elif grp_code in []:
#     #     # remaining are covered by BC coastal polygon
#     #     coast_geometry = bc_coast.copy()
#     # else:
#     #     print(f'   ..{grp_code} does not lie along the coast.')
#     #     return None
    
#     filtered_coast = gpd.sjoin(coast_geometry, group_polygon, how='inner', predicate='intersects')
#     coastline_polygons = filtered_coast[filtered_coast['is_ring']].copy()
#     coastline_lines = filtered_coast[~filtered_coast['is_ring']].copy()
#     coastline_lines['line_type'] = 'coastline'

#     # merge the coastline linestrings into a multiline string
#     merged_coastline = linemerge(coastline_lines.geometry.values)
#     gdf = gpd.GeoDataFrame(geometry=[merged_coastline], crs=coastline_lines.crs)

#     coastline = gdf.explode(index_parts=False)
#     coastline['is_ring'] = coastline.geometry.is_ring
#     coastline = coastline[~coastline['is_ring']]

#     group_polygon = group_polygon.to_crs(3005).reset_index()
#     coastline = coastline.to_crs(3005).reset_index()
#     coast_section_length = coastline.geometry.length / 1E3
#     print(f'   ...{grp_code} segment coast length is {coast_section_length:.1} km long')

#     # if you add a 1m buffer to the coastline linestring    
#     # and subtract it from the polygon, you get new polygons
#     split_polygon = group_polygon.difference(coastline.buffer(1))
#     # explode the resulting split (multipolygon) into individual polygons
#     area_polygons = split_polygon.explode(index_parts=False)
#     # filtering out the rings and merging the coastline linestring
#     # should leave a single line object that can cut the overlapped
#     # polygon in two.  There should only be two resulting polygons,
#     # so discart the one with ~zero mean elevation because it's ocean.
#     polygon_to_keep, raster_crs = filter_polygons_by_elevation(grp_code, area_polygons)

#     # add any small islands that are included in the coastline geometry
#     coastline_polygons = coastline_polygons.to_crs(raster_crs)
#     coastline_polygons['area'] = coastline_polygons.geometry.area
#     # Trim out small islands
#     coastline_polygons = coastline_polygons[coastline_polygons['area'] >= 1E6]
    
#     all_polygons = coastline_polygons.append(gpd.GeoDataFrame(geometry=[polygon_to_keep], crs=raster_crs))

#     all_polygons['group_code'] = grp_code
#     dissolved_polygons = all_polygons.dissolve(by='group_code', dropna=True, aggfunc='sum')
#     dissolved_polygons['area'] = dissolved_polygons.geometry.area

#     # save the file as a new polygon to be used as the dem mask
#     mask_polygon = dissolved_polygons.to_crs(4326)
#     mask_output_fpath = DATA_DIR + f'merged_basin_groups/split_groups/{grp_code}_4326_trimmed.geojson'
#     mask_polygon.to_file(mask_output_fpath, driver='GeoJSON')

dem_crs, (w_res, h_res) = get_crs_and_resolution(dem_mosaic_file)

# bc_basins = bc_basins.to_crs(dem_crs)


def check_mask_validity(mask_path):
    mask = gpd.read_file(mask_path)
    
    if mask.geometry.is_valid.values[0]:
        print(f'   ...mask is valid.')
    else:
        file = mask_path.split('/')[-1]
        print(f'   ...{file} mask is invalid:')
        
        mask = mask.to_crs(3005)
        mask = mask.explode(index_parts=False).simplify(10)
        mask = gpd.GeoDataFrame(geometry=mask.values, crs='EPSG:3005')
        mask['valid'] = mask.geometry.is_valid
        # drop invalid geometries
        mask = mask[mask['valid']]
        mask['area'] = mask.geometry.area
        # reproject to 4326 to correspond with DEM tile mosaic
        mask = mask.to_crs(4326)
        
        fixed = all(mask.geometry.is_valid)
        
        if fixed:
            mask.to_file(mask_path, driver='GeoJSON')
            print(f'   ...invalid mask corrected: {fixed}')
        else:
            print(f'   ...invalid mask could not be corrected')


all_masks = os.listdir(bc_region_final_polygons_folder)

for file in all_masks:

    fpath = bc_region_final_polygons_folder + file

    grp_code = file.split('_')[0]
    print(f'Starting polygon merge on {grp_code}.')

    mask_polygon = gpd.read_file(fpath, driver='GeoJSON')
    print(f'mask crs: {mask_polygon.crs}')
    bounds = mask_polygon.geometry.bounds

    # eligible_mask_files = [e for e in os.listdir(mask_dir) if e.startswith(grp_code)]

    # # coastal region masks have the 'trimmed' suffix because they
    # # required an extra step to remove the ocean
    # trimmed_file = [e for e in eligible_mask_files if e.endswith('trimmed.geojson')]
    
    named_layer = file.split('.')[0]
    # if len(trimmed_file) == 1:
    #     named_layer = f'{grp_code}_4326_trimmed'

    # mask_path = mask_dir + f'{named_layer}.geojson'

    mask_check = check_mask_validity(fpath)
    
    for res in ['res8', 'res4', 'res2', 'res1']:
        # set the output initial path and reprojected path
        out_path = f'{save_path}{grp_code}_DEM_4326_{res}.tif'
        out_path_reprojected = f'{save_path}{grp_code}_DEM_3005_{res}.tif'

        if res == 'res8':
            rfactor = 8
        elif res == 'res4':
            rfactor = 4
        elif res == 'res2':
            rfactor = 2
        else:
            rfactor = 1

        trw = abs(w_res*rfactor)
        trh = abs(h_res*rfactor)

        if not os.path.exists(out_path_reprojected):

            command = f'gdalwarp -s_srs epsg:4326 -cutline {fpath} -cl {named_layer} -crop_to_cutline -tr {trw} {trh} -multi -of gtiff {dem_mosaic_file} {out_path} -wo NUM_THREADS=ALL_CPUS'
            print('')
            print('__________________')
            print(command)
            print('')
            try:
                os.system(command)
            except Exception as e:
                raise Exception; e
        else:
            fname = out_path_reprojected.split('/')[-1]
            print(f'   ...{fname} exists, skipping dem cutline operation..')

        
        # check # pixels low res        
        if not os.path.exists(out_path_reprojected):
            # reproject to epsg 3005
            lr = rxr.open_rasterio(out_path, masked=True, default='dem')
            lr = lr.rio.reproject(3005)
            lr.rio.to_raster(out_path_reprojected)
            lr_shape = lr.rio.shape
            n_pix = lr_shape[0] * lr_shape[0]
            print(f'   ...{res} img has {n_pix:.2e} pixels')

            os.remove(out_path)
        else:
            fname = out_path_reprojected.split('/')[-1]
            print(f'   ...{fname} exists, skipping dem reprojection..')
    
    t1 = time.time()
    print(f'      {i}/{len(all_masks)} Completed tile merge: {grp_code}_DEM_1as.tif created in {t1-t0:.1f}s.')
    print('')
    print('')
    