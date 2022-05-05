import os
from tabnanny import check
import time
import pickle

# import pandas as pd
import numpy as np

# import shapely
# from shapely.geometry import Polygon, Point
import geopandas as gpd

import xarray as xr
import rioxarray as rxr
import rasterio  as rio
from rasterio import features

from numba import jit

from skimage.morphology import skeletonize
from scipy import ndimage

from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder

from pyproj import CRS, Proj

import warnings
warnings.filterwarnings('ignore')

t0 = time.time()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'source_data/')

processed_data_dir = os.path.join(BASE_DIR, 'processed_data')
data_dir = '/media/danbot/Samsung_T5/geospatial_data/'

# dem_dir = os.path.join(DATA_DIR, 'dem_data/')
# dem_dir = os.path.join(data_dir, 'DEM_data/')

# processed_dem_dir = os.path.join(dem_dir, 'processed_dem/')

processed_dem_dir = '/home/danbot/Documents/code/hysets_validation/source_data/dem_data/processed_dem/'

# specify the DEM source
# either 'EarthEnv_DEM90' or 'USGS_3DEP'
DEM_source = 'EarthEnv_DEM90'
# DEM_source = 'USGS_3DEP'

def retrieve_and_preprocess_raster(region_code):
    # load the region DEM once and iterate through all
    # region_dem_path = os.path.join(processed_dem_dir, f'{region_code}_DEM_3005_{resolution}.tif')
    if DEM_source == 'EarthEnv_DEM90':
        region_dem_path = os.path.join(processed_dem_dir, f'{region_code}_{DEM_source}_3005_{resolution}.tif')
    else:
        region_dem_path = os.path.join(processed_dem_dir, f'{region_code}_DEM_3005_{resolution}.tif')
    assert os.path.exists(region_dem_path)

    rds = rxr.open_rasterio(region_dem_path, mask_and_scale=True, dtype=np.float32)    
    
    grid = Grid.from_raster(region_dem_path)
    dem = grid.read_raster(region_dem_path, dtype=np.float64)

    viewfinder = ViewFinder(affine=dem.affine, shape=dem.shape, crs=dem.crs, nodata=dem.nodata)

    dat = rds.data[0]
    
    raster = Raster(dat, viewfinder=viewfinder)
    return raster, rds


def get_river_mask(region_code, rds):
    nhn_grouped_vector_path = os.path.join(processed_data_dir, 'grouped_hydrographic_features/')
    vector_rivers_path = os.path.join(nhn_grouped_vector_path, f'{region_code}/NLFLOW/')

    # mask = create_mask(vector_rivers_path)
    rivers = gpd.read_file(vector_rivers_path, mask_and_scale=True)
    affine = rds.rio.transform(recalc=False)
    # Rasterize river shapefile
    river_raster = features.rasterize(rivers.geometry, out_shape=rds.shape[1:],
                                  transform=affine, all_touched=False)

    # Skeletonize river raster
    river_raster = skeletonize(river_raster).astype(np.uint8)

    # Create boolean mask based on rasterized river shapes
    mask = river_raster.astype(bool)
    return mask


# now create polygons using the raster just generated
def retrieve_raster(fpath):
    dem = rxr.open_rasterio(fpath, mask_and_scale=True, dtype=np.float32)
    crs = dem.rio.crs
    affine = dem.rio.transform(recalc=False)
    return dem, crs, affine


@jit(nopython=True)
def flatten_streams_windowed(dem, max_depth=1):

    n_adjustments = 0
    tot_adjustment = 0
    
    rows = dem.shape[0] # # of steps in y direction
    cols = dem.shape[1] # # of steps in x direction
    # print(f' dem shape = {dem.shape}')
    # create an array to track indices of all stream ends.
    stream_ends = np.empty((3,))
    stream_ends.fill(np.nan)
    # stream_nodes = np.empty((1,2))

    for i in range(rows):
        for j in range(cols):
            px_el = dem[i, j]

            # instead of iterating through the whole image,
            # just get a list of pixel indices (streams are sparse)
            if np.isfinite(px_el):
                # print(i, j, px_el)
                c1, c2 = max(0, j - max_depth), min(j + max_depth + 1, cols)
                r1, r2 = max(0, i - max_depth), min(i + max_depth + 1, rows)

                window = dem[r1:r2, c1:c2]
                # print('---------')
                # print(f'target cell ({i},{j}) el = {px_el:.1f}')
                # print(window)

                # the target cell is the centre of the flattened matrix
                # or different indices based on whether or not it's at an edge
                target_idx_coords, flat_index_loc = find_del_loc(i, j, window.shape[1], max_depth)

                # get the elevations surrounding the target pixel
                outer_vals = np.delete(window, flat_index_loc)
               
                # faster method of sorting when we just want the two smallest
                # and don't care about order
                two_smallest = np.partition(outer_vals, 2)[:2]

                # print(outer_vals)
                # print(np.count_nonzero(~np.isnan(outer_vals)))
                
                # print('')

                if np.count_nonzero(~np.isnan(outer_vals)) == 1:
                    # append the target index to track streamline terminus
                    stream_ends = np.append(stream_ends,(i, j, px_el))


                # print('outer vals and two smallest: ')
                # print(outer_vals, two_smallest)
                # if the centre pixel is higher or lower than both lowest
                # neighbours, set the pixel value to the average
                if np.isfinite(two_smallest).all():
                    if np.less(two_smallest, np.array(px_el)).all() | np.less(np.array(px_el), two_smallest).all():
                        new_el = np.mean(two_smallest)
                        dem[i, j] = new_el
                        n_adjustments += 1
                        tot_adjustment += px_el - new_el
                #         print('edited window')
                #         print(dem[r1:r2, c1:c2])
                #         print('')
                # print('')
                # print('_______')

    return dem, stream_ends, n_adjustments, tot_adjustment

@jit(nopython=True)
def find_del_loc(i, j, window_width, max_depth=1):   
    if i <= max_depth:
        ti = i
    else:
        ti = max_depth
    if j <= max_depth:            
        tj = j
    else:
        tj = max_depth
    # get the index of the target pixel
    # if the matrix is flattened
    flat_index = window_width * ti + tj
    # print(f'(ti, tj) = ({ti},{tj})')
    # print(f'flat_index: {flat_index}  {window_width}')
    return (int(ti), int(tj)), int(flat_index)


@jit(nopython=True)
def get_min_outer_pixel_indices(window, outer_vals, flat_index_loc, xs, ys, ix, jx, checked_indices, prevent_inf_loop, rows, cols, max_depth):            

    # print(f'outer vals: {outer_vals}')
    # get the index of the smallest outer value
    total_len = len(window.flatten())

    # print('outer vals: ', outer_vals)
    min_outer_val_idx = np.nanargmin(outer_vals)
    # print('min outer val idx: ', min_outer_val_idx)

    # the middle pixel was deleted from the outer_vals array,
    # so add 1 if the minimum index is in the back half of the 
    # array so we can reconstruct its 2d position
    if min_outer_val_idx >= flat_index_loc:
        min_outer_val_idx += 1

    # reconstruct the 2d indices of the outer pixel with min elevation
    # retrieve the row index that the min outer value falls in
    min_row_idx = int(np.floor(min_outer_val_idx / window.shape[1]))
    # retrieve the column index that the min outer value falls in
    min_col_idx = int(min_outer_val_idx - (min_row_idx) * window.shape[1])
    # test that we referenced the correct elevation.
    min_outer_el = window[min_row_idx, min_col_idx]

    # print(f'min idx ({min_row_idx},{min_col_idx}): el: {min_outer_el}')
    # print(f'(ix, jx)=({ix},{jx})')

    # convert the min neighbor's window index to dem index
    new_dem_ix = ix + (min_row_idx - max_depth)
    if ix < max_depth:
        new_dem_ix = ix + min_row_idx

    new_dem_jx = jx + (min_col_idx - max_depth)
    if jx < max_depth:
        new_dem_jx = jx + min_col_idx

    new_dem_idx = (new_dem_ix, new_dem_jx)

    # print('new dem idx: ', new_dem_idx)

    indices_idx = np.where(np.logical_and(xs == new_dem_idx[0], ys == new_dem_idx[1]))[0][0]

    if not np.any(np.in1d(checked_indices, indices_idx)):
        # print(f'outer val index: {min_outer_val_idx}')
        if min_outer_val_idx >= flat_index_loc:
            min_outer_val_idx -= 1
        
        # print('already checked cell. Update outer vals: ', outer_vals)
        outer_vals[min_outer_val_idx] = np.nan
        # print('already checked cell. Update outer vals: ', outer_vals)
        # print('are all outer vals nan?: ', np.all(np.isnan(outer_vals)))
        if np.all(np.isnan(outer_vals)):
            return new_dem_idx, min_outer_el, indices_idx, prevent_inf_loop, True
        else:
            new_dem_idx, min_outer_el, indices_idx, prevent_inf_loop, end_of_line = get_min_outer_pixel_indices(window, outer_vals, flat_index_loc, xs, ys, ix, jx, checked_indices, prevent_inf_loop, rows, cols, max_depth)
            prevent_inf_loop += 1
            if prevent_inf_loop >= 4:
                raise Exception; 'infinite loop!'


    return new_dem_idx, min_outer_el, indices_idx, prevent_inf_loop, False


def get_windows(raster, dem, ix, jx, rows, cols):
    i1, i2 = max(0, ix - 1), min(ix + 2, rows)
    j1, j2 = max(0, jx - 1), min(jx + 2, cols)
    # don't let window indices go beyond raster edges
    dem_window = dem[i1:i2, j1:j2]
    raster_window = raster[i1:i2, j1:j2]
    return raster_window, dem_window


def check_adj_slope_elevations(raster_window, dem_window, ix, jx, rows, cols):
    ta = time.time()
    # sometimes the stream vector will not line
    # up with the thalweg in the dem
    # look at surrounding (nan) cells and 
    # replace the target cell elevation with a lower
    # value if there is one (not on the headwater cell)
    pairs = [[1, 0], [0, 1], [1, 2], [2, 1]]
    if ix == 0: # top row
        if jx == 0: # top left
            pairs = [[1, 0], [0, 1]]
        elif jx == cols: # top right
            pairs = [[0, 0], [1, 1]]
        else: # top middle
            pairs = [[0, 0], [1, 1], [0, 2]]

    if ix == rows: # bottom row
        if jx == cols: # bottom right
            pairs = [[1, 0], [0, 1]]
        elif jx == 0: # bottom left
            pairs = [[0, 0], [1, 1]]
        else:
            pairs = [[1, 0], [0, 1], [1, 2]]

    nan_ixs = np.argwhere(np.isnan(raster_window)).tolist()
    ics = [e for e in nan_ixs if e in pairs]
    min_adjacent_el = 1E9
    if len(ics) > 0:
        els = [raster_window[ic[0], ic[1]] for ic in ics]
        if len(els) > 0:
            min_adjacent_el = min(els)

    return min_adjacent_el
    

def check_adj_stream_els(dem, ci, cj):
    stream_cells = np.argwhere(np.isfinite(dem))
    adj_px = [e for e in stream_cells if tuple(e) != (ci, cj)]
    # max_adj_idx = np.argmax(dem)
    adj_els = [dem[p[0], p[1]] for p in adj_px]
    if len(adj_els) > 0:
        return (min(adj_els), max(adj_els))
    else:
        return None


def travel_stream(raster, dem, indices, check, tot_adjustment, n_adjustments, max_depth=1):
    # don't check the first (headwater) cell
    headwater_cell_unchecked = False
    n_checks = 0
    while len(indices) > 0:
        check += 1
        # if check >= 10:
        #     break
        
        (ix, jx) = indices.pop()

        px_el = dem[ix, jx]
        px_el_og = px_el

        rows, cols = raster.shape[0], raster.shape[1]

        (ci, cj), flat_idx = find_del_loc(ix, jx, dem.shape[1])
        
        raster_window, dem_window = get_windows(raster, dem, ix, jx, rows, cols)

        min_adjacent_slope_el, next_idx = check_adj_slope_elevations(raster_window, dem_window, ix, jx, rows, cols)

        if not min_adjacent_slope_el:
            pass

        # print(f'current: {i}, el: {px_el:.1f} checked_indices', checked_indices)
        # checked_el[i] = px_el
            # if there is an outer pixel with a lower elevation
        # that isn't in the stream, change the current target 
        # elevation to a value slightly smaller
        if headwater_cell_unchecked & (min_adjacent_slope_el < px_el):
            # print(f'    Updating target cell el from {px_el:.1f} to {min_outer_el - 0.1:.1f}')
            dem[ix, jx] = min_adjacent_slope_el - 0.1
            headwater_cell_unchecked = False
            px_el = min_adjacent_slope_el - 0.1
            tot_adjustment +=  px_el_og - px_el
            n_adjustments += 1
            n_checks += 1

        neighbor_stream_cells = check_adj_stream_els(dem_window, ci, cj)

        headwater_cell_unchecked = True

    
    return dem, check, n_adjustments, tot_adjustment, n_checks


def find_and_sort_stream_cells(dem):
    # indices of stream elements
    nzidx = np.where(np.isfinite(dem))
    # ordered stream pixel indices by elevation descending
    el_ranking = np.argsort(dem[nzidx])[::-1]
    xs, ys = tuple(np.array(nzidx)[:, el_ranking])    
    stream_cell_indices = np.array(tuple(zip(xs, ys)))
    return stream_cell_indices


def find_end_cells(dem, raster, stream_cell_indices, adjust_dem=True):
    """Find all terminations of the stream network, 
    either headwaters or outlets. 

    Args:
        dem (_type_): _description_
        raster (_type_): _description_
        stream_cell_indices (_type_): _description_
        adjust_dem (bool, optional): _description_. Defaults to True.
    """
    # n_cells = len(stream_cell_indices)
    rows, cols = dem.shape[0], dem.shape[1]
    for (ix, jx) in stream_cell_indices:
        raster_window, dem_window = get_windows(raster, dem, ix, jx, rows, cols)
    

       
def flatten_streams_streamwise(raster, dem, max_depth=1):
    n_adjustments = 0
    tot_adjustment = 0

    # ta = time.time()
    # # stream_cell_indices = find_and_sort_stream_cells(raster, dem)
    # tb = time.time()
    # print(f'Time to sort indices: {tb-ta:.1e}s')

    # headwater_indices = find_headwater_cells(dem, raster, stream_cell_indices)

    # rows, cols = dem.shape[0], dem.shape[1]
    # checked_indices = np.empty(0)

    # print(dem)
    check = 0
    # dem, check, n_adjustments, tot_adjustment, n_checks = travel_stream(raster, dem, indices, tot_adjustment, n_adjustments, check)
    

    # print(dem)
    # print('')
    # print('')

    # the target cell is the centre of the flattened 3x3 matrix
    # or different indices based on whether or not it's at an edge
    # which makes 2x3 (left or right edge), 3x2 (top or bottom), 
    # or 2x2 (corners)
    # target_idx_coords, flat_index_loc = find_del_loc(ix, jx, window.shape[1], rows, cols, max_depth)

    # # print(f' delete flat index loc: {flat_index_loc}')

    # # get the elevations of STREAM PIXELS 
    # # surrounding the target pixel
    # outer_vals = np.delete(window, flat_index_loc)

    # prevent_inf_loop = 0
    # new_dem_idx, min_outer_el, indices_idx, prevent_inf_loop, end_of_line = get_min_outer_pixel_indices(window, outer_vals, flat_index_loc, xs, ys, ix, jx, checked_indices, prevent_inf_loop, rows, cols, max_depth)


    # # if we've reached the end of the current streamline
    # # skip the rest of the loop and go to the next highest index
    # if end_of_line:
    #     # print('')
    #     # print('END OF CURRENT STREAMLINE')
    #     # print('')
    #     # reset the headwater check status so we don't modify
    #     # the headwater cell at the start of the next stream
    #     headwater_cell_unchecked = False
    #     continue

    # indices = np.delete(indices, indices_idx, axis=0)
    # if i == n_cells-1:
    #     # print(f'before append: {indices[-5:]}')
    #     indices = np.append(indices, new_dem_idx)
    #     # print(f'new index appended: {indices[-5:]}')
    # else:
    #     indices = np.insert(indices, i+1, new_dem_idx, axis=0)
    # # track how the order of indices ends up being checked

    # if not np.any(np.in1d(checked_indices, indices_idx)):
    #     checked_indices = np.append(checked_indices, int(indices_idx)) 

    # if np.all(np.isnan(outer_vals)):
    #     continue

    # # if the minimum outer elevation is greater than
    # # the current pixel, change it to slightly less than 
    # # the current pixel
    # if min_outer_el >= px_el:
    #     # print(f'     outer px el reduced from {min_outer_el:.1f} to {px_el-0.1:.1f}')
    #     dem[new_dem_idx[0], new_dem_idx[1]] = round(px_el - 0.1, 1)
    #     tot_adjustment += px_el_og - round(px_el - 0.1, 1)
    #     n_adjustments += 1

    # # remove the updated cell from the main list 
    # # so we don't loop around it again
    # # find the min neighbor's index in the ordered indices array
    # el_new = dem[new_dem_idx]
      

    return dem, n_adjustments, tot_adjustment, n_checks


dir_method = 'D8' # D8, DINF
delineation_method = 'PYSHEDS'
# for region in code

resolution = 'res1'

dem_files = os.listdir(processed_dem_dir)
dem_files = [e for e in dem_files if resolution in e]


region_codes = sorted(list(set([e.split('_')[0] for e in dem_files])))

i = 1
for region_code in region_codes:
    print('___________________________________________________')
    print('')
    print(f'Starting stream burn on region {region_code} {i}/{len(region_codes)}.')
       
    # get the covering region for the station
    t_start = time.time()

    if 'EarthEnv' in DEM_source:
        resolution = 'res1'
    
    load_start = time.time()
    raster, rds = retrieve_and_preprocess_raster(region_code)
    load_end = time.time()
    print(f'    Time to load raster = {load_end-load_start:.1f}s')
    print(f'        --raster is {raster.shape[0]}px high by {raster.shape[1]}px wide')

    river_mask = get_river_mask(region_code, rds)
    mask_end = time.time()
    print(f'    Time to get river mask = {mask_end-load_end:.1f}s')

    # Blur mask using a gaussian filter
    blurred_mask = ndimage.filters.gaussian_filter(river_mask.astype(np.float64), sigma=2.5)

    # Set central channel to max to prevent pits
    blurred_mask[river_mask.astype(np.bool)] = blurred_mask.max()

    mask = blurred_mask

    # Create a view onto the DEM array
    # dem = grid.view('dem', dtype=np.float64, nodata=np.nan)
    # Set elevation change for burned cells

    dz = 6.5

    # n_px = 6

    masked_dem = np.empty_like(raster)
    masked_dem.fill(np.nan)
    # masked_dem = masked_dem.reshape()
    center = (int(raster.shape[0] / 2), int(raster.shape[1]*0.3))

    # dem = raster.view('dem', dtype=np.float64, nodata=np.nan)
    # mask the DEM by the river pixels
    masked_dem[river_mask > 0] = raster[river_mask > 0]

    t_start = time.time()
    
    # two-diagonal hump
    # test_dem = masked_dem[center[0]-10:center[0]+10, center[1]-70:center[1]-60].round(0)

    # # cut a sample window to test the algorithm
    # v1, v2 = -50, 50
    # h1, h2 = -50, 50
    # top_px, bot_px = max(center[0]+v1, 0), min(center[0]+v2, raster.shape[0])
    # left_px, right_px = max(center[0]+h1, 0), min(center[0]+h2, raster.shape[1])
    # test_dem = masked_dem[top_px:bot_px, left_px:right_px].round(1)
    # test_raster = raster[top_px:bot_px, left_px:right_px].round(1)

    # test_size = test_dem.shape[0] * test_dem.shape[1]
    # print(f'test raster size = ({test_dem.shape}) ({test_size:.1e}px)')

    # if (test_dem.shape == masked_dem.shape):
    #     print('***************************')
    #     print('Maximum test size reached.')
    #     print('***************************')
    # print(test_dem)
    modified_dem, stream_ends, n_adjustments, tot_adjustment = flatten_streams_windowed(masked_dem)

    # print('raster range: ', np.nanmin(raster),np.nanmax(raster))
    # print('min el in masked dem')
    # print(np.nanmin(masked_dem))

    # remove the first nan entry of a 3-tuple
    # stream_ends = stream_ends[3:]
    # stream_ends = stream_ends.reshape(int(len(stream_ends)/3),3)
    # n_ends = len(stream_ends)
    # print(f'Found {n_ends} stream ends.')
    # # sort the array by elevations (2nd index)
    # stream_ends = stream_ends[stream_ends[:,2].argsort()]

    # t_win = time.time()
    # raster_size = masked_dem.shape[0] * masked_dem.shape[1]
    # print(f'   {raster_size} px raster processed in {t_win-t_start:.2e}s')


    # modified_dem, n_adjustments, tot_adjustment, n_checks = flatten_streams_streamwise(test_raster, test_dem)
    # modified_dem, n_adjustments, tot_adjustment = flatten_streams_streamwise(raster, masked_dem)
    # print('')
    # t_end = time.time()

    
    # print(modified_dem)
    # print('')
    # print(f'{n_adjustments}/{test_size} cells adjusted in {n_checks} checks, total adjustment = {tot_adjustment:.1f}m')
    # print(f'   {test_size} px raster processed in {t_end-t_start:.2e}s')
    
    # raster_size = raster.shape[0] * raster.shape[1]
    # print(f'   Stream flattening completed in {t_end -t_start:.2e} for {raster_size:.2e} pixels. {n_adjustments} pixels adjusted, {tot_adjustment} m total adjustment.')
    
    # replace the river pixels in the original raster
    # with the modified values
    # raster[river_mask > 0] = modified_dem[river_mask > 0]

    # Subtract a constant dz from all stream pixels
    raster[river_mask > 0] -= dz

    out_fname = f'{region_code}_{DEM_source}_burned_streams_{resolution}.tif'
    processed_out_dir = f'/media/danbot/Samsung_T5/geospatial_data/DEM_data/processed_dem/'
    out_path = os.path.join(processed_out_dir, out_fname)

    rds.data[0] = raster.data
    # rds.rio.write_nodata(-32768, inplace=True)

    rds.rio.to_raster(out_path)#, nodata=-32768)
    # dem.to_raster(grid, out_path)
    
    t_end_cond = time.time()
    t_cond = t_end_cond - t_start
    print(f'    ...stream burn for {region_code} in {t_cond:.1f}.  Created {out_fname}')
    i += 1
    