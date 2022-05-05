import os
import time

import pandas as pd
import numpy as np
import geopandas as gpd

from PIL import Image, ImageOps, ImageFont, ImageDraw

import shapely
from shapely.geometry import Point, mapping, Polygon

from functools import reduce

import holoviews as hv
import hvplot.pandas
hv.extension('bokeh', logo=False)

from multiprocessing import Pool

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

source_DEM = 'EarthEnv_DEM90' # EarthEnv_DEM90 or USGS_3DEP
source_DEM = 'USGS_3DEP'
# preprocess_method = 'BURNED' # filled or (stream) burned
preprocess_method = 'FILLED' # filled or (stream) burned
snap_method = 'SNAPMINMAX' # SNAPMIN, SNAPMINEX, SNAPMINMAX

target_folder = f'{source_DEM}_{preprocess_method}_{snap_method}'

date_ext = '20220502'

processed_basin_path = os.path.join(BASE_DIR, f'processed_data/processed_basin_polygons_{date_ext}/{target_folder}')

fig_folder = os.path.join(BASE_DIR, f'validate_hysets/overlay_figs/{target_folder}_{date_ext}') 
if not os.path.exists(fig_folder):
    os.mkdir(fig_folder)

mosaic_output_fname = f'{source_DEM}_{preprocess_method}_{snap_method}_collage_{date_ext}.png'

print('')

wsc_path = '/home/danbot/Documents/code/hysets_validation/source_data/WSC_data/WSC_basin_polygons/'
wsc_stns = os.listdir(wsc_path)

wsc_df = pd.read_csv('/home/danbot/Documents/code/hysets_validation/source_data/WSC_data/WSC_Stations_2020.csv')
stn_locs = [Point(row['Longitude'], row['Latitude']) for i, row in wsc_df.iterrows()]
wsc_gdf = gpd.GeoDataFrame(wsc_df, geometry=stn_locs, crs='EPSG:4326')
wsc_gdf = wsc_gdf.to_crs(3005)
wsc_gdf.set_index('Station Number', inplace=True)

data_dir = '/media/danbot/Samsung_T5/geospatial_data/'

hysets_dir = os.path.join(data_dir, 'HYSETS_data/')
# file containing derived watershed properties used in hysets
hysets_props_fname = 'HYSETS_watershed_properties.txt'

hysets_derived_basin_folder = '/media/danbot/T7 Touch/thesis_data/processed_stations/'

hysets_basins = gpd.read_file(hysets_dir + 'HYSETS_watershed_boundaries.zip')
hysets_basins = hysets_basins.set_crs('EPSG:4326')
hysets_basins = hysets_basins.to_crs('EPSG:3005')
hysets_basins.set_index('OfficialID', inplace=True)

hysets_station_locs = gpd.read_file('/home/danbot/Documents/code/hysets_validation/source_data/HYSETS_data/USGS_station_locations')
hysets_locs = stn_locs = [Point(row['LON_SITE'], row['LAT_SITE']) for i, row in hysets_station_locs.iterrows()]
hysets_locs_df = gpd.GeoDataFrame(hysets_station_locs, geometry=hysets_locs, crs='EPSG:4326')
hysets_locs_df = hysets_locs_df.to_crs(3005)
hysets_locs_df.set_index('SITE_NO', inplace=True)
# hysets_locs_df.head()

hysets_props_path = os.path.join(BASE_DIR, 'source_data/HYSETS_data/HYSETS_watershed_properties.txt')
hysets_properties = pd.read_csv(hysets_props_path, delimiter=';')


def get_overlay_plot(stn):
    wsc_comparision = True
    eenv_comparison = True
    if wsc_comparision:
        try:
            wsc_polygon_path = os.path.join(wsc_path, f'{stn}/DrainageBasin')
            wsc_polygon = gpd.read_file(wsc_polygon_path, layer=f'{stn}_DrainageBasin_BassinDeDrainage')
            wsc_polygon = wsc_polygon.to_crs(3005)

            wsc_pp_path = os.path.join(wsc_path, f'{stn}/PourPoint')
            wsc_pp = gpd.read_file(wsc_pp_path, layer=f'{stn}_PourPoint_PointExutoire', crs='4326')
            wsc_pp = wsc_pp.to_crs(3005)

            wsc_stn_loc = wsc_gdf[wsc_gdf.index == stn]
            wsc_found = True
        except Exception as e:
            print(f'   no WSC delineation found for {stn}.')
            wsc_found = False
    
    hysets_polygon = hysets_basins[hysets_basins.index == stn]
    baseline_area = hysets_polygon.copy().to_crs(3005).geometry.area.values[0] / 1E6
        

    val_path = os.path.join(processed_basin_path, f'{stn}_{source_DEM}_basin.geojson')
    
    val_polygon = gpd.read_file(val_path, driver='GeoJSON')
    val_polygon = val_polygon.to_crs(3005)
    derived_area = val_polygon.copy().geometry.area.values[0] / 1E6
    area_coverage = 100 * (derived_area / baseline_area)
    # val_plot = val_polygon.hvplot(alpha=0.3, color='firebrick')

    if area_coverage > 200:
        area_coverage = 2  
    
    loc_is_centroid = False
    if stn in hysets_locs_df.index:
        print(f'    {stn} location is in HYSETS.')
        stn_loc = hysets_locs_df[hysets_locs_df.index == stn]
    elif stn in wsc_gdf.index:
        print(f'    {stn} location is in WSC.')
        stn_loc = wsc_gdf[wsc_gdf.index == stn]
    else:
        print(f"    Can't find station loc for {stn}")
        basin = hysets_basins[hysets_basins.index == stn].centroid
        # centroid = Point(basin['
        stn_loc = gpd.GeoDataFrame(geometry=basin, crs=hysets_locs_df.crs)
        loc_is_centroid = True
        
    hysets_data = hysets_properties[hysets_properties['Official_ID'] == stn]
    area_flag = 'NOFLAG'
    if hysets_data['Flag_GSIM_boundaries'].values[0]:
        area_flag = 'GSIMFLAG'
    if hysets_data['Flag_Artificial_Boundaries'].values[0]:
        area_flag = 'ABFLAG'
        # if the HYsets polygon is artificial (square), 
        # use the published WSC as the baseline
        if wsc_found:
            hysets_polygon = wsc_polygon
    
   
    tp_polygon = gpd.overlay(val_polygon, hysets_polygon, how='intersection').dissolve(aggfunc='sum')
    fp_polygon = val_polygon.overlay(hysets_polygon, how='symmetric_difference').dissolve(aggfunc='sum')
    fn_polygon = val_polygon.overlay(hysets_polygon, how='difference').dissolve(aggfunc='sum')
    
    plots = []
    tp_exists, fp_exists = False, False
    if tp_polygon.empty:
        tpa = 0
    else:
        tpa = 100 * ((tp_polygon.geometry.area.values[0]/1E6) / baseline_area)
        tp_polygon = tp_polygon.to_crs(3857)
        tp_plot = tp_polygon.hvplot(
            tiles='OSM', width=900, height=600,
            alpha=0.7, color='#94f024', line_width=0)
        tp_plot.border_fill_color = None  
        tp_plot.background_fill_color = None
        plots.append(tp_plot)
        tp_exists = True
        
    if fp_polygon.empty:
        fpa = 100
    else:
        fpa = 100 * ((fp_polygon.geometry.area.values[0]/1E6) / baseline_area)    
        fp_polygon = fp_polygon.to_crs(3857)
        fp_exists = True
        if tp_exists:
            fp_plot = fp_polygon.hvplot(
                alpha=0.7, color='#8554ff', line_width=0)
        else:
            fp_plot = fp_polygon.hvplot(
                tiles='OSM', width=900, height=600,
                alpha=0.7, color='#8554ff', line_width=0)
            fp_plot.border_fill_color = None  
            fp_plot.background_fill_color = None
        plots.append(fp_plot)
        
    if fn_polygon.empty:
        fna = 100
        # fn_polygon = gpd.GeoDataFrame(geometry=[])
    else:
        fna = 100 * ((fn_polygon.geometry.area.values[0]/1E6) / baseline_area)
        fn_polygon = fn_polygon.to_crs(3857)
        if tp_exists | fp_exists:
            fn_plot = fn_polygon.hvplot(
                # tiles='OSM', width=900, height=600,
                alpha=0.7, color='#ff5444', line_width=0)
        else:
            fn_plot = fn_polygon.hvplot(
                tiles='OSM', width=900, height=600,
                alpha=0.7, color='#ff5444', line_width=0)
            fn_plot.border_fill_color = None  
            fn_plot.background_fill_color = None
            
        plots.append(fn_plot)   
    
    print(f'            TP: {tpa:.0f}  FP: {fpa:.0f} FN: {fna:.0f}')

    
    # create plots
    stn_loc = stn_loc.to_crs(3857)
    plot_title = f'{stn} {baseline_area:.0f} km2 {area_flag} TP{tpa:.0f} FP{fpa:.0f} FN{fna:.0f}'
    
    if loc_is_centroid:
        marker = 'asterisk'
    else:
        marker = 'star'
    
    stn_loc_plot = stn_loc.hvplot(title=plot_title, 
                          # tiles='OSM', width=900, height=600,
                          xaxis=None, yaxis=None, tools=[],
                          marker='star', size=300, line_width=2,
                          fill_color='deepskyblue', line_color='dodgerblue')
    
    stn_loc_plot.background_fill_color = None
    stn_loc_plot.border_fill_color = None  
    
    if wsc_found:
        wsc_polygon = wsc_polygon.to_crs(3857)
        wsc_plot = wsc_polygon.hvplot(
            # tiles='OSM', width=900, height=600,
            line_dash='dashed', line_color='grey', 
            fill_color=None, line_width=2,  
            )
        wsc_pp = wsc_pp.to_crs(3857)
        pp_loc_plot = wsc_pp.hvplot(
            marker='circle', size=300, fill_color='blue', line_color='yellow')
        
    if wsc_found:
        layout =  reduce(lambda x, y: x*y, plots) * wsc_plot * stn_loc_plot 
    else:
        layout =  reduce(lambda x, y: x*y, plots) * stn_loc_plot
            
    filename = f'{stn}_{baseline_area:.2f}KM2_{area_flag}_TP{tpa:.0f}_FP{fpa:.0f}_FN{fna:.0f}.png'
    save_path = os.path.join(fig_folder, filename)
    hvplot.save(layout, save_path)
    return True


files_created = os.listdir(fig_folder)


stations = [e.split('_')[0] for e in os.listdir(processed_basin_path)]

existing_images = os.listdir(fig_folder)
completed_stations = [e.split('_')[0] for e in existing_images]

stations_to_process = [e for e in stations if e not in completed_stations]

t_start = time.time()
p = Pool()
p.map(get_overlay_plot, stations_to_process)
t_end = time.time()
print(f' Processed {len(stations)} images in {t_end-t_start:.1f}s')

existing_images = os.listdir(fig_folder)
d = {}
for f in existing_images:
    
    if 'Legend' not in f:
        data = f.split('_')
        stn = data[0]
        area = data[1].split('K')[0]
        flag = data[2]
        tp = data[3][2:]
        fp = data[4][2:]
        fn = data[-1].split('.')[0][2:]
        
        d[stn] = {
            'area': area,
            'flag': flag,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
        }


def equal_bins(x, nbin=30):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))


df = pd.DataFrame(d).T
df.index.name = 'stn'

df['col'] = np.nan
df['row'] = np.nan

df['tp'] = df['tp'].astype(float)
df['area'] = df['area'].astype(float)

def get_binned_vals(param):
    bins = equal_bins(df[param])
    max_len = 0
    for i in range(1, len(bins)):
        left = bins[i-1]
        right = bins[i]
        in_bin = df[(df[param] >= left) & (df[param] < right)]
        if len(in_bin) >  max_len:
            max_len = len(in_bin)
    for i in range(1, len(bins)):
        left = bins[i-1]
        right = bins[i]
        print(f'{left:.0f} - {right:.0f}')
        in_bin = (df[param] >= left) & (df[param] < right)
        df.loc[in_bin, 'col'] = i
        row_vals = df.loc[in_bin, 'tp'].rank(ascending=False,  method='first')
        df.loc[in_bin, 'row'] =  max_len - row_vals
        
    return df, bins

df, eq_bins = get_binned_vals('area')

df['path'] = df.apply(lambda row: [os.path.join(fig_folder, e) for e in existing_images if row.name in e][0], axis=1)

height = df['row'].max()
width = df['col'].max()

img_width = 900
img_height = 600

collage = Image.new('RGB', (int(width*900), int(height*600) + 20), (255,255,255))
for y1 in range(0, int(height)):
    for x1 in range(0, int(width)):
        paste_img_path = df[(df['col'] == x1+1) & (df['row']== y1+1)]['path'].values
        n_imgs = len(paste_img_path)
        if n_imgs ==  0:
            pass
        elif n_imgs == 1:    
            paste_img = Image.open(paste_img_path[0])
            collage.paste(paste_img, (int(x1*img_width), int(y1*img_height)))
        else:
            print(' too many images returned')


legend_img = Image.open('overlay_figs/00-Legend.png')
collage.paste(legend_img, (x1*img_width, 0))

mosaic_path = os.path.join(BASE_DIR, f'validate_hysets/overlay_figs/{mosaic_output_fname}') 
collage.save(mosaic_path)

print(f'    Created image mosaic: {mosaic_output_fname}')
print('######################')



