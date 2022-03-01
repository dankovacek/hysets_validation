Data Acquisition and Preprocessing
==================================

Set up Cloud Computing
----------------------

(Or local machine if it’s beefy enough, &gt;=128GB ram required!).
**Assumes installation on Ubuntu Linux because it’s commonly offered in
cloud compute services.** These instructions are intended to minimize
setup time and costs associated with hourly compute resources (~$1-3 per
hour on [DigitalOcean](https://www.digitalocean.com/pricing/calculator/)
for 128 GB RAM, 12 core processor.)

First update:  
&gt;`sudo apt-get update`

Clone the repo:  
&gt;`git clone https://github.com/dankovacek/hysets_validation`

### Install required software

If not automatically installed, install Python and virtualenv:

> `sudo apt install python3 python3.8-venv pip`

Create Python 3.8+ virtual environment at the root level directory:

> `python3 -m venv /env/`

**GDAL**

> `sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update && sudo apt-get upgrade`

Install software utilities:
&gt;`sudo apt-get install parallel unzip dos2unix`  
&gt;`sudo apt-get install libgdal-dev`
&gt;`sudo apt-get install python3-gdal` <!-- gdal-bin  -->

Export environment variables:  
&gt;`export CPLUS_INCLUDE_PATH=/usr/include/gdal`  
&gt;`export C_INCLUDE_PATH=/usr/include/gdal`

Activate the virual environment:  
&gt;`source env/bin/activate`

Install python GDAL bindings:  
&gt;`pip install GDAL`

Install Python packages:  
&gt;`pip install geopandas whitebox shapely rioxarray`

Create folders and copy files using `scp` command:

> `scp setup_scripts/ account@1XX.XX.XXX.XX`. Here, `account@1XX.XX...`
> is the account name and IP address of the machine. You can specify a
> different path by adding a colon after the IP address (:),
> i.e. `account@XX.XX.XX:home/custom_path/data/`. The path must first be
> created on the destination machine.

Download Source Data
--------------------

### DEM Data (from USGS)

The folder `setup_scripts/` contains two files
(`1-1arcsecond-PNW-files.txt`, `2-2arcsecond-files.txt`) of links to DEM
data covering the study area provided by the USGS. Unfortunately the
1-arcsecond set contains several small gaps near the Yukon-Alaska
border, so we need to import the 2 arcsecond data to fill in the gaps.
We can combine the two sources and specify the highest resolution where
available automatically using gdal.

Make source directories for DEM data and raw files. At the root
directory level:

> `mkdir source_data/DEM_data/ source_data/DEM_data/dem_files`

> `cd setup_scripts/file_lists/`

Merge the filenames into a single file:

> `cat 1-1arcsecond-PNW-files.txt 2-2arcsecond-files.txt > merged-files.txt`

You’ll likely have a carriage return added to urls (%0D) that needs to
be removed (Windows issue that seems to pop up in cloud instance):
&gt;`dos2unix merged-files.txt`

and download files in parallel (~ 18GB total. Check what folder you are
in when executing, should be in `hysets_validation/`):

> `cat setup_scripts/file_lists/merged-files.txt | parallel --gnu "wget -P source_data/dem_data/dem_files/ {}"`

Create a mosaic (.vrt) of the DEM files covering the Pacific North West
called `BC_DEM_mosaic.vrt`. This is the main DEM index that will be
called in the analysis. Note that the vrt srs will be in EPSG 4269 and
will have to be reprojected during processing.

> `cd source_data/dem_data/`  
> `gdalbuildvrt -resolution highest BC_DEM_mosaic.vrt dem_files/USGS_1_*.tif`

### HYSETS

From the root directory, create a directory to store the HYSETS study
data (~ 14.6 GB. *Note: update the filename in the unzip command.*):

> `mkdir source_data/HYSETS_data/`  
> `curl -O https://files.osf.io/v1/resources/rpc3w/providers/googledrive/?zip=HYSETS_2020.zip > HYSETS_2020_data.zip`  
> `unzip HYSETS_2020_data.zip`

This file is particularly large, so open a second ssh connection and
continue while this file is downloading.

**GLHYMPHS**

For porosity and permeability, HYSETS used the dataset from GLobal
HYdrogeology MaPS (Gleeson et al. 2014).

> `mkdir source_data/GLHYMPHS_data/`

You can’t use wget here unfortunately. Not straightforward to
auto-download because of license modal. Visit the link below, download,
and transfer to the data drive (~1.1 GB):

> `https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/DLGXYO`

**NALCMS**

Land use percentages (forest, grassland, crops, etc.) is derived from
the North American Land Change Monitoring System:

> `cd source_data/`  
> `mkdir NALCMS_data/`  
> `wget http://www.cec.org/wp-content/uploads/wpallimport/files/Atlas/Files/2010nalcms30m/north_america_2010.zip`
> `unzip north_america_2020.zip -d .`

### National Hydrographic Network (NHN)

The [national blue-line
network](https://open.canada.ca/data/en/dataset/a4b190fe-e090-4e6d-881e-b87956c07977)
is a vector-based representation of surface water in Canada. The index
contains 1341 polygons representing “Work Unit Limits” (WLU) based on
sub-sub drainage areas. Here we will filter the polygons to find all
those intersecting with the polygon describing the BC provincial border,
and then group the work units into larger regions representing the major
drainage basins covering the province. The purpose of merging into
larger groups is to avoid having polygons break up continuous rivers, as
the subsequent step is to calculate flow accumulation networks for all
basins covering the province.

Map the group basins into the minimum covering set. Use the script found
in `setup_scripts/`

> `python process_hydrologic_regions.py`

The script takes the sub-sub-drainage polygons and converts them to
hydraulically separate regions, in other words polygons do not cut flow
paths anywhere in the province. See the images below for original and
processed data polygons:

![Original WSC sub-sub-drainage regions](img/wsc-ssda.png) ![Grouped
Major Drainage Basins after processing](img/wsc-ssda-processed.png)

Once the major groups are formed, the file
`/processed_data/BC_basin_region_groups.geojson` should have been
created. For each of the regional basins, create a clipped DEM and
reproject to EPSG:3005.  
&gt;`cd setup_scripts/`  
&gt;`python dem_basin_mapper.py`

The NHN also contains many hydrologic features in detail, provided in
shape files by WLU, as described in the documentation linked above:

> *“It provides geospatial digital data compliant with the NHN Standard
> such as lakes, reservoirs, watercourses (rivers and streams), canals,
> islands, drainage linear network, toponyms or geographical names,
> constructions and obstacles related to surface waters, etc.”*

Data such as obstacles will be important for catchment delineation, as
obstacles such as bridges can interfere with the flow direction and
accumulation steps.

To access individual WLU attribute objects, use the following
convention:

> `https://ftp.maps.canada.ca/pub/nrcan_rncan/vector/geobase_nhn_rhn/shp_en/<XX>/<filename>`

Where `<XX>` is the two-digit major drainage area (MDA) prefix (BC is
covered by 07, 08, 09, and 10). `<filename>` is for example
`nhn_rhn_07aa000_shp_en.zip`.

The feature files related to the WLU groups in the set covering BC were
downloaded and saved to the `processed_data/` folder when the
`process_hydrologic_regions.py` script was executed above.

An example of the features contained in these files is shown below
(shown imported in [QGIS](https://qgis.org/en/site/)):

![Example hydrologic features from NHN (from
QGIS)](img/nhn-features.png)

<!-- ### BC Extreme Flood Project (OPTIONAL)

Download basin polygons and basin metadata from the BCEF study.  From the root directory, create a folder:

>`mkdir source_data/BCEF_data/`  
>`cd source_data/BCEF_data`

Get the basin polygons used in the study:
>`wget https://www2.gov.bc.ca/assets/gov/environment/air-land-water/water/dam-safety/bulletin_2020-1-rffa_nhc_all_watershedsr0.zip`

Get the station / basin metadata used in the study:
>`wget https://www2.gov.bc.ca/assets/gov/environment/air-land-water/water/dam-safety/bulletin_2020-1-rffa_station_metadatar0.csv` -->

### WSC Hydrometric Station Catchment Polygons and Metadata

Updated set of basin polygons from WSC, published in December 2021.

> `mkdir source_data/WSC_data` `cd source_data/WSC_data`
> `wget https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/HydrometricNetworkBasinPolygons/07.zip`
> `wget https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/HydrometricNetworkBasinPolygons/08.zip`
> `wget https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/HydrometricNetworkBasinPolygons/09.zip`
> `wget https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/HydrometricNetworkBasinPolygons/10.zip`

The file containing currently publicly available WSC station basin
polygons can be retrieved by the command below. The collection is less
complete than the set available above, however the set above is not
finalized/approved and may be subject to revision.

> `cd source_data/WSC_data/`
> `wget -P source_data/  http://donnees.ec.gc.ca/data/water/products/national-hydrometric-network-basin-polygons/WSC_Basins.gdb.zip`

The `source_data` folder contains the WSC station metadata file of
active and historic hydrometric stations `WSC_Stations_2020.csv`.

Merge all files into one zip and create a geojson file – geopandas can
read zip files of polygons:

> `mkdir all` `for x in *.zip; do unzip -d all -o -u $x ; done`
> `zip -r WSC_basins.zip all` `cd all/`
> `for dir in */; do mkdir -- "$dir"{basin,pour_point,station}; done`
> `for dir in */; do mv "$dir"/*DrainageBasin* "$dir"/basin`
> `for dir in */; do mv "$dir"/*PourPoint* "$dir"/pour_point`
> `for dir in */; do mv "$dir"/*Station* "$dir"/station`

Create geojson objects as separate data structures of all basins:
&gt;`python process_wsc_basins.py`

Basin Delineation and Attribute Validation
------------------------------------------

<!-- Copy the processed files back to local machine and proceed to validate HYSETS basin attributes.  Note: the scope of this step may have to be reduced to include basins within some limited size to limit computation. -->

This step represents the heavy lifting where large regions such as the
Liard River and Fraser River basins DEM are processed into flow
direction and flow accumulation at the highest available resolution.

> `python process_dem_by_basin.py`

Note: the breach [depression
function](https://jblindsay.github.io/ghrg/Whitebox/Help/BreachDepressions.html)
run on the DEM is a bottleneck step.

The final step is to validate the basin attributes derived in HYSETS (or
other dataset) using the set of stations whose catchment boundaries
intersect BC. The manual basin delineation step is the most
computationally intensive step of the validation process, and it’s
executed with the script `derive_basin_polygons.py`

> `python derive_basin_polygons.py`

Extras
------

Example to automate citations

> `pandoc -t markdown_strict -citeproc README-draft.md -o README.md --bibliography bib/bibliography.bib`

### Global River Classification (OPTIONAL)

https://ln.sync.com/dl/3d4952ac0/isuvquck-82fv4ca6-2vxh8nwm-fpeg9mra

Gleeson, Tom, Nils Moosdorf, Jens Hartmann, and LPH Van Beek. 2014. “A
Glimpse Beneath Earth’s Surface: GLobal Hydrogeology Maps (Glhymps) of
Permeability and Porosity.” *Geophysical Research Letters* 41 (11):
3891–8.
