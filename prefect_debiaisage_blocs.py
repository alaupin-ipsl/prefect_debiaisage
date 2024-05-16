"""Docstring"""

import os
from os import path
from pathlib import Path
import itertools
from enum import Enum
from typing import Literal

import asyncio

import numpy as np
import xarray as xr

import intake

import xesmf as xe
import cordex as cx
import cf_xarray as cfxr

from dask.distributed import Client
import dask_jobqueue as djq

from pydantic import BaseModel

from prefect import flow, task
from prefect.concurrency.sync import concurrency
from prefect.tasks import exponential_backoff

#########################
#########################
######             ######
######   CLUSTER   ######
######             ######
#########################
#########################

### Cluster configuration
# Specify if the cluster adapts the number of workers.
# Read adapt method documentation.
# Should be False for performance testing / configuration optimization <=> determinism.
# Should be True for production, as it play nicely on cluster.
IS_CLUSTER_ADAPTIVE = True
# Spirit and SpiritX: slurm
# Ciclad and Climserv: torque
CLUSTER_SCHEDULER = "slurm"
# Extra time to Wait so that the job are really ready.
JOB_READINESS_EXTRA_WAIT_TIME = 20  # Time in seconds.
DASK_DIRECTORY_PATH = path.join(os.getcwd(), "tmp_dask_workspace")

### Dask worker configuration
# Total number of workers availables for distributed computation.
NB_WORKERS = 8
if IS_CLUSTER_ADAPTIVE:
    NB_MIN_WORKERS = 1
else:
    NB_MIN_WORKERS = "not applied"
# In Gio (power of 2 ; not Go (power of 10) ; For Dask Gio == GiB != GB == Go).
WORKER_MEM = 16  # 4
# Number of threads per worker availables for parallel computation.
# Set more than 1 if your are sure to take advantage of parallelization.
NB_THREADS_PER_WORKER = 1

### Cluster job configuration
# Must be consistent with NB_WORKERS: NB_WORKERS = NB_WORKERS_PER_JOB * nb_jobs
# where nb_jobs is an integer.
NB_WORKERS_PER_JOB = 1
# Equivalent to worker lifespan if the cluster not adaptive.
JOB_WALLTIME = "08:00:00"
# Specify the network interface name that jobs use to communicate together (run ip a).
NET_INTERFACE = "ibs2"
# Specify where to write worker log files.
WORKER_LOG_DIRECTORY = path.join(DASK_DIRECTORY_PATH, "logs")
TMP_WORKER_DIRECTORY = path.join(DASK_DIRECTORY_PATH, "tmp")

### Infered worker configuration
# Insure worker specifications.
# Dask Jobqueue 0.7.3 is unable to handle this correctly...
worker_extra_opts = [
    f"--nthreads={NB_THREADS_PER_WORKER}",
    # f'--nprocs={NB_WORKERS_PER_JOB}',
    f"--memory-limit={WORKER_MEM}GiB",
]

### Infered job configuration
# Number of cluster job requested.
nb_jobs = int(np.ceil(NB_WORKERS / NB_WORKERS_PER_JOB))
# Number of CPU cores per job.
nb_cores_per_job = NB_THREADS_PER_WORKER * NB_WORKERS_PER_JOB
# Quantity of memory for a job.
job_mem = NB_WORKERS_PER_JOB * WORKER_MEM
job_vmem = job_mem
# Specific to Ciclad and Climserv.
# Dask Jobqueue 0.7.3 is unable to handle this correctly...
# Unnecessary for Spirit.
job_extra_opts = (
    f"-l mem={job_mem}gb",
    f"-l vmem={job_vmem}gb",
    f"-l nodes=1:ppn={nb_cores_per_job}",
)

print(
    f"""
> Configuration:

# Cluster configuration
- cluster adaptive: {IS_CLUSTER_ADAPTIVE}
- cluster scheduler: {CLUSTER_SCHEDULER}
- job extra waiting time: {JOB_READINESS_EXTRA_WAIT_TIME} seconds

# Worker configuration
- nb workers: {NB_WORKERS}
- min workers: {NB_MIN_WORKERS}
- memory per worker: {WORKER_MEM} Gio
- nb threads per worker: {NB_THREADS_PER_WORKER}

# Cluster job configuration
- nb jobs: {nb_jobs}
- nb cores per job {nb_cores_per_job}
- nb workers per job: {NB_WORKERS_PER_JOB}
- memory per job: {job_mem} Gio
- virtual memory per job: {job_vmem} Gio
- job walltime: {JOB_WALLTIME}
- job network interface: {NET_INTERFACE}
- log directory path: {WORKER_LOG_DIRECTORY}
"""
)


def launch_cluster():
    cluster = djq.SLURMCluster(
        cores=nb_cores_per_job,
        processes=NB_WORKERS_PER_JOB,
        memory=f"{job_mem}GiB",
        queue="zen16",  # interface=NET_INTERFACE, disable!
        walltime=JOB_WALLTIME,
        log_directory=WORKER_LOG_DIRECTORY,
        local_directory=TMP_WORKER_DIRECTORY,
        worker_extra_args=worker_extra_opts,
    )
    print(cluster.job_script())
    if IS_CLUSTER_ADAPTIVE:
        cluster.adapt(minimum=NB_MIN_WORKERS, maximum=NB_WORKERS)
        client = Client(cluster)
        print(f"> Waiting for {NB_MIN_WORKERS} worker(s) at least")
        # This instruction blocks until the number of ready jobs reaches the specified minimum.
        client.wait_for_workers(n_workers=NB_MIN_WORKERS)
    else:
        # Better control when scaling on jobs instead of workers.
        cluster.scale(jobs=nb_jobs)
        client = Client(cluster)
        print(f"> Waiting for {NB_WORKERS} worker(s)")
        # This instruction blocks until all jobs are ready.
        client.wait_for_workers(n_workers=NB_WORKERS)

    print(f"> The dashboard link: {cluster.dashboard_link}")

    return cluster, client


def close_cluster(cluster, client):
    client.shutdown()
    client.close()
    cluster.close()


########################
########################
######            ######
######   INPUTS   ######
######            ######
########################
########################


def get_inputs():
    # Catalog search
    catalog = "CORDEX"

    # Model search
    # variables = ["tasmax", "tasmin", "dtr", "pr"]
    experiments = ["historical", "rcp85"]
    variables = ["tasmax", "tasmin", "pr"]
    domains = ["AFR-44"]
    driving_models_list = [
        "NCC-NorESM1-M",
        "ICHEC-EC-EARTH",
        "IPSL-IPSL-CM5A-MR",
        "MIROC-MIROC5",
        "CCCma-CanESM2",
        "CSIRO-QCCCE-CSIRO-Mk3-6-0",
        "NOAA-GFDL-GFDL-ESM2M",
        "MPI-M-MPI-ESM-LR",
        "CNRM-CERFACS-CNRM-CM5",
    ]
    # driving_models_list = ['IPSL-IPSL-CM5A-MR', 'CCCma-CanESM2']
    institutes = ["SMHI"]
    ensemble = "r1i1p1"
    time_frequency = "day"

    # Regrid
    min_lon = -18
    max_lon = 52
    min_lat = -36
    max_lat = 38

    grid_path = None
    grid_lon_180_360 = None

    dict_regrid_algo = {
        "tasmax": "bilinear",
        "tasmin": "bilinear",
        "dtr": "bilinear",
        "pr": "conservative_normed",
    }
    dict_grid_model = {
        "tasmax": "CHIRPS_v2",
        "tasmin": "CHIRPS_v2",
        "dtr": "CHIRPS_v2",
        "pr": "CHIRPS_v2",
    }

    dict_inputs = {
        "catalog": catalog,
        "variables": variables,
        "variables_catalog": variables.copy(),
        "experiments": experiments,
        "domains": domains,
        "driving_models_list": driving_models_list,
        "institutes": institutes,
        "ensemble": ensemble,
        "time_frequency": time_frequency,
        "min_lon": min_lon,
        "max_lon": max_lon,
        "min_lat": min_lat,
        "max_lat": max_lat,
        "grid_path": grid_path,
        "grid_lon_180_360": grid_lon_180_360,
        "dict_regrid_algo": dict_regrid_algo,
        "dict_grid_model": dict_grid_model,
    }

    # return InterpolationInputs(dict_inputs)
    return dict_inputs


#########################
#########################
######             ######
######   CLASSES   ######
######             ######
#########################
#########################


class InputEnum(Enum):
    """
    Custom Enum class to add a method to check a whole list
    """

    @classmethod
    def from_list(cls, values):
        """Method to check the Enum for each element of a list"""

        return [cls.from_value(value) for value in values]

    @classmethod
    def from_value(cls, value):
        """Method to check the Enum for an element of the list"""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid value for {cls.__name__}")


class Variables(str, InputEnum):
    """
    Enum listing the variables used in the catalog search and interpolation methods
    """

    TASMAX = "tasmax"
    TASMIN = "tasmin"
    DTR = "dtr"
    PR = "pr"


class Experiments(str, InputEnum):
    """
    Enum listing the scenarios used in the interpolation
    """

    HISTORICAL = "historical"
    RCP26 = "rcp26"
    RCP45 = "rcp45"
    RCP85 = "rcp85"


class Domains(str, InputEnum):
    """
    Enum listing the cordex domains used in the catalog search
    """

    AFR44 = "AFR-44"
    EUR44 = "EUR-44"


class Models(str, InputEnum):
    """
    Enum listing the models to interpolate
    """

    NCC = "NCC-NorESM1-M"
    ICHEC = "ICHEC-EC-EARTH"
    IPSL = "IPSL-IPSL-CM5A-MR"
    MIROC = "MIROC-MIROC5"
    CCCMA = "CCCma-CanESM2"
    CSIRO = "CSIRO-QCCCE-CSIRO-Mk3-6-0"
    NOAA = "NOAA-GFDL-GFDL-ESM2M"
    MPI = "MPI-M-MPI-ESM-LR"
    CNRM = "CNRM-CERFACS-CNRM-CM5"


class Institutes(str, InputEnum):
    """
    Enum listing the institutes
    """

    SMHI = "SMHI"


class TimeFrequency(str, InputEnum):
    """
    Enum listing the time frequency of the model data
    """

    DAY = "day"
    TRIHOUR = "3hr"


class Bbox(BaseModel):
    min_lon: int
    max_lon: int
    min_lat: int
    max_lat: int


class GridLon180or360(str, InputEnum):
    """
    Enum listing the grid longitudes allowed by the XESMF Regridder
    """

    L180 = "180"
    L360 = "360"


class RegridAlgo(str, InputEnum):
    """
    Enum listing the regrid algos allowed by the XESMF Regridder
    """

    BILINEAR = "bilinear"
    CONSERVATIVE = "conservative"
    CONSERVATIVE_NORMED = "conservative_normed"
    PATCH = "patch"
    NEAREST_S2D = "nearest_s2d"
    NEAREST_D2S = "nearest_d2s"


class GridOutput(str, InputEnum):
    """
    Enum listing the available grids to regrid the model data on
    """

    CHIRPS_V2 = "CHIRPS_v2"
    ERA5 = "ERA5"


class InterpolationInput(BaseModel):
    catalog: Literal["CORDEX"]
    experiment: Experiments
    variable: Variables
    domain: Domains
    model: Models
    institute: Institutes
    ensemble: str
    time_frequency: TimeFrequency
    bbox: Bbox
    grid_path: str | None = None
    grid_lon_180_360: GridLon180or360
    dict_regrid_algo: RegridAlgo
    dict_grid_model: GridOutput


class InterpolationInputs(BaseModel):
    catalogs: list[Literal["CORDEX"]]
    experiments: list[Experiments]
    variables: list[Variables]
    domains: list[Domains]
    models: list[Models]
    institutes: list[Institutes]
    ensembles: list[str]
    time_frequencies: list[TimeFrequency]
    bboxes: list[Bbox]
    grid_paths: list[str] | None = None
    grid_lon_180_360: list[GridLon180or360]
    dict_regrid_algo: list[RegridAlgo]
    dict_grid_models: list[GridOutput]

    def to_input_list(self) -> list[InterpolationInput]:
        result = []
        for catalog in self.catalogs:
            for experiment in self.experiments:
                for variable in self.variables:
                    for domain in self.domains:
                        for model in self.models:
                            for institute in self.institutes:
                                for ensemble in self.ensembles:
                                    for time_frequency in self.time_frequencies:
                                        for bbox in self.bboxes:
                                            for grid_path in self.grid_paths:
                                                for grid_lon_180_360 in self.grid_lon_180_360:
                                                    for dict_regrid_algo in self.dict_regrid_algo:
                                                        for dict_grid_model in self.dict_grid_models:
                                                            result.append(
                                                                InterpolationInput(
                                                                    catalog=catalog,
                                                                    experiment=experiment,
                                                                    variable=variable,
                                                                    domain=domain,
                                                                    model=model,
                                                                    institute=institute,
                                                                    ensemble=ensemble,
                                                                    time_frequency=time_frequency,
                                                                    bbox=bbox,
                                                                    grid_path=grid_path,
                                                                    grid_lon_180_360=grid_lon_180_360,
                                                                    dict_regrid_algo=dict_regrid_algo,
                                                                    dict_grid_model=dict_grid_model,
                                                                )
                                                            )
        return result


#############################
#############################
######                 ######
######   VALIDATIONS   ######
######                 ######
#############################
#############################


@task
def validate_coords(min_lon, max_lon, min_lat, max_lat):
    """
    Complete sanity check of the input coordinates

    Inputs:
    min_lon -- Longitude of the lower-left corner of the bounding box
    max_lon -- Longitude of the upper-right corner of the bounding box
    min_lat -- Latitude of the lower left corner of the bounding box
    max_lat -- Latitude of the upper-right corner of the bounding box

    Outputs:
    boolean -- True if valid custom coordinates are provided, False if no custom coordinates are provided
    """

    list_coords = [min_lon, max_lon, min_lat, max_lat]
    # All the coordinates have been provided...
    if all(coord is not None for coord in list_coords):
        # The coordinates are numbers...
        if all(isinstance(coord, int) for coord in list_coords):
            # Minimum < Maximum
            if min_lon < max_lon and min_lat < max_lat:
                # Coordinates are outliers => ERROR
                if min_lon < -180:
                    print(f"""Wrong values for the bounding box coordinates, minimum longitude must be superior to -180 : 
                          Longitude={min_lon}-{max_lon}""")
                elif max_lon > 360:
                    print(f"""Wrong values for the bounding box coordinates, maximum longitude must be inferior to 360 : 
                          Longitude={min_lon}-{max_lon}""")
                elif min_lat < -90:
                    print(f"""Wrong values for the bounding box coordinates, minimum latitude must be superior to -90 : 
                          Latitude={min_lat}-{max_lat}""")
                elif max_lat > 90:
                    print(f"""Wrong values for the bounding box coordinates, maximum latitude must be inferior to 90 : 
                          Latitude={min_lat}-{max_lat}""")
                # Coordinates aren't outliers => OK
                else:
                    print("Custom coordinates input.")
                    return True
            else:
                # Min > Max for Longitude and Latitude => ERROR
                if min_lon > max_lon and min_lat > max_lat:
                    raise ValueError(
                        f"""Wrong values for the bounding box coordinates, minimum must be inferior to maximum : 
                        Longitude={min_lon}-{max_lon} Latitude={min_lat}-{max_lat}"""
                    )
                else:
                    # Min > Max for Longitude => ERROR
                    if min_lon > max_lon:
                        raise ValueError(
                            f"""Wrong values for the Longitude bounding box coordinates, minimum must be inferior to 
                            maximum : Longitude={min_lon}-{max_lon}"""
                        )
                    # Min > Max for Latitude => ERROR
                    elif min_lat > max_lat:
                        raise ValueError(
                            f"""Wrong values for the Latitude bounding box coordinates, minimum must be inferior to 
                            maximum : Latitude={min_lat}-{max_lat}"""
                        )
        # Wrong format for coordinates => ERROR
        else:
            raise ValueError(
                f"""Wrong values for the bounding box coordinates, only input numbers :
                Longitude={min_lon}-{max_lon} Latitude={min_lat}-{max_lat}"""
            )
    else:
        # Some coordinates are missing => ERROR
        if any(coord is not None for coord in list_coords):
            raise ValueError(
                f"""Missing coordinates for the bounding box :
                Longitude={min_lon}-{max_lon} Latitude={min_lat}-{max_lat}"""
            )
        # All the coordinates are missing => Use the CORDEX coordinates
        else:
            print("No custom coordinates input. Using CORDEX coordinates")
            return False


@task
def fill_dictionary_variables(target_dictionary, variables):
    """
    Fill the data dictionaries with missing variables and assign None as default value

    Inputs:
    target_dictionary -- Dictionary to check and fill
    variables -- List of the variables used in the interpolation

    Outputs:
    target_dictionary -- Checked and filled dictionary
    """

    for var in variables:
        if var not in list(target_dictionary.keys()):
            target_dictionary[var] = None
    return target_dictionary


@task
def check_inputs(dict_inputs):
    """
    Check the validity of all the inputs

    Inputs:
    dict_inputs -- Dictionary of all the inputs provided by the user and used in the interpolation process

    Outputs:
    custom_coordinates -- Boolean showing if a full set of custom coordinates were provided by the user or not
    """

    # General inputs
    Variables.from_list(dict_inputs["variables"])
    if (
        "dtr" in dict_inputs["variables_catalog"]
        and "tasmax" not in dict_inputs["variables_catalog"]
    ):
        dict_inputs["variables_catalog"].append("tasmax")
    if (
        "dtr" in dict_inputs["variables_catalog"]
        and "tasmin" not in dict_inputs["variables_catalog"]
    ):
        dict_inputs["variables_catalog"].append("tasmin")
    Experiments.from_list(dict_inputs["experiments"])
    Domains.from_list(dict_inputs["domains"])
    Models.from_list(dict_inputs["driving_models_list"])
    Institutes.from_list(dict_inputs["institutes"])
    TimeFrequency.from_value(dict_inputs["time_frequency"])
    print(dict_inputs["variables_catalog"])

    # Interpolation inputs
    custom_coordinates = validate_coords(
        dict_inputs["min_lon"],
        dict_inputs["max_lon"],
        dict_inputs["min_lat"],
        dict_inputs["max_lat"],
    )
    if dict_inputs["grid_lon_180_360"]:
        GridLon180or360.from_value(dict_inputs["grid_lon_180_360"])
    if dict_inputs["grid_path"] and not dict_inputs["grid_lon_180_360"]:
        raise ValueError(
            f"""Please specify if the grid provided for regridding have a 180° or 360° based longitude"""
        )
    RegridAlgo.from_list(list(dict_inputs["dict_regrid_algo"].values()))
    fill_dictionary_variables(dict_inputs["dict_regrid_algo"], dict_inputs["variables"])
    GridOutput.from_list(list(dict_inputs["dict_grid_model"].values()))
    fill_dictionary_variables(dict_inputs["dict_grid_model"], dict_inputs["variables"])

    return custom_coordinates


#########################
#########################
######             ######
######   CATALOG   ######
######             ######
#########################
#########################


def preprocess_noleap(df):
    """
    Convert a dictionary to a noleap calendar

    Inputs:
    df -- Dataframe with its own calendar format

    Outputs:
    df -- DataFrame converted to noleap calendar format
    """
    df = df.convert_calendar("noleap")
    return df


@task
def intake_search(dict_inputs, catalog_path, driving_model):
    cat = intake.open_esm_datastore(catalog_path)

    # Dataset search for input variables
    model_searched = cat.search(
        variable=dict_inputs["variables_catalog"],
        experiment=dict_inputs["experiments"],
        domain=dict_inputs["domains"],
        driving_model=driving_model,
        institute=dict_inputs["institutes"],
        ensemble=dict_inputs["ensemble"],
        time_frequency=dict_inputs["time_frequency"],
    )
    # ds_model = model_searched.to_dataset_dict(xarray_open_kwargs={"chunks": -1}, preprocess=preprocess_noleap)
    ds_model = model_searched.to_dataset_dict(xarray_open_kwargs={"chunks": -1})
    return ds_model


@task
def run_get_catalog(dict_inputs, driving_model):
    """
    Search in the catalog for all the inputs provided by the user and returns the requested datasets

    Inputs:
    dict_inputs -- Dictionary of all the inputs provided by the user and used in the interpolation process

    Outputs:
    dict_ds_cat -- Dictionary of all the requested datasets stored as dict_ds_cat[institute][domain][experiment]
    """
    dict_ds_cat = {}

    # cat = intake.open_esm_datastore(f'/modfs/catalogs/{dict_inputs["catalog"]}.json')
    ##cat = intake.open_esm_datastore("/ciclad-home/ltroussellier/dev/custom_cat/cat_alain.json")

    ##for driving_model in dict_inputs["driving_models_list"]:
    ##    print(f'  Starting {driving_model} model...')

    ## Dataset search for input variables
    # model_searched = cat.search(variable=dict_inputs["variables_catalog"],
    #                            experiment=dict_inputs["experiments"],
    #                            domain=dict_inputs["domains"],
    #                            driving_model=driving_model,
    #                            institute=dict_inputs["institutes"],
    #                            ensemble=dict_inputs["ensemble"],
    #                            time_frequency=dict_inputs["time_frequency"]
    #                            )
    # print(model_searched)
    # ds_model = model_searched.to_dataset_dict(xarray_open_kwargs={"chunks": -1}, preprocess=preprocess_noleap)
    catalog_path = f'/modfs/catalogs/{dict_inputs["catalog"]}.json'
    ds_model = intake_search(dict_inputs, catalog_path, driving_model)

    # Extraction of each dataset found and creation of the dictionary
    for institute in dict_inputs["institutes"]:
        dict_ds_cat[institute] = {}
        for domain in dict_inputs["domains"]:
            print(f"    Starting {domain} domain...")
            dict_ds_cat[institute][domain] = {}
            for experiment in dict_inputs["experiments"]:
                print(f"      Starting {experiment} experiment...")
                try:
                    ds_exp = ds_model[
                        f'{domain}.{institute}.{driving_model}.{experiment}.{institute}-RCA4.{dict_inputs["time_frequency"]}'
                    ]
                    dict_ds_cat[institute][domain][experiment] = ds_exp
                except KeyError:
                    print(
                        f'{domain}.{institute}.{driving_model}.{experiment}.{institute}-RCA4.{dict_inputs["time_frequency"]} dataset not found'
                    )

    return dict_ds_cat


###############################
###############################
######                   ######
######   INTERPOLATION   ######
######                   ######
###############################
###############################


def split_by_chunks(dataset):
    """
    Split the dataset in smaller datasets based on its chunks

    Inputs:
    dataset -- Full dataset

    Outputs:
    dataset[selection] -- Chunk of the dataset
    """
    chunk_slices = {}
    for dim, chunks in dataset.chunks.items():
        slices = []
        start = 0
        for chunk in chunks:
            if start >= dataset.sizes[dim]:
                break
            stop = start + chunk
            slices.append(slice(start, stop))
            start = stop
        chunk_slices[dim] = slices
    for slices in itertools.product(*chunk_slices.values()):
        selection = dict(zip(chunk_slices.keys(), slices))
        yield dataset[selection]


@task
def create_filepath(
    ds, prefix="filename", root_path="/homedata/alaupin/Data/TEST", experiment=""
):
    """
    Generate a filepath when given a xarray dataset

    Inputs:
    ds -- Dataset to save
    prefix -- First part of the filename, usually the variable name (default "filename")
    root_path -- Folder where the files will be stored (defaults "???")
    experiments -- RCP scenario to include in the filename (default "")

    Outputs:
    filepath -- Complete path of the file to be saved
    """
    if isinstance(ds.time.data[0], np.datetime64):
        start = np.datetime_as_string(ds.time.data[0], unit="D")
        end = np.datetime_as_string(ds.time.data[-1], unit="D")
    else:
        start = ds.time.data[0].strftime("%Y-%m-%d")
        end = ds.time.data[-1].strftime("%Y-%m-%d")

    if experiment:
        experiment = f"_{experiment}"

    filepath = f"{root_path}/{prefix}{experiment}_{start}_{end}.nc"
    return filepath


@task
def create_cordex_grid(domain_cordex):
    """
    Generate an input grid to be regridded based on the CORDEX domain request

    Inputs:
    domain_cordex -- Cordex domain of the requested domain, from the cordex library

    Outputs:
    ds -- Input grid based on the CORDEX domain
    """
    lat_corners = cfxr.bounds_to_vertices(
        domain_cordex.lat_vertices, "vertices", order="counterclockwise"
    )
    lon_corners = cfxr.bounds_to_vertices(
        domain_cordex.lon_vertices, "vertices", order="counterclockwise"
    )
    ds = domain_cordex.assign_coords(lon_b=lon_corners, lat_b=lat_corners).drop_vars(
        [vv for vv in domain_cordex.data_vars]
    )
    return ds


@task
def create_era5_grid(ll_lon, ur_lon, ll_lat, ur_lat):
    """
    Generate an ERA5 output grid for regridding, with a mask to keep only land data

    Inputs:
    ll_lon -- Longitude of the lower-left corner (= min_lon)
    ur_lon -- Longitude of the upper-right corner (= max_lon)
    ll_lat -- Latitude of the lower-left corner (= min_lat)
    ur_lat -- Latitude of the upper-right corner (= max_lat)

    Outputs:
    ds_era5 -- ERA5 Dataset with a land/non-land mask
    """
    # Load the land/sea mask for ERA5 and filter all the non-land data (lsm>0.5, determined empirically)
    ds_era5 = xr.open_dataset("/homedata/alaupin/era5_025_global_landseamask.nc")
    ds_era5 = ds_era5.drop_vars(["time"])
    ds_era5["mask"] = np.abs(ds_era5["lsm"]) > 0.5
    ds_era5["mask"] = ds_era5["mask"].isel(time=0)
    ds_era5 = ds_era5.drop_vars(["lsm"])

    # Convert the longitude to a 180° base, sort the longitude/latitude and crop to the requested area
    ds_era5 = ds_era5.assign_coords(
        {"longitude": (((ds_era5.longitude + 180) % 360) - 180)}
    )
    ds_era5 = ds_era5.reindex({"longitude": np.sort(ds_era5.longitude)})
    ds_era5 = ds_era5.reindex({"latitude": np.sort(ds_era5.latitude)})
    ds_era5 = ds_era5.sel(
        longitude=slice(ll_lon, ur_lon), latitude=slice(ll_lat, ur_lat)
    )

    return ds_era5


@task
def create_chirpsv2_grid(ll_lon, ur_lon, ll_lat, ur_lat):
    """
    Generate an ChirpsV2 output grid for regridding, with a mask to keep only land data

    Inputs:
    ll_lon -- Longitude of the lower-left corner (= min_lon)
    ur_lon -- Longitude of the upper-right corner (= max_lon)
    ll_lat -- Latitude of the lower-left corner (= min_lat)
    ur_lat -- Latitude of the upper-right corner (= max_lat)

    Outputs:
    ds_chirps -- ChirpsV2 Dataset with a land/non-land mask
    """
    # Load the ChirpsV2 mask with already non-land data filtered as nan
    ds_chirps = xr.open_dataset("/bdd/CHIRPSv2/p25/chirps-v2.0.1981.days_p25.nc")

    # Crop to the selected area and generated a "mask" variable based on the non-land data (= nan)
    ds_chirps = ds_chirps.sel(
        longitude=slice(ll_lon, ur_lon), latitude=slice(ll_lat, ur_lat)
    )
    ds_chirps = ds_chirps.drop_vars(["time"])
    ds_chirps["mask"] = xr.where(~np.isnan(ds_chirps["precip"].isel(time=0)), 1, 0)
    ds_chirps = ds_chirps.drop_vars(["precip"])

    return ds_chirps


@task
def create_custom_grid(dict_inputs, ll_lon, ur_lon, ll_lat, ur_lat, grid_path):
    """
    Generate a custom output grid for regridding from a provided file path, with NO land/sea mask

    Inputs:
    ll_lon -- Longitude of the lower-left corner (= min_lon)
    ur_lon -- Longitude of the upper-right corner (= max_lon)
    ll_lat -- Latitude of the lower-left corner (= min_lat)
    ur_lat -- Latitude of the upper-right corner (= max_lat)
    grid_path -- File path of the grid8 to load as output grid

    Outputs:
    ds_custom -- Custom Dataset with no land/sea mask
    """

    # Load the mask from the provided path
    ds_custom = xr.open_dataset(grid_path)
    ds_custom = ds_custom.drop_vars(["time"])

    # Convert a 360° based longitude grid to a 180° one
    if dict_inputs["grid_lon_180_360"] == "360":
        ds_custom = ds_custom.assign_coords(
            {"longitude": (((ds_custom.longitude + 180) % 360) - 180)}
        )

    # Sort the longitude and latitude and crop to the selected area
    lon_name = [dim for dim in ds_custom.coords if "lon" in dim][0]
    lat_name = [dim for dim in ds_custom.coords if "lat" in dim][0]
    ds_custom = ds_custom.reindex({lon_name: np.sort(ds_custom[lon_name])})
    ds_custom = ds_custom.reindex({lat_name: np.sort(ds_custom[lat_name])})
    ds_custom = ds_custom.sel(
        longitude=slice(ll_lon, ur_lon), latitude=slice(ll_lat, ur_lat)
    )

    return ds_custom


@task(
    tags=["xesmf"],
    retries=3,
    retry_delay_seconds=10,
)
def run_regrid(
    ds_input_cordex, ds_out, regrid_algo, periodic, reuse_weights, regridder_path
):
    print("Starting regrid")
    regridder_cordex_conserv = xe.Regridder(
        ds_input_cordex,
        ds_out,
        regrid_algo,
        periodic=periodic,
        reuse_weights=reuse_weights,
        filename=regridder_path,
    )
    print("Regrid ended")

    return regridder_cordex_conserv


@task
def run_prep_regrid(dict_ds_cat, dict_inputs, driving_model, custom_coordinates):
    """
    Regrid the input/CORDEX grid to the output/OBS grid

    Inputs:
    dict_ds_cat -- Dictionary of all the requested datasets stored as dict_ds_cat[institute][domain][experiment]
    dict_inputs -- Dictionary of all the inputs provided by the user and used in the interpolation process
    custom_coordinates -- Boolean showing if a full set of custom coordinates were provided by the user or not
    """

    dict_regrid = {}
    # dict_ds_cat Institute loop
    for institute, dict_ds_institute in dict_ds_cat.items():
        print(f"{institute}")
        dict_regrid[institute] = {}
        # dict_ds_cat Domain loop
        for domain, dict_ds_domain in dict_ds_institute.items():
            print(f" {domain}")
            dict_regrid[institute][domain] = {}
            # Extract the boundaries for the selected domain
            domain_cordex = cx.cordex_domain(domain, bounds=True)
            interpolated_domain = cx.domains.table[
                (cx.domains.table.index.str.startswith(domain))
                & (cx.domains.table.index.str.endswith("i"))
            ]

            # Used either the custom coordinates, or the CORDEX coordinates extracted from the requested domain
            # interpolated to a common North Pole based latitude-longitude grid (AFR44i when requesting AFR44)
            if custom_coordinates:
                ll_lon = dict_inputs["min_lon"]
                ur_lon = dict_inputs["max_lon"]
                ll_lat = dict_inputs["min_lat"]
                ur_lat = dict_inputs["max_lat"]
            else:
                if interpolated_domain.empty:
                    ll_lon = domain_cordex["lon"].isel(rlon=0, rlat=0).item()
                    ur_lon = domain_cordex["lon"].isel(rlon=-1, rlat=-1).item()
                    ll_lat = domain_cordex["lat"].isel(rlon=0, rlat=0).item()
                    ur_lat = domain_cordex["lat"].isel(rlon=-1, rlat=-1).item()
                else:
                    ll_lon = interpolated_domain.ll_lon.values[0]
                    ur_lon = interpolated_domain.ur_lon.values[0]
                    ll_lat = interpolated_domain.ll_lat.values[0]
                    ur_lat = interpolated_domain.ur_lat.values[0]

            # Create the input and output grids for the interpolation
            ds_input_cordex = create_cordex_grid(domain_cordex)
            dict_grids = {
                "CHIRPS_v2": create_chirpsv2_grid(ll_lon, ur_lon, ll_lat, ur_lat),
                "ERA5": create_era5_grid(ll_lon, ur_lon, ll_lat, ur_lat),
            }
            if dict_inputs["grid_path"]:
                dict_grids["custom"] = create_custom_grid(
                    dict_inputs,
                    ll_lon,
                    ur_lon,
                    ll_lat,
                    ur_lat,
                    dict_inputs["grid_path"],
                )

            # dict_ds_cat Experiment loop
            for experiment, ds_exp in dict_ds_domain.items():
                print(f"   Starting {experiment} experiment...")
                dict_regrid[institute][domain][experiment] = {}
                for variable in dict_inputs["variables"]:
                    print(f"        Starting {variable} variable...")
                    # Config interpolation parameters based on the selected variable
                    regrid_algo = dict_inputs["dict_regrid_algo"][variable]
                    grid_model = dict_inputs["dict_grid_model"][variable]

                    # Default interpolation parameters for precipitation variable
                    if variable == "pr":
                        if not regrid_algo:
                            regrid_algo = "conservative_normed"
                        if not grid_model:
                            grid_model = "CHIRPS_v2"
                    # Default interpolation parameters for generic variables
                    else:
                        # DTR is a computed variable from tasmax and tasmin
                        if "dtr" in dict_inputs["variables"]:
                            ds_exp["dtr"] = ds_exp["tasmax"] - ds_exp["tasmin"]
                        if not regrid_algo:
                            regrid_algo = "bilinear"
                        if not grid_model:
                            grid_model = "ERA5"

                    if dict_inputs["grid_path"]:
                        grid_model = "custom"

                    ds_out = dict_grids[grid_model]

                    # Set if the interpolation have to be periodic or not. Periodic interpolation must be used
                    # only with longitudinally global grids, and mustn't be used with conservative algorithms
                    if (
                        regrid_algo == "conservative"
                        or regrid_algo == "conservative_normed"
                    ):
                        periodic = False
                    else:
                        lon_name_in = [
                            dim for dim in ds_input_cordex.coords if "lon" in dim
                        ][0]
                        lon_name_out = [dim for dim in ds_out.coords if "lon" in dim][0]
                        lon_step_in = (
                            ds_input_cordex[lon_name_in]
                            .diff(lon_name_in)
                            .median()
                            .item()
                        )
                        lon_step_out = (
                            ds_out[lon_name_out].diff(lon_name_out).median().item()
                        )

                        if (
                            ds_input_cordex[lon_name_in].min().item() - lon_step_in
                            < -180
                            and ds_out[lon_name_out].min().item() - lon_step_out < -180
                            and ds_input_cordex[lon_name_in].max().item() + lon_step_in
                            > 180
                            and ds_out[lon_name_out].max().item() - lon_step_out > 180
                        ):
                            periodic = True
                        else:
                            periodic = False

                    # Set the path of the regridder, to load or save it to gain process time
                    if custom_coordinates:
                        regridder_path = f"/homedata/alaupin/regridder/{regrid_algo}_{domain}_to_{grid_model}_lon{ll_lon}-{ur_lon}_lat{ll_lat}-{ur_lat}_p25.nc"
                    else:
                        regridder_path = f"/homedata/alaupin/regridder/{regrid_algo}_{domain}_to_{grid_model}_{domain}_p25.nc"

                    # Regrid the input dataset to the output grid
                    if os.path.isfile(regridder_path):
                        reuse_weights = True
                    else:
                        reuse_weights = False

                    dict_regrid[institute][domain][experiment]["ds_input"] = (
                        ds_input_cordex
                    )
                    dict_regrid[institute][domain][experiment]["ds_output"] = ds_out
                    dict_regrid[institute][domain][experiment]["regrid_algo"] = (
                        regrid_algo
                    )
                    dict_regrid[institute][domain][experiment]["periodic"] = periodic
                    dict_regrid[institute][domain][experiment]["reuse_weights"] = (
                        reuse_weights
                    )
                    dict_regrid[institute][domain][experiment]["regridder_path"] = (
                        regridder_path
                    )

    return dict_regrid


@task(retries=3, retry_delay_seconds=15)
def run_interp(
    ds_exp,
    dict_inputs,
    driving_model,
    institute,
    domain,
    experiment,
    variable,
    regridder_cordex_conserv,
):
    """
    Regrid the input/CORDEX grid to the output/OBS grid

    Inputs:
    dict_ds_cat -- Dictionary of all the requested datasets stored as dict_ds_cat[institute][domain][experiment]
    dict_inputs -- Dictionary of all the inputs provided by the user and used in the interpolation process
    custom_coordinates -- Boolean showing if a full set of custom coordinates were provided by the user or not
    """
    print(
        f"Start - Model : {driving_model} | Institute : {institute} | Domain : {domain} | Experiment : {experiment} | Variable : {variable}"
    )
    # Clean the dataset's variables and coordinates
    if "dtr" in dict_inputs["variables"]:
        ds_exp["dtr"] = ds_exp["tasmax"] - ds_exp["tasmin"]
    ds_exp_var = ds_exp.copy()
    ds_exp_var = ds_exp_var.drop_vars(
        [
            dropped_var
            for dropped_var in list(ds_exp_var.keys())
            if dropped_var != variable
        ]
    )
    droppable_coords = ["height", "ensemble", "time_bnds", "rotated_pole"]
    dropped_coords = [var for var in droppable_coords if var in ds_exp_var.coords]
    data_interp_conserv = regridder_cordex_conserv(
        ds_exp_var.squeeze().drop_vars(dropped_coords),
        keep_attrs=True,
        output_chunks={"time": "auto", "latitude": -1, "longitude": -1},
    )

    # Save the datasets to files
    datasets = list(split_by_chunks(data_interp_conserv))

    folder_name = f"/homedata/alaupin/DATA/INTERPOLATED_MODELS/{institute}/{domain}/{driving_model}/p25"
    # if not os.path.isdir(folder_name):
    #    os.makedirs(folder_name)
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    paths = [
        create_filepath(
            ds, prefix=variable, root_path=folder_name, experiment=experiment
        )
        for ds in datasets
    ]

    xr.save_mfdataset(datasets=datasets, paths=paths)
    print(
        f"End - Model : {driving_model} | Institute : {institute} | Domain : {domain} | Experiment : {experiment} | Variable : {variable}"
    )


@task
def run_prep_grid(dict_inputs, driving_model, custom_coordinates):
    print(f"Starting {driving_model} model...")
    dict_ds_cat = run_get_catalog(dict_inputs, driving_model)
    dict_prep_regrid = run_prep_regrid(
        dict_ds_cat, dict_inputs, driving_model, custom_coordinates
    )
    return dict_ds_cat, dict_prep_regrid


@task
def run_interpolation_loop(dict_ds_cat, dict_regridders, dict_inputs):
    for driving_model, dict_ds_model in dict_ds_cat.items():
        print(driving_model)
        for institute, dict_ds_institute in dict_ds_model.items():
            print(institute)
            print(dict_ds_institute)
            for domain, dict_ds_domain in dict_ds_institute.items():
                print(domain)
                print(dict_ds_domain)
                for experiment, ds_exp in dict_ds_domain.items():
                    print(experiment)
                    print(ds_exp)
                    for variable in dict_inputs["variables"]:
                        print(variable)
                        run_interp.submit(
                            ds_exp,
                            dict_inputs,
                            driving_model,
                            institute,
                            domain,
                            experiment,
                            variable,
                            dict_regridders[driving_model][institute][domain][domain],
                        )


@flow(log_prints=True)
def run_prep_flow(inputs: InterpolationInputs = None):
    cluster, client = launch_cluster()
    try:
        dict_inputs = get_inputs()
        custom_coordinates = check_inputs(dict_inputs)

        dict_results = {}
        for driving_model in dict_inputs["driving_models_list"]:
            dict_results[driving_model] = run_prep_grid.submit(
                dict_inputs, driving_model, custom_coordinates
            )

        dict_ds_cats = {}
        dict_regrids = {}
        dict_regridders = {}
        for driving_model in dict_inputs["driving_models_list"]:
            dict_regridders[driving_model] = {}
            dict_ds_cats[driving_model], dict_regrids[driving_model] = dict_results[
                driving_model
            ].result()
            for institute, dict_ds_institute in dict_ds_cats[driving_model].items():
                dict_regridders[driving_model][institute] = {}
                print(institute)
                print(dict_ds_institute)
                for domain, dict_ds_domain in dict_ds_institute.items():
                    dict_regridders[driving_model][institute][domain] = {}
                    print(domain)
                    print(dict_ds_domain)
                    for experiment, ds_exp in dict_ds_domain.items():
                        print(experiment)
                        print(ds_exp)
                        print(dict_regrids)
                        print(driving_model)
                        print(institute)
                        print(domain)
                        print(experiment)
                        dict_prep_regrid = dict_regrids[driving_model][institute][
                            domain
                        ][experiment]
                        dict_regridders[driving_model][institute][domain][domain] = (
                            run_regrid(
                                dict_prep_regrid["ds_input"],
                                dict_prep_regrid["ds_output"],
                                dict_prep_regrid["regrid_algo"],
                                dict_prep_regrid["periodic"],
                                dict_prep_regrid["reuse_weights"],
                                dict_prep_regrid["regridder_path"],
                            )
                        )

        print(dict_regridders)

        run_interpolation_loop(dict_ds_cats, dict_regridders, dict_inputs)
        # print(dict_ds_cats[driving_model])
        # print(dict_regrids[driving_model])
        # for institute, dict_ds_institute in dict_ds_cats[driving_model].items():
        #    print(institute)
        #    print(dict_ds_institute)
        #    for domain, dict_ds_domain in dict_ds_institute.items():
        #        print(domain)
        #        print(dict_ds_domain)
        #        for experiment, ds_exp in dict_ds_domain.items():
        #            print(experiment)
        #            print(ds_exp)
        #            dict_prep_regrid = dict_regrids[driving_model][institute][domain][experiment]
        #            regridder_cordex_conserv = run_regrid(
        #                dict_prep_regrid["ds_input"],
        #                dict_prep_regrid["ds_output"],
        #                dict_prep_regrid["regrid_algo"],
        #                dict_prep_regrid["periodic"],
        #                dict_prep_regrid["reuse_weights"],
        #                dict_prep_regrid["regridder_path"]
        #            )
        #            run_interp.submit(dict_ds_cats[driving_model], dict_inputs, driving_model, regridder_cordex_conserv, driving_model)

        print("Interpolation successful !")
    finally:
        close_cluster(cluster, client)


if __name__ == "__main__":
    run_prep_flow.visualize()
