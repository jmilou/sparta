# -*- coding: utf-8 -*-

"""
This script reads the reduced IRDIS files in the offline, and computes the 
contrast on the 

Author: J. Milli
Creation date: 2019-05-01
"""

import argparse
from astropy.io import fits
from pathlib import Path, PosixPath
import sys 
import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import date as dtdate
from datetime import timedelta 
import pandas as pd
#from astropy.time import Time
#from datetime import timedelta #datetime
#import matplotlib.gridspec as gridspec 
#import matplotlib as mpl
#from scipy.interpolate import interp1d
#import subprocess
#from astropy import units as u
#from astropy import coordinates
#import getopt
#from astropy.utils.exceptions import AstropyWarning
#import warnings
#warnings.filterwarnings("ignore",category=UserWarning)
#warnings.simplefilter('ignore',category=AstropyWarning)
#from astropy.coordinates import SkyCoord
#from astropy.coordinates import Galactic, FK5
#import pdb # for debugging purposes

  
# Definition of the UT3 coordinates
#latitude =  -24.6268*u.degree
#longitude = -70.4045*u.degree  
#altitude = 2648.0*u.meter 
#location = coordinates.EarthLocation(lon=longitude,lat=latitude,height=altitude)

def compute_contrast_sphere_offline(path_raw,path_reduced,path_output,debug=True):
    """
    Function that reads the reduced files in the offline machine, select the 
    IRDIS DBI files (left and right cubes), select the off-axis PSF (FLUX), 
    and compute the contrast. 
    """
    try:
        if not type(path_raw) is PosixPath:
            path_raw = Path(path_raw)
        if type(path_reduced) is PosixPath:
            path_reduced = Path(path_reduced)
        if not type(path_raw) is PosixPath:
            path_output = Path(path_output)        
        for path in [path_raw,path_reduced]:
            if not path.exists():
                print('The input path {0:s} does not exists. Returning'.format(str(path)))
                return
        if not path_output.exists():
            path_output.mkdir()
    except:
        print(sys.exc_info()[0])
    
    
    #%% Step 0 : 
    # Load all raw frames and list the original filename, along with the type 
    # and archive names

    raw_files = []
    raw_archive_names = []
    raw_dpr_types = []
    
    files_sphere_raw = sorted(path_raw.glob('SPHERE*IRDIS*OBS*fits'))
    for file in files_sphere_raw:
        try:
            h=fits.getheader(file)
            if 'ALC' in h['HIERARCH ESO INS COMB ICOR'] and 'IRDIS' in h['HIERARCH ESO SEQ ARM']:                
                raw_files.append(files_sphere_raw)
                raw_archive_names.append(h['ARCFILE'])
                raw_dpr_types.append(h['HIERACH ESO DPR TYPE'])
        except:
            continue
    

    #%% Step 1 : 
    
    # You look for all reduced frames that correspond to 
    # coronagraphic data with PRO.CATG as 
    #     - IRD_SCIENCE_DBI_LEFT_CUBE 
    #     - IRD_SCIENCE_DBI_RIGHT_CUBE
    #     - IRD_STAR_CENTER
    
    files_sphere_reduced = sorted(path_reduced.glob('r.SPHER*.fits'))
    reduced_files_dbi_left  = [] 
    raw_files_dbi_left = [] # careful !! This is not the raw file name as saved in raw/date/
    
    reduced_files_dbi_right = []
    raw_files_dbi_right = []
    
    for file in files_sphere_reduced:
        try:
            h=fits.getheader(file)
            # check whether this is a coronagraphic frame
            if 'ALC' in h['HIERARCH ESO INS COMB ICOR']:
                if h['HIERARCH ESO PRO CATG'] == 'IRD_SCIENCE_DBI_LEFT_CUBE':
                    reduced_files_dbi_left.append(file)
                    raw_files_dbi_left.append(h['HIERARCH ESO PRO REC1 RAW1 NAME'])
                if h['HIERARCH ESO PRO CATG'] == 'IRD_SCIENCE_DBI_RIGHT_CUBE':
                    reduced_files_dbi_right.append(file)
                    raw_files_dbi_right.append(h['HIERARCH ESO PRO REC1 RAW1 NAME'])
        except:
            continue

                
    
    #%% Step 2: 
    # You isolate for each raw frames one single reduced left and right cube 
    # (currently the pipeline does multiple reductions of the same raw cube for 
    # an unknown reason, we need to get rid of the duplicates here).
    
    unique_reduced_files_dbi_left,indices = np.unique(reduced_files_dbi_left,return_index=True)
    unique_raw_files_dbi_left = [raw_files_dbi_left[i] for i in indices]
    
    unique_reduced_files_dbi_right,indices = np.unique(reduced_files_dbi_right,return_index=True)
    unique_raw_files_dbi_right = [raw_files_dbi_right[i] for i in indices]
    
    
    #%% Step 3
    # You extract DIT, NDIT, NAXIS3, coronagraph name:
    #      Lyot stop: HIERARCH ESO INS1 OPTI1 NAME
    #      Lyot mask: HIERARCH ESO INS4 OPTI11 NAME
    #      coro combination name: HIERARCH ESO INS COMB ICOR
    # along with the ND filter and the IRDIS filter from each RAW frame.
    # Also extract all info that will be useful later for the analysis:
    # star name and magnitude, airmass, seeing, coherence time, telescope Seeing...)
    # This is important to do that on each raw frame on not on each reduced frame
    # as the keywords are changed by the pipeline and unreliable.
    
    
    
    #%% Step 4
    # Assoiate each FLUX with a CORONAGRAPHIC image
    
    
    
    #%% Step 5
    # Work on the Flux frame first:
    #     - Detect the central star
    #     - measure the FWHM, validate that this is the star by checking the FWHM
    #        (should be the diffraction limit)
    #     - aperture photometry on the star using a diameter of 1 lambda/D 
    #     - divide the flux by the DIT and correct by the ND transmission: this
    #       gives you the star reference flux
    
    
    #%% Step 6
    # Work on the coronographic frame now:
    #    - Detect the coronagraphic center.
    #    - compute the contrast (using the standard deviation of many apertures 
    #      place at a given separation, using for instance the python module VIP)


    #%% Step 7:
    #   Divide the azinuthal standard deviations by the star reference flux to obtain
    #   the contrast as a  function of radius. 
    #   Plot the result and save in a csv file along with all relevant parameters 
    #  for the analysis (coronagraph name, star magnitude, airmass, seeing, coherence time)

if __name__ == "__main__":

    # we define the locations of the raw and reduced data in the offline    
    path_root_data = Path('/data-ut3/')
    path_root_reduced = path_root_data.joinpath('reduced')
    path_root_raw = path_root_data.joinpath('raw')

    # by default, we want to use the current night, once the night has started the UT
    # time is always already the next day, so the current night is the day before
    # in UT time.
    yesterday = dtdate.today()-timedelta(days=1)
    date = yesterday.isoformat()

    path_reduced_default = path_root_reduced.joinpath(date)
    path_raw_default = path_root_raw.joinpath(date)

    # we also need to specify where the output data will be saved by default.
    # Here we specify my home folder
    path_root_output = Path('/diska/home/astro4/PrivateData/jmilli/sparta')
    path_output_default = path_root_output.joinpath(date)
    
    ###create the parser
    parser = argparse.ArgumentParser(description='Command line interface of the compute_contrast_sphere_offline module')
    ###add arguments        
    parser.add_argument('-reduced', help='Path of the reduced files', \
                        default=path_reduced_default,type=PosixPath) 
    parser.add_argument('-raw', help='Path of the raw files', \
                        default=path_raw_default,type=PosixPath)
    parser.add_argument('-output', help='Path to save the output files', \
                        default=path_output_default,type=PosixPath)
    parser.add_argument('-debug', help='debug mode', action='store_true')

    arguments = parser.parse_args()
    
    ####assign argument to parameters
    path_reduced = arguments.reduced
    path_raw = arguments.raw
    path_output = arguments.output
    debug = arguments.debug

    compute_contrast_sphere_offline(path_raw,path_reduced,path_output,debug=debug)



