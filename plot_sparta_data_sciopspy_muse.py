#!/diskc/PrivateData/SciOpsPy/envs/SciOpsPy/bin/./python
# -*- coding: utf-8 -*-

"""

---plot_sparta_data---

This script creates a summary of the atmospheric conditions and AO performances 
of MUSE WFM or NFM for a given night by analyzing the SPARTA data and 
downloading the corresponding ASM data and telescope data. 

@place: ESO - La Silla - Paranal Observatory
@author(s): Julien Milli
@year(s):  ex 2018
@Telescope(s): UT4
@Instrument(s): MUSE
@Valid for SciOpsPy: v0.1
@Documentation url: http://sciops.pl.eso.org/wiki/index.php/SciOps_scripts
@Last SciOps review [date + name]: 
@Usage: plot_sparta_data 2018-06-12
@Licence: ESO-license or GPLv3 (a copy is given in the root directory of this program)
@Testable: Yes
@Test data place (if any required): nothing
"""

import os
#import pdb
import sys
import glob
import subprocess
import warnings
import getopt
import operator
import functools
from astropy.io import fits, ascii
import numpy as np
import paramiko
import matplotlib.pyplot as plt
from astropy.time import Time
from datetime import timedelta 
from datetime import date as dtdate
import matplotlib.gridspec as gridspec 
import pandas as pd
import matplotlib as mpl
#from scipy.interpolate import interp1d
from astropy.utils.exceptions import AstropyWarning
from astropy import units as u
from astropy import coordinates
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=RuntimeWarning)
warnings.simplefilter('ignore',category=AstropyWarning)
from astropy.coordinates import SkyCoord
#from astropy.coordinates import ICRS, Galactic, FK4, FK5

# Definition of the UT4 coordinates
latitude =  -24.6270*u.degree
longitude = -70.4040*u.degree  
altitude = 2648.0*u.meter 
location = coordinates.EarthLocation(lon=longitude,lat=latitude,height=altitude)

def plot_sparta_data_muse(path_raw='.',path_output=None,plot=True,debug=True):
    """
    Function that reads the Sparta files and extract atmospheric and AO data 
    Input:
    - path_raw: the path where the SPARTA files are stored
    - path_output: the path were the plots, fits and .csv output files will
        be stored. By default it is the same as path_raw
    - plot: True to save a pdf plot
    - debug: True to print additional information
    Output:
    None
    """
    print('\n')
    if path_output == None:
        path_output = path_raw
    #test whether a path for results exists, and if not, create it
    if os.path.exists(path_output) == False:
        os.mkdir(path_output)
        print("The directory {0:s} did ".format(path_output),\
              "not exist and was created. Output data will be saved here.")
    else:
        print("Output data will be saved in {0:s}".format(path_output))

    # find the science files in the raw directory
#    files=sorted(glob.glob(os.path.join(path_raw,'MUSE*.fits')))
#    files=sorted(glob.glob(os.path.join(path_raw,'MUSE_NFM-AO_OBS*.fits')))
    files=sorted(glob.glob(os.path.join(path_raw,'MUSE_NFM-AO_OBS*.fits'))+\
                 glob.glob(os.path.join(path_raw,'MUSE_WFM-AO_OBS*.fits')))
    num_files = len(files)
    if num_files == 0:
        print('No SPARTA files found in {0:s}'.format(path_raw))
        return

    lam = 500.e-9 # wavelength at which r0 is given

    # List of parameters contained in the header of each file 
    name=[] # list of OB names (length=the number of sparta files)
    date_start_str=[] # list of the starting times of each OB
    date_end_str =[] # list of the ending times of each OB
    coords = [] # list of astropy.coordinates  (length=the number of sparta files)

    # Lists of atmospheric parameters contained in the binary table SPARTA_ATM_DATA
    sec_atmos=[] # list of strings in unix format for time of the atmospheric 
                 # parameters (seeing,r0,strehl,...)
    r0_atmos=[]# list of r0
    L0_atmos=[]# list of L0
    strehl=[] # list of strehl
    glf=[] # ground layer fraction
    altitude_atmos = [] # telescope altitude
    airmass_atmos = [] # telescope airmass
    mode_atmos = [] # MUSE mode (e.g. WFM-AO-N)
    ob_id_atmos = []
    ob_name_atmos = []
    target_name_atmos = []

    # List of Cn2 data from the profiler
    sec_cn2=[] # list of strings in unix format for time of the Cn2  
                 # parameters (seeing,r0,...)
    r0_cn2=[]# list of r0
    L0_cn2=[]# list of L0
    altitude_cn2 = [] # telescope altitude
    airmass_cn2 = [] # telescope airmass    

    # Reading the SPARTA files
    for i in range(num_files):
        hdu_list=fits.open(files[i])
        header = hdu_list[0].header
#        try:
#            # here you can add a test to filter bad files,
#            # based on some keywords. An example is given here.
#            if header['HIERARCH ESO DPR TYPE'] != 'OBJECT,AO' or \
#               header['HIERARCH ESO OBS PROG ID'] == 'Maintenance':
#                continue 
#        except: #if these keywords are not in the header we skip this file
#            continue
        wfs_mode = header['HIERARCH ESO INS MODE']
        ob_id = header['HIERARCH ESO OBS ID']
        ob_name = header['HIERARCH ESO OBS NAME']
        target_name = header['HIERARCH ESO OBS TARG NAME']
        name = np.append(name,target_name) # OB name
        date_start_str_tmp = header['DATE-OBS'] # starting date (temporary 
                           # because earlier times might be contained in the binary ext.)
        date_end_str_tmp = header['DATE'] # ending date (temporary 
                           # because later times might be contained in the binary ext.)
        date_start_tmp = Time(date_start_str_tmp) # converted to a Time object
        date_end_tmp = Time(date_end_str_tmp) # converted to a Time object
        ra = header['RA']*u.degree
        dec = header['DEC']*u.degree
        coords_J2000 = SkyCoord(ra,dec)
        coords = np.append(coords,coords_J2000)
        if debug:
            print('Reading {0:1} ({1:s} data)'.format(files[i][files[i].index('MUSE'):],wfs_mode))

        # We read the atmospheric data
        AtmPerfParams = hdu_list['SPARTA_ATM_DATA'].data 
        if len(AtmPerfParams["Sec"]) > 0:
            sec_atmos = np.append(sec_atmos, AtmPerfParams["Sec"]+\
                                 AtmPerfParams["USec"]/1.e6 )

            r0_atmos = np.append(r0_atmos,functools.reduce(operator.add,\
                    map(AtmPerfParams.__getattribute__, ['LGS1_R0','LGS2_R0',\
                                                         'LGS3_R0','LGS4_R0']))/4)
            L0_atmos = np.append(L0_atmos,functools.reduce(operator.add,\
                    map(AtmPerfParams.__getattribute__, ['LGS1_L0','LGS2_L0',\
                                                         'LGS3_L0','LGS4_L0']))/4)
            strehl = np.append(strehl,functools.reduce(operator.add,\
                    map(AtmPerfParams.__getattribute__, ['LGS1_STREHL','LGS2_STREHL',\
                                                         'LGS3_STREHL','LGS4_STREHL']))/4)
            glf = np.append(glf,functools.reduce(operator.add,\
                    map(AtmPerfParams.__getattribute__, ['LGS1_TUR_GND','LGS2_TUR_GND',\
                                                         'LGS3_TUR_GND','LGS4_TUR_GND']))/4)
            mode_atmos = np.append(mode_atmos,np.repeat([wfs_mode[0:6]],len(AtmPerfParams["Sec"])))
            ob_id_atmos = np.append(ob_id_atmos,np.repeat([ob_id],len(AtmPerfParams["Sec"])))
            ob_name_atmos = np.append(ob_name_atmos,np.repeat([ob_name],len(AtmPerfParams["Sec"])))
            target_name_atmos = np.append(target_name_atmos,np.repeat([target_name],len(AtmPerfParams["Sec"])))
    
            times_atmos = Time(AtmPerfParams["Sec"],format='unix')
            times_atmos.format='isot'
            for obstime in times_atmos:
                current_coords_altaz = coords_J2000.transform_to(\
                            coordinates.AltAz(obstime=obstime,location=location))
                altitude_atmos.append(current_coords_altaz.alt)
                z_atmos = current_coords_altaz.zen
                airmass_atmos = np.append(airmass_atmos,1/np.cos(z_atmos))
            if debug:
                print('   {0:3d} atmospheric parameters'.format(\
                                        len(AtmPerfParams["Sec"])))
            date_start_tmp = np.min(np.append(date_start_tmp,times_atmos))
            date_end_tmp = np.max(np.append(date_end_tmp,times_atmos))
        
        #  we read the turbulence profiling dat
        Cn2Data=hdu_list['SPARTA_CN2_DATA'].data # we read the turbulence profiling data
        if len(Cn2Data["Sec"]) > 0:
            sec_cn2 = np.append(sec_cn2, Cn2Data["Sec"]+\
                                 Cn2Data["USec"]/1.e6 )
            r0_cn2 = np.append(r0_cn2,Cn2Data['r0Tot'])
            L0_cn2 = np.append(L0_cn2,Cn2Data['L0Tot'])
            times_cn2 = Time(Cn2Data["Sec"]+Cn2Data["USec"]/1.e6,format='unix')
            times_cn2.format='isot'
            for obstime in times_cn2:
                current_coords_altaz = coords_J2000.transform_to(\
                            coordinates.AltAz(obstime=obstime,location=location))
                altitude_cn2.append(current_coords_altaz.alt)
                z_cn2 = current_coords_altaz.zen
                airmass_cn2 = np.append(airmass_cn2,1/np.cos(z_cn2))
            if debug:
                print('   {0:3d} Cn2 parameters'.format(len(Cn2Data["Sec"])))
            date_start_tmp = np.min(np.append(date_start_tmp,times_cn2))
            date_end_tmp = np.max(np.append(date_end_tmp,times_cn2))

        hdu_list.close()

        # now date_start_tmp is the true starting date so we can append 
        # it to the list
        date_start_str = np.append(date_start_str,date_start_tmp.iso)
        date_end_str = np.append(date_end_str,date_end_tmp.iso)

    # We create the Time objects after converting the list of strings 
    # using the unix format, check that the data is valid, and 
    # compute the seeing from r0
    try:
        time_atmos=Time(sec_atmos,format='unix')
        time_atmos.format='isot'
        strehl[np.logical_or(strehl>90.,strehl<0)]=np.nan
        seeing_atmos = np.rad2deg(lam/r0_atmos)*3600
        seeing_zenith_atmos = seeing_atmos*np.power(airmass_atmos,-3./5.)
        print('We read in total {0:d} atmospheric parameters among '\
          'which {1:d=.0f} are valid.'.format(len(time_atmos),\
                 np.sum(np.isfinite(strehl)*1.)))
        dico_atmos = {'strehl':strehl,'r0':r0_atmos,'date':time_atmos,\
                  'L0':L0_atmos,\
                  'glf':glf,'airmass':airmass_atmos,'seeing_los':seeing_atmos,\
                  'seeing_zenith':seeing_zenith_atmos,'mode':mode_atmos,\
                  '	OBS.ID':ob_id_atmos,'OBS.NAME':ob_name_atmos,\
                  'OBS.TARG.NAME':target_name_atmos}
        pd_atmos = pd.DataFrame(dico_atmos)
        pd_atmos.set_index('date', inplace=True)       
    except:
        if debug:
            print('Problem during the construction of'+\
                  ' the table of Sparta atmospheric data')
            print(sys.exc_info()[0])

    try:
        r0_atmos[np.logical_or(r0_atmos>0.9,r0_atmos<=0.)]=np.nan
        r0_cn2[np.logical_or(r0_cn2>0.9,r0_cn2<=0.)]=np.nan
        time_cn2=Time(sec_cn2,format='unix')
        time_cn2.format='isot'
        seeing_cn2 = np.rad2deg(lam/r0_cn2)*3600
        seeing_zenith_cn2 = seeing_cn2*np.power(airmass_cn2,-3./5.)
        print('We read in total {0:d} Cn2 parameters among '\
          'which {1:d=.0f} are valid.'.format(len(time_cn2),\
                                              np.sum(np.isfinite(r0_cn2)*1.)))
        dico_cn2 = {'date':time_cn2,'r0':r0_cn2,'seeing_los':seeing_cn2,\
                'seeing_zenith':seeing_zenith_cn2,'L0':L0_cn2}
        pd_cn2 = pd.DataFrame(dico_cn2)
        pd_cn2.set_index('date', inplace=True)
    except:
        if debug:
            print('Problem during the construction of'+\
                  ' the table of Sparta profiler data')
            print(sys.exc_info()[0])
        

    # we store in time_max the time of the last data
    try:
        time_max = np.max([np.max(time_atmos),np.max(time_cn2)])
    except:
        try:
            time_max = np.max([np.max(time_atmos)])
        except:
            print('There are no relevent data in the sparta files')
            return              

    # We create new arrays with unique OB names and dates
    # We create new arrays with unique OB names and dates
    new_date_start_str = []
    new_date_end_str = []
    new_name = []
    for i,namei in enumerate(name):
        if i==0: #the first OB
            new_name.append(namei)
            new_date_start_str.append(date_start_str[i])
        elif namei != name[i-1]: # in case this is a different OB
            new_name.append(namei)
            new_date_start_str.append(date_start_str[i])
            new_date_end_str.append(date_end_str[i-1])
#    new_date_end_str.append(time_max.iso)
    new_date_end_str.append(date_end_str[-1])

    time_file = Time(new_date_start_str) 
    nb_obs = len(new_name)

    # we store here the starting date 
    time_min = time_file[0]
    current_datetime = time_max.datetime 
    if current_datetime.hour > 12:
        current_night = current_datetime.date()
    else:
        current_night_datetime = current_datetime - timedelta(days=1)
        current_night = current_night_datetime.date()
    current_night_str = str(current_night)

    # We save the csv files
    try:
        pd_atmos.to_csv(os.path.join(path_output,\
                'SPARTA_ATM_DATA_{0:s}.csv'.format(current_night_str)))
    except:
        if debug:
            print('Problem while saving'+\
                  ' the table of Sparta atmospheric data')
            print(sys.exc_info()[0])

    try:
        pd_cn2.to_csv(os.path.join(path_output,\
                'SPARTA_CN2_DATA_{0:s}.csv'.format(current_night_str)))
    except:
        if debug:
            print('Problem while saving'+\
                  ' the table of Sparta profiler data')
            print(sys.exc_info()[0])

    #We query the TCS seeing 
    ######################################################
    #path of the log on wt4tcs
    tcs_log_path = '/vltuser/tcs/TIOs/Scripts/PlotLogSeeing2/data' 
    tcs_log_name = '{0:s}_seeinglog.log'.format(current_night_str)
    tcs_original_pathname =  os.path.join(tcs_log_path,tcs_log_name)
    tcs_final_pathname = os.path.join(path_output,tcs_log_name)

    # We make the request to the TCS here. This requires haveing exchanged 
    # public keys beforehand to avoid having to enter the tcs password. 
    # In case this does not work: log to astro3@wgsoff3 and type:
    # > chmod 0600 ~/.ssh/id_*.pub
    # > cat ~/.ssh/id_rsa.pub | ssh tcs@wt4tcs "mkdir -p ~/.ssh; 
    #                             cat >> ~/.ssh/authorized_keys"

    try:
        #connecting to server
        hostname='wt4tcs'
        usn='tcs'
        pw='Bnice2me'
        ssh=paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
        ssh.connect(hostname, username=usn, password=pw)
        ftp = ssh.open_sftp()

        # copying the shutter error hc plot
        if debug:
            info_str = 'Copying '+tcs_original_pathname+' --> '+tcs_final_pathname
            print(info_str)
        ftp.get(tcs_original_pathname,tcs_final_pathname)
        ftp.close()
    except:
        if debug:
            print('Error during the request of TCS log: ')
            print(sys.exc_info()[0])
        
#    request_tcs = ['scp','tcs@wt4tcs:'+tcs_original_pathname,path_output]
#    output,error = subprocess.Popen(request_tcs,stdout=subprocess.PIPE,\
#                                    stderr=subprocess.STDOUT).communicate()
#    if debug:
#        print(' '.join(request_tcs))
#        print(output)
#        if error != None:
#            print('Error during the request of TCS log: ')
#            print(error)
    try:
        tcs_log = ascii.read(tcs_final_pathname,fill_values=[('N/A',np.nan)])
        orig_columnNames = tcs_log.colnames
        columnNames = ['date','hms','dimm_seeing','FWHM_AG','FWHM_AG_zenith',\
                       'FWHM_IA_Normal','FWHM_IA_LinearObs',\
                       'FWHM_IA_LinearFit','airmass','junk','winddir',\
                       'windspeed_TopRing','windspeed_ASM']
        for i,origColumnName in enumerate(orig_columnNames):
            tcs_log.rename_column(origColumnName,columnNames[i])
        time_str_list_tcs_log = []
        for i,d in enumerate(tcs_log['date']):
            time_str_tcs_log = d+'T'+tcs_log['hms'][i]
            time_str_list_tcs_log.append(time_str_tcs_log)
        time_tcs_log = Time(time_str_list_tcs_log)
        if debug:
            print('The TCS log contains {0:d} values.'.format(len(tcs_log['date'])))
        ascii.write(tcs_log,tcs_final_pathname.replace('.log','.csv'),format='tab')
    except:
        if debug:
            print('Error while reading the TCS log {0:s}. No TCS values '.format(tcs_final_pathname)+\
                  'will be displayed.')

    #######################################################
    # We print the info for each target.
    # We first create arrays (same dimension as new name) to store 
    # the median values and later save it.
    start_ob_list = []
    end_ob_list = []

    median_r0_list=[]# list of r0
    median_L0_list=[]# list of L0
    median_strehl_list=[] # list of strehl
    median_glf_list=[] # ground layer fraction
    median_seeing_list = []
    median_seeing_zenith_list = []
    median_L0_list =[]    
    median_airmass_list = []
    for i,start_tmp_str in enumerate(new_date_start_str):
        end_tmp_str = new_date_end_str[i]
        start_tmp = Time(start_tmp_str,out_subfmt='date_hm')
        end_tmp = Time(end_tmp_str,out_subfmt='date_hm')
        try:
            id_atmos = np.logical_and(time_atmos>=start_tmp,time_atmos<=end_tmp)
            median_strehl = np.nanmedian(strehl[id_atmos])
            median_seeing = np.nanmedian(seeing_atmos[id_atmos])
            median_seeing_zenith = np.nanmedian(seeing_zenith_atmos[id_atmos])
            median_L0= np.nanmedian(L0_atmos[id_atmos])
            median_r0= np.nanmedian(r0_atmos[id_atmos])
            median_glf= np.nanmedian(glf[id_atmos])
            median_airmass = np.nanmedian(airmass_atmos[id_atmos])
        except:
            median_strehl = np.nan
            median_seeing = np.nan
            median_seeing_zenith = np.nan
            median_L0 = np.nan
            median_r0 = np.nan
            median_glf = np.nan
            median_airmass =  np.nan
            if debug:
                print('Problem during the construction of'+\
                  ' the summary table of Sparta atmospheric data')
                print(sys.exc_info()[0])         
        try:
            id_cn2 =  np.logical_and(time_cn2>=start_tmp,time_cn2<=end_tmp)
            median_seeing_cn2 = np.nanmedian(seeing_cn2[id_cn2])
            median_L0_cn2 = np.nanmedian(L0_cn2[id_cn2])
        except:
            median_seeing_cn2 = np.nan
            median_L0_cn2 = np.nan
            if debug:
                print('Problem during the construction of'+\
                  ' the summary table of Sparta profiler data')
                print(sys.exc_info()[0])         

        start_ob_list.append(start_tmp)
        end_ob_list.append(end_tmp)
        median_strehl_list.append(median_strehl)
        median_seeing_list.append(median_seeing)
        median_seeing_zenith_list.append(median_seeing_zenith)
        median_L0_list.append(median_L0)
        median_r0_list.append(median_r0)
        median_glf_list.append(median_glf)
        median_airmass_list.append(median_airmass)

    # Now we query the meteo tower to get the wind speed, direction and temperature
    start_date_asm_str = time_min.iso.replace(' ','T')
    end_date_asm_str = time_max.iso.replace(' ','T')
    if debug:
        print('Querying ASM data')
    request_asm = ['wget','-O',os.path.join(path_output,'asm_{0:s}.csv'.format(str(current_night))),\
            'http://archive.eso.org/wdb/wdb/asm/meteo_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&tab_press=0&tab_presqnh=0&tab_temp1=1&tab_temp2=0&tab_temp3=0&tab_temp4=0&tab_tempdew1=0&tab_tempdew2=0&tab_tempdew4=0&tab_dustl1=0&tab_dustl2=0&tab_dusts1=0&tab_dusts2=0&tab_rain=0&tab_rhum1=0&tab_rhum2=0&tab_rhum4=0&tab_wind_dir1=1&tab_wind_dir1_180=0&tab_wind_dir2=0&tab_wind_dir2_180=0&tab_wind_speed1=1&tab_wind_speed2=0&tab_wind_speedu=0&tab_wind_speedv=0&tab_wind_speedw=0'.format(start_date_asm_str,end_date_asm_str)]
    output,error = subprocess.Popen(request_asm,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
    if debug:
        print(' '.join(request_asm))
        print(output.decode('UTF8'))
    if error != None:
        print('Error during the request of the ASM database:')
        print(error)
    try:
        asm_df = pd.read_csv(os.path.join(path_output,'asm_{0:s}.csv'.format(str(current_night))),skiprows=1,skipfooter=5,engine='python') # 1st line is bank
        if debug:
            if len(asm_df.keys())<2:
                print('No data to be read in the ASM file.')
                raise IOError('Empty data in {0:s}'.format(os.path.join(path_output,'asm_{0:s}.csv'.format(str(current_night)))))
            else:
                print('The ASM file contains {0:d} values.'.format(len(asm_df)))
        asm_df.rename(columns={'Date time': 'date','Air Temperature at 30m [C]':'air_temperature_30m[deg]',\
                          'Wind Direction at 30m (0/360) [deg]':'winddir_30m',\
                          'Wind Speed at 30m [m/s]':'windspeed_30m'}, inplace=True)
        time_asm = Time(list(asm_df['date']),format='isot',scale='utc')
        asm_df.to_csv(os.path.join(path_output,'asm_{0:s}.csv'.format(str(current_night))),index=False)
    except Exception as e:
        if debug:
            print(e)        
            print("The plot won't contain any data from the ASM")


    # Now we query the MASS database to get the seeing, tau0 and GLF from the MASS-DIMM
    if debug:
        print('Querying mass-dimm data')
    request_asm_str = ['wget','-O',os.path.join(path_output,\
                    'mass_dimm_{0:s}.csv'.format(current_night_str)),\
                    'http://archive.eso.org/wdb/wdb/asm/mass_paranal/query?'\
                    'wdbo=csv&start_date={0:s}..{1:s}&tab_fwhm=1&tab_fwhmerr=0'\
                    '&tab_tau=1&tab_tauerr=0&tab_tet=0&tab_teterr=0&tab_alt=0'\
                    '&tab_alterr=0&tab_fracgl=1&tab_turbfwhm=1&tab_tau0=1&tab_'\
                    'tet0=0&tab_turb_alt=0&tab_turb_speed=1'.format(\
                    start_date_asm_str,end_date_asm_str)]
    if debug:
        print(' '.join(request_asm_str))
    output,error = subprocess.Popen(request_asm_str,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
    if debug:
        print(output.decode('UTF8'))
    if error != None:
        print('Error during the request of the ASM MASS-DIMM database:')
        print(error)

    # Now we read the MASS-DIMM file
    try:
        mass_df = pd.read_csv(os.path.join(path_output,\
                'mass_dimm_{0:s}.csv'.format(str(current_night))),\
                skiprows=1,skipfooter=5,engine='python')
        if debug:
            if len(mass_df.keys())<2:
                print('No data to be read in the mass-dimm file.')
                raise IOError('Empty data in {0:s}'.format(\
                    os.path.join(path_output,'mass_dimm_{0:s}.csv'.format(\
                    str(current_night)))))
            else:
                print('The MASS-DIMM file contains {0:d} values.'.format(\
                      len(mass_df)))
        mass_df.rename(columns={'Date time': 'date',\
                        'MASS Tau0 [s]':'MASS_tau0',\
                        'MASS-DIMM Cn2 fraction at ground':'MASS-DIMM_fracgl',\
                        'MASS-DIMM Tau0 [s]':'MASS-DIMM_tau0',\
                        'MASS-DIMM Turb Velocity [m/s]':'MASS-DIMM_turb_speed',\
                        'MASS-DIMM Seeing ["]':'MASS-DIMM_seeing',\
                        'Free Atmosphere Seeing ["]':'MASS_freeatmos_seeing'},\
                        inplace=True)        
        time_mass_dimm_asm = Time(list(mass_df['date']),format='isot',\
                                  scale='utc')
        mass_df.to_csv(os.path.join(path_output,'mass_dimm_{0:s}.csv'.format(\
                                    str(current_night))),index=False)
    except Exception as e:
        time_mass_dimm_asm=None
        if debug:
            print(e)        
            print("The plot won't contain any MASS-DIMM data.")

    # Now we read the DIMM file
    if debug:
        print('Querying dimm data')        
    # Now we query the ASM database to get the seeing from the DIMM
    request_asm_str = ['wget','-O',os.path.join(path_output,'dimm_{0:s}.csv'.format(str(current_night))),\
                       'http://archive.eso.org/wdb/wdb/asm/dimm_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&tab_fwhm=1&tab_rfl=0&tab_rfl_time=0'.format(\
                       start_date_asm_str,end_date_asm_str)]
    output,error = subprocess.Popen(request_asm_str,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
    if debug:
        print(' '.join(request_asm_str))
        print(output.decode('UTF8'))
    if error != None:
        print('Error during the request of the ASM DIMM database:')
        print(error)
    
    try:
        dimm_df = pd.read_csv(os.path.join(path_output,\
                    'dimm_{0:s}.csv'.format(str(current_night))),\
                    skiprows=1,skipfooter=5,engine='python')
        if debug:
            if len(dimm_df.keys())<2:
                print('No data to be read in the dimm file.')
                raise IOError('Empty data in {0:s}'.format(os.path.join(\
                    path_output,'dimm_{0:s}.csv'.format(str(current_night)))))
            else:
                print('The DIMM file contains {0:d} values.'.format(\
                      len(dimm_df)))
        dimm_df.rename(columns={'Date time': 'date',\
                        'DIMM Seeing ["]':'dimm_seeing'}, inplace=True)
        time_dimm_asm = Time(list(dimm_df['date']),format='isot',scale='utc')
        dimm_df.to_csv(os.path.join(path_output,'dimm_{0:s}.csv'.format(\
                                    str(current_night))),index=False)
    except Exception as e:
        time_dimm_asm=None
        if debug:
            print(e)
            print("The plot won't contain any DIMM data.")

    # Now we query the SLODAR database to get the seeing from the SLODAR
    if debug:
        print('Querying slodar data')
    request_asm_str = ['wget','-O',os.path.join(path_output,\
            'slodar_{0:s}.csv'.format(str(current_night))),\
            'http://archive.eso.org/wdb/wdb/asm/slodar_paranal/query?wdbo=csv'\
            '&start_date={0:s}..{1:s}&tab_cnsqs_uts=1&tab_fracgl300=1&'\
            'tab_fracgl500=1&tab_hrsfit=1&tab_fwhm=1'.format(\
            start_date_asm_str,end_date_asm_str)]
    output,error = subprocess.Popen(request_asm_str,stdout=subprocess.PIPE,\
                                    stderr=subprocess.STDOUT).communicate()
    if debug:
        print(' '.join(request_asm_str))
        print(output.decode('UTF8'))
    if error != None:
        print('Error during the request of the ASM SLODAR database:')
        print(error)
    
    # Now we read the SLODAR file
    try:
        slodar_df = pd.read_csv(os.path.join(path_output,\
                    'slodar_{0:s}.csv'.format(str(current_night))),skiprows=1,\
                    skipfooter=5,engine='python')
        if debug:
            if len(slodar_df.keys())<2:
                print('No data to be read in the slodar file.')
                raise IOError('Empty data in {0:s}'.format(os.path.join(\
                    path_output,'slodar_{0:s}.csv'.format(str(current_night)))))
            else:
                print('The slodar file contains {0:d} values.'.format(\
                      len(slodar_df)))
        slodar_df.rename(columns={'Date time': 'date',\
                        'Cn2 above UTs [10**(-15)m**(1/3)]':'Cn2_above_UT',\
                        'Cn2 fraction below 300m':'slodar_GLfrac_300m',\
                        'Cn2 fraction below 500m':'slodar_GLfrac_500m',\
                        'Surface layer profile [10**(-15)m**(1/3)]':\
                        'slodar_surface_layer','Seeing ["]':'slodar_seeing'},\
                        inplace=True)
        wave_nb=2*np.pi/lam
        time_slodar_asm = Time(list(slodar_df['date']),format='isot',scale='utc')   
        slodar_df['slodar_r0_above_UT'] = np.power(0.423*(wave_nb**2)*\
                                     slodar_df['Cn2_above_UT']*1.e-15,-3./5.)
        slodar_df['slodar_seeing_above_UT']= np.rad2deg(\
                 lam/slodar_df['slodar_r0_above_UT'])*3600.
        slodar_df['slodar_Cn2_total'] = np.power(\
                 slodar_df['slodar_seeing']/2.e7,1./0.6) # in m^1/3
        slodar_df['slodar_surface_layer_fraction'] = \
            slodar_df['slodar_surface_layer']*1.e-15 / slodar_df['slodar_Cn2_total']
        slodar_df.to_csv(os.path.join(path_output,\
                    'slodar_{0:s}.csv'.format(str(current_night))),index=False)
    except KeyError as e:
        if debug:
            print(e)
            print("The plot won't contain any SLODAR data.")
    except ascii.core.InconsistentTableError as e:
         if debug:
            print(e)
            print('There was probably only one SLODAR data point.')
            print("The plot won't contain any SLODAR data.")       
    except Exception as e:
        if debug:
            print(e)        
            print("The plot won't contain any SLODAR data.")



    #######################################################
    median_seeing_MASSDIMM_list = []
    median_tau0_MASSDIMM_list = []
    median_seeing_DIMM_list = []
    median_GLF_MASSDIMM_list =[]
    median_turbspeed_MASSDIMM_list = []
    median_topring_windspeed_list = []
    median_windspeed_ASM_list = []
    median_FWHM_IALINFIT_list = []
    median_FWHM_AG_zenith_list = []
#    median_slodar_GLfrac_300m_list = []
#    median_slodar_GLfrac_500m_list = []
#    median_slodar_seeing_above_UT_list = []
    for i,start_ob in enumerate(start_ob_list):
        end_ob = end_ob_list[i]
        try:
            id_MASSDIMM = np.logical_and(time_mass_dimm_asm>=start_ob,\
                                     time_mass_dimm_asm<=end_ob)
            median_seeing_MASSDIMM = np.nanmedian(\
                                    mass_df['MASS-DIMM_seeing'][id_MASSDIMM])
            median_tau0_MASSDIMM = np.nanmedian(\
                                    mass_df['MASS-DIMM_tau0'][id_MASSDIMM])
            median_GLF_MASSDIMM = np.nanmedian(\
                                    mass_df['MASS-DIMM_fracgl'][id_MASSDIMM])
            median_turbspeed_MASSDIMM = np.nanmedian(\
                                    mass_df['MASS-DIMM_turb_speed'][id_MASSDIMM])
        except:
            median_seeing_MASSDIMM = np.nan
            median_tau0_MASSDIMM = np.nan
            median_GLF_MASSDIMM = np.nan
            median_turbspeed_MASSDIMM = np.nan
        try:
            id_ASM = np.logical_and(time_asm>=start_ob,\
                                     time_asm<=end_ob)
            median_windspeed_ASM = np.nanmedian(asm_df['windspeed_30m'][id_ASM])
        except:
            median_windspeed_ASM = np.nan 
            print(sys.exc_info()[0])
        try:
            id_TCS = np.logical_and(time_tcs_log>=start_ob,\
                                     time_tcs_log<=end_ob)
            median_topring_windspeed = np.nanmedian(tcs_log['windspeed_TopRing'][id_TCS])
            median_FWHM_IALINFIT =  np.nanmedian(tcs_log['FWHM_IA_LinearFit'][id_TCS])
            median_FWHM_AG_zenith =  np.nanmedian(tcs_log['FWHM_AG_zenith'][id_TCS])
        except:
            median_topring_windspeed = np.nan
            median_FWHM_IALINFIT =  np.nan
            median_FWHM_AG_zenith =  np.nan

#            print(sys.exc_info()[0])            
        median_seeing_MASSDIMM_list.append(median_seeing_MASSDIMM)
        median_tau0_MASSDIMM_list.append(median_tau0_MASSDIMM)
        median_GLF_MASSDIMM_list.append(median_GLF_MASSDIMM)
        median_turbspeed_MASSDIMM_list.append(median_turbspeed_MASSDIMM)
        median_windspeed_ASM_list.append(median_windspeed_ASM)
        median_topring_windspeed_list.append(median_topring_windspeed)
        median_FWHM_IALINFIT_list.append(median_FWHM_IALINFIT)
        median_FWHM_AG_zenith_list.append(median_FWHM_AG_zenith)

    print('################################################################'\
          '###############################')
    print('                                                           '\
          'Strehl Seeing Seeing Seeing GLF  GLF  tau0   L0  airmass wind     Topring')
    print('                                                           '\
          '       LoS    Zenith MDIMM  LoS MDIMM MDIMM  LoS  LoS    ASM      TCS')
    for i,start_tmp_str in enumerate(new_date_start_str):
        try:
            print('{0:s} '.format(new_name[i].ljust(35)[0:35]),\
                  'from {0:s} to '.format(start_ob_list[i].value[11:]),\
                  '{0:s} '.format(end_ob_list[i].value[11:]),\
                  '{0:3.1f}%   {1:2.1f}"  '.format(\
                  median_strehl_list[i],median_seeing_list[i]),\
                  '{0:2.1f}"  '.format(median_seeing_zenith_list[i]),\
                  '{0:3.1f}"  '.format(median_seeing_MASSDIMM_list[i]),\
#                  '{0:3.1f}"  '.format(median_seeing_FWHM_IALINFIT_list[i]),\
#                  '{0:3.1f}"  '.format(median_seeing_AG_zenith_list[i]),\
                  '{0:2.0f}% '.format(median_glf_list[i]*100),\
                  '{0:2.0f}% '.format(median_GLF_MASSDIMM_list[i]*100),\
                  '{0:3.1f}ms '.format(median_tau0_MASSDIMM_list[i]*1000),\
                  '{0:2.0f}m '.format(median_L0_list[i]),\
                  '{0:4.2f}   '.format(median_airmass_list[i])+\
                  '{0:4.1f}m/s'.format(median_windspeed_ASM_list[i]),
                  '{0:4.2f}m/s'.format(median_topring_windspeed_list[i]))
        except:
            print('{0:s} from {1:s} to {2:s} '.format(new_name[i].ljust(35),\
                   start_tmp.value[11:],end_tmp.value[11:]))

    ascii.write([new_name,[str(st) for st in start_ob_list],\
                 [str(st) for st in end_ob_list],median_strehl_list,\
                 median_seeing_list,median_seeing_zenith_list,\
                 median_seeing_MASSDIMM_list,median_glf_list,\
                 median_GLF_MASSDIMM_list,median_tau0_MASSDIMM_list,\
                 median_L0_list,median_airmass_list,median_windspeed_ASM_list,\
                 median_topring_windspeed_list],\
                os.path.join(path_output,'summary_{0:s}.csv'.format(\
                current_night_str)),names=['OB_name','start_UT','end_UT',\
                'strehl_sparta','seeing_los_sparta','seeing_zenith_sparta',\
                'seeing_MASSDIMM','GLF_sparta','GLF_MASSDIMM',\
                'tau0_MASSDIMM','L0_sparta','airmass','windspeed_ASM',\
                'topring_windspeed'],format='csv',overwrite=True)




    if plot:
        majorFormatter = mpl.dates.DateFormatter('%H:%M')
        plt.close(1)
        fig = plt.figure(1, figsize=(12,15))
        plt.rcParams.update({'font.size':14})
        
        gs = gridspec.GridSpec(4,2, height_ratios=[1,1,1,1],)
        gs.update(left=0.1, right=0.95, bottom=0.1, top=0.98, wspace=0.2, hspace=0.3)
        
        ax1 = plt.subplot(gs[0,0]) # Area for the seeing
        ax2 = plt.subplot(gs[0,1]) # Area for the tau0
        ax3 = plt.subplot(gs[1,0]) # Area for 
        ax4 = plt.subplot(gs[1,1]) # Area for the Strehl
        ax5 = plt.subplot(gs[2:4,0]) # Area for the combined plot
        ax6 = plt.subplot(gs[2,1]) # Area for GLF
        ax7 = plt.subplot(gs[3,1]) # Area for the wind
        
        # Plot the seeing
        try:
            ax1.plot_date(time_atmos.plot_date,seeing_zenith_atmos,'.', color='darkorange',\
                      markeredgecolor='none',label='sparta')    
        except:
            if debug:
                print('No Sparta data available for plot 1',sys.exc_info()[0])            
        try:
            ax1.plot_date(time_mass_dimm_asm.plot_date,\
                          mass_df['MASS-DIMM_seeing'],'.',\
                          color='navy',markeredgecolor='none',\
                          label='MASS-DIMM')
        except:
            if debug:
                print('No MASS-DIMM data available for the plot ',sys.exc_info()[0])
        try:
            ax1.plot_date(time_cn2.plot_date,seeing_zenith_cn2,'.', color='forestgreen',\
                      markeredgecolor='none',label='sparta profiler')    
        except:
            if debug:
                print('No Sparta profiler data available for the plot ',sys.exc_info()[0])
        try:
            ax1.plot_date(time_tcs_log.plot_date,tcs_log['FWHM_AG'],'.',\
                          color='blue',markeredgecolor='none',label='AG')   
        except:
            if debug:
                print('No TCS AG data available for the plot 1: ',sys.exc_info()[0])
        try:
            ax1.plot_date(time_dimm_asm.plot_date,dimm_df['dimm_seeing'],\
                    '.', color='dimgrey',markeredgecolor='none',label='DIMM')   
        except:
            if debug:
                print('No DIMM data available for the plot 1:',sys.exc_info()[0])
        try:
            ax1.plot_date(time_slodar_asm.plot_date,\
                slodar_df['slodar_seeing_above_UT'],'.',\
                color='magenta',markeredgecolor='none',label='SLODAR above UT')   
        except:
            if debug:
                print('No SLODAR data available for the plot 1: ',sys.exc_info()[0])
        try:
            ax1.plot_date(time_tcs_log.plot_date,tcs_log['FWHM_IA_LinearFit'],'.', color='rosybrown',\
                          markeredgecolor='none',label='TEL.IA.FWHMLIN')
        except:
            if debug:
                print('No TCS data available for the plot 1: ',sys.exc_info()[0])
        ax1.set_ylabel('Seeing (arcsec)')
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
        # add filling for even OBs 
        min_y,max_y = ax1.get_ybound()
        for i in range(0,nb_obs-2,2):
            ax1.fill_between(time_file[i:i+2].plot_date,0,max_y, \
                             facecolor='blue', alpha=0.1)
        if np.mod(nb_obs,2)==0: #then you add the last one which is defined
            ax1.fill_between(time_file[nb_obs-2:nb_obs].plot_date,0,\
                             max_y, facecolor='blue', alpha=0.1)
        else:
            ax1.fill_between([time_file[-1].plot_date,time_atmos[-1].plot_date],\
                             0,100, facecolor='blue', alpha=0.2)            
        ax1.set_ylim(min_y,max_y)
        ax1.set_xlim(time_min.plot_date,time_max.plot_date)
        ax1.grid()
        ax1.legend(frameon=False,loc='best',fontsize=10)
        ax1.xaxis.set_major_formatter(majorFormatter)

        
        # Plot the tau0
        try:
            ax2.plot_date(time_mass_dimm_asm.plot_date,\
                          np.asarray(mass_df['MASS-DIMM_tau0'])*1000.,'.',\
                          color='navy',label='MASS-DIMM',\
                          markeredgecolor='none')
            ax2.plot_date(time_mass_dimm_asm.plot_date,\
                          np.asarray(mass_df['MASS_tau0'])*1000.,'.',\
                          color='dimgrey',label='MASS',markeredgecolor='none')
        except:
            if debug:
                print('No MASS-DIMM data available for the plot 2: ',sys.exc_info()[0])
        ax2.set_ylabel('$\\tau_0$ (ms)')
        for tick in ax2.get_xticklabels():
            tick.set_rotation(45)
        # add filling for even OBs 
        min_y,max_y = ax2.get_ybound()
        for i in range(0,nb_obs-2,2):
            ax2.fill_between(time_file[i:i+2].plot_date,min_y,max_y,\
                             facecolor='blue', alpha=0.1)
        if np.mod(nb_obs,2)==0: #then you add the last one which is defined
            ax2.fill_between(time_file[nb_obs-2:nb_obs].plot_date,min_y,max_y,\
                             facecolor='blue', alpha=0.1)
        else:
            ax2.fill_between([time_file[-1].plot_date,time_atmos[-1].plot_date],\
                             min_y,max_y, facecolor='blue', alpha=0.1)            
        ax2.set_xlim(time_min.plot_date,time_max.plot_date)
        ax2.set_ylim(min_y,max_y)
        ax2.grid()
        ax2.legend(frameon=False,loc='best',fontsize=10)
        ax2.xaxis.set_major_formatter(majorFormatter)
    

        # plot the L0
        try:
            ax3.plot_date(time_atmos.plot_date,L0_atmos,\
                '.', color='red',markeredgecolor='none',label='Sparta')
        except:
            if debug:
                print('No L0 Sparta data available for the plot 3: ',sys.exc_info()[0])
        try:
            ax3.plot_date(time_cn2.plot_date,L0_cn2,'.', color='forestgreen',\
                      markeredgecolor='none',label='Sparta profiler')    
        except:
            if debug:
                print('No Sparta profiler data available for plot 3',sys.exc_info()[0])
        #ax3.set_ylim(ymin,ymax)
        ax3.set_yscale("log", nonposy='clip')
        ax3.set_ylabel('Sparta L0 in m')
        ax3.legend(frameon=False,loc='best')
        for tick in ax3.get_xticklabels():
            tick.set_rotation(45)
        # add filling for even OBs 
        ax3.grid()
        ax3.set_xlim(time_min.plot_date,time_max.plot_date)
        ax3.xaxis.set_major_formatter(majorFormatter)
        min_y,max_y = ax3.get_ybound()
        for i in range(0,nb_obs-2,2):
            ax3.fill_between(time_file[i:i+2].plot_date,0,ax3.get_ybound()[1],\
                             facecolor='blue', alpha=0.1)
        if np.mod(nb_obs,2)==0: #then you add the last one which is defined
            ax3.fill_between(time_file[nb_obs-2:nb_obs].plot_date,0,\
                             ax3.get_ybound()[1], facecolor='blue', alpha=0.1)
        else:
            ax3.fill_between([time_file[-1].plot_date,time_atmos[-1].plot_date],\
                             0,ax3.get_ybound()[1], facecolor='blue', alpha=0.1)            
        ax3.set_ylim(min_y,max_y)

             
        # plot the airmass
        try:
            ax4.plot_date(time_atmos.plot_date,airmass_atmos,'.', color='darkorchid',\
                      markeredgecolor='none')
        except:
            if debug:
                print('No airmass data available for the plot 4: ',sys.exc_info()[0])
        ax4.set_ylabel('Airmass')        
        min_y,max_y = ax4.get_ybound()
        for i in range(0,nb_obs-2,2):
            ax4.fill_between(time_file[i:i+2].plot_date,0,100, facecolor='blue',\
                             alpha=0.1)
        if np.mod(nb_obs,2)==0: #then you add the last one which is defined
            ax4.fill_between(time_file[nb_obs-2:nb_obs].plot_date,0,100,\
                             facecolor='blue', alpha=0.1)
        else:
            ax4.fill_between([time_file[-1].plot_date,time_atmos[-1].plot_date],\
                             0,100, facecolor='blue', alpha=0.1)            
        for tick in ax4.get_xticklabels():
            tick.set_rotation(45)
        ax4.grid()
        ax4.set_xlim(time_min.plot_date,time_max.plot_date)
        ax4.xaxis.set_major_formatter(majorFormatter)
        ax4.set_ylim(min_y,max_y)
        
        #Plot 5: large plot of Strehl with size prop to seeing
        try:
            med_seeing_zenith_atmos = np.median(seeing_atmos)
            size_strehl = seeing_zenith_atmos/med_seeing_zenith_atmos*20
            ax5.scatter(time_atmos.plot_date,strehl, color='darkorchid',\
                        s=size_strehl,marker='s',alpha=0.5)
        except:
            if debug:
                print('No Strehl data available for the plot 5: ',sys.exc_info()[0]) 
        ax5.set_ylim(0,40)
        ax5.axhline(5,color='red')
        min_y,max_y = ax5.get_ybound()
        for tick in ax5.get_xticklabels():
            tick.set_rotation(45)
        ax5.set_xlabel('Night of {0:s}'.format(current_night_str))
        ax5.set_ylabel('Strehl in % (size prop. to Sparta seeing)')
        for i in range(0,nb_obs-2,2):
            ax5.fill_between(time_file[i:i+2].plot_date,min_y,max_y,\
                             facecolor='blue', alpha=0.1)
        if np.mod(nb_obs,2)==0: #then you add the last one which is defined
            ax5.fill_between(time_file[nb_obs-2:nb_obs].plot_date,min_y,\
                             max_y, facecolor='blue', alpha=0.1)
        else:
            ax5.fill_between([time_file[-1].plot_date,\
                              time_atmos[-1].plot_date],min_y,max_y,\
                facecolor='blue', alpha=0.1)            
        for i,time_filei in enumerate(time_file):
            ax5.text(time_filei.plot_date, max_y-(max_y-min_y)*0.02,\
                     new_name[i], fontsize=10,rotation=90)#min_y+(max_y-min_y)/5.
        for tick in ax5.get_xticklabels():
            tick.set_rotation(45)
        ax5.grid()
        ax5.set_ylim(min_y,np.max([2,max_y]))
        ax5.set_xlim(time_min.plot_date,time_max.plot_date)
        ax5.xaxis.set_major_formatter(majorFormatter)        

        # Plot the ground layer and surface layer fraction
        try:
            ax6.plot_date(time_atmos.plot_date,\
                          glf*100,'.',\
                          color='darkorange',label='SPARTA',\
                          markeredgecolor='none')
        except:
            if debug:
                print('No SPARTA data available for the plot 6:',sys.exc_info()[0])
        try:
            ax6.plot_date(time_mass_dimm_asm.plot_date,\
                          np.asarray(mass_df['MASS-DIMM_fracgl'])*100.,'.',\
                          color='navy',label='MASS-DIMM',\
                          markeredgecolor='none')
        except:
            if debug:
                print('No MASS-DIMM data available for the plot 6:',sys.exc_info()[0])
        try:
            ax6.plot_date(time_slodar_asm.plot_date,\
                          slodar_df['slodar_surface_layer_fraction']*100,'.',\
                          color='seagreen',markeredgecolor='none',\
                          label='SLODAR surface layer')   
            ax6.plot_date(time_slodar_asm.plot_date,\
                          slodar_df['slodar_GLfrac_500m']*100,'.',\
                          color='red',markeredgecolor='none',label='SLODAR 500m')   
            ax6.plot_date(time_slodar_asm.plot_date,\
                          slodar_df['slodar_GLfrac_300m']*100,'.',\
                          color='blue',markeredgecolor='none',label='SLODAR 300m')   
        except:
            if debug:
                print('No SLODAR data available for the plot 6:',sys.exc_info()[0])
        ax6.set_ylabel('Ground layer fraction (%)')
        for tick in ax6.get_xticklabels():
            tick.set_rotation(45)
        # add filling for even OBs 
        min_y,max_y = [0,100] #ax6.get_ybound()
        for i in range(0,nb_obs-2,2):
            ax6.fill_between(time_file[i:i+2].plot_date,min_y,max_y,\
                             facecolor='blue', alpha=0.1)
        if np.mod(nb_obs,2)==0: #then you add the last one which is defined
            ax6.fill_between(time_file[nb_obs-2:nb_obs].plot_date,min_y,\
                             max_y, facecolor='blue', alpha=0.1)
        else:
            ax6.fill_between([time_file[-1].plot_date,\
                              time_atmos[-1].plot_date],min_y,max_y,\
                facecolor='blue', alpha=0.1)            
        ax6.set_xlim(time_min.plot_date,time_max.plot_date)
        ax6.set_ylim(min_y,max_y)
        ax6.grid()
        ax6.legend(frameon=False,loc='best',fontsize=10)
        ax6.xaxis.set_major_formatter(majorFormatter)

        # Plot the wind speed
        try:
            ax7.plot_date(time_mass_dimm_asm.plot_date,\
                          np.asarray(mass_df['MASS-DIMM_turb_speed']),'.',\
                          color='navy',label='MASS-DIMM',\
                          markeredgecolor='none')
        except :
            if debug:
                print('No MASS-DIMM data available for the plot 7:',sys.exc_info()[0])
        try:
            ax7.plot_date(time_tcs_log.plot_date,tcs_log['windspeed_TopRing'],'.',\
                          color='rosybrown',markeredgecolor='none',\
                          label='Top ring')
        except:
            if debug:
                print('No TCS top ring data available for the plot 7:',sys.exc_info()[0])
        try:
            ax7.plot_date(time_asm.plot_date,asm_df['windspeed_30m'],'.',\
                          color='forestgreen',markeredgecolor='none',\
                          label='ASM 30m')
        except:
            print('No ASM wind speed data available for the plot 7:',sys.exc_info()[0])
        ax7.set_ylabel('Wind speed (m/s)')
        for tick in ax7.get_xticklabels():
            tick.set_rotation(45)
        # add filling for even OBs 
        min_y,max_y = ax7.get_ybound()
        for i in range(0,nb_obs-2,2):
            ax7.fill_between(time_file[i:i+2].plot_date,min_y,max_y,\
                             facecolor='blue', alpha=0.1)
        if np.mod(nb_obs,2)==0: #then you add the last one which is defined
            ax7.fill_between(time_file[nb_obs-2:nb_obs].plot_date,min_y,\
                             max_y, facecolor='blue', alpha=0.1)
        else:
            ax7.fill_between([time_file[-1].plot_date,\
                              time_atmos[-1].plot_date],min_y,max_y,\
                facecolor='blue', alpha=0.1)            
        ax7.set_xlim(time_min.plot_date,time_max.plot_date)
        ax7.set_ylim(min_y,max_y)
        ax7.grid()
        ax7.legend(frameon=False,loc='best',fontsize=10)
        ax7.xaxis.set_major_formatter(majorFormatter)
            
        fig.savefig(os.path.join(path_output,\
                            'sparta_plot_{0:s}.pdf'.format(current_night_str)))

    return 

if __name__ == "__main__":
    debug=False
    path_raw=None
    path_output=None
    try:
        opts, args = getopt.getopt(sys.argv[1:],"i:o:d:",\
                                   ["ifolder=","ofolder=","debug="])
    except:
        print(sys.exc_info()[0])
        print('Syntax error. Use one of those syntaxes:')
        print('plot_sparta_data_muse -i <inputfolder> -o '\
              '<outputfolder> -d=True/False')
        print('plot_sparta_data_muse 2018-09-10')
        print('plot_sparta_data_muse')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ["-d","--debug"]:
            debug=arg
        elif opt in ("-i", "--ifolder"):
            path_raw = arg
        elif opt in ("-o", "--ofolder"):
            path_output = arg
    if path_raw==None or path_output==None:
        # in this case we assume the 1st arg is the date and we save 
        # the result in my folder
        if len(sys.argv)==2:
            date = sys.argv[1]
        elif len(sys.argv)==1:
            yesterday = dtdate.today()-timedelta(days=1)
            date = yesterday.isoformat()
        else:
            print('Syntax error. Use one of those syntaxes:')
            print('plot_sparta_data_muse -i <inputfolder> -o '\
              '<outputfolder> -d=True/False')
            print('plot_sparta_data_muse 2018-09-10')
            print('plot_sparta_data_muse')
            sys.exit(2)
        path_raw = os.path.join('/data-ut4/raw',date)
        path_sparta = '/diska/home/astro4/PrivateData/jmilli/sparta'
        path_output = os.path.join(path_sparta,date)
    plot_sparta_data_muse(path_raw=path_raw,path_output=path_output,debug=debug)
