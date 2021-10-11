# -*- coding: utf-8 -*-

"""
This script creates a summary of the atmospheric conditions and AO performances
for a given night by analyzing the SPARTA data and downloading the corresponding
ASM data.
Author: J. Milli
Creation date: 2017-01-10
"""

from astropy.io import fits ,ascii
import os,sys
import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.time import Time
from datetime import timedelta #datetime
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from scipy.interpolate import interp1d
import subprocess
from astropy import units as u
from astropy import coordinates
import getopt
import pandas as pd
from astropy.utils.exceptions import AstropyWarning
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
warnings.simplefilter('ignore',category=AstropyWarning)
from astropy.coordinates import SkyCoord
#from astropy.coordinates import Galactic, FK5
# import pdb # for debugging purposes
from query_eso_archive import query_simbad
from ecmwf_utilities import request_ecmwf
#from astropy.utils.iers import conf
#conf.auto_max_age = None

# Definition of the UT3 coordinates
latitude =  -24.6268*u.degree
longitude = -70.4045*u.degree
altitude = 2648.0*u.meter
location = coordinates.EarthLocation(lon=longitude,lat=latitude,height=altitude)

def plot_sparta_data(path_raw='.',path_output=None,plot=True,debug=True):
    """
    Function that reads the Sparta files and extract atmospheric and AO data
    Input:
    - path_raw: the path where the SPARTA files are stored. They must have
        a DPR.TYPE=OBJECT,AO and be named SPHER*.fits
    - path_output: the path were the plots, fits and .txt output files will
        be stored. By default it is the same as path_raw
    - plot: True to save a pdf plot
    - debug: True to print additional information
    Output:
    - None
    """
    if path_output == None:
        path_output = path_raw
    #test whether a path for results exists, and if not, create it
    if os.path.exists(path_output) == False:
        os.mkdir(path_output)
        print("The directory {0:s} did not exist and was created".format(path_output))
    # find the sparta files in the raw directory
    files=sorted(glob.glob(os.path.join(path_raw,'SPHER*.fits')))
    num_files = len(files)
    if num_files == 0:
        print('No SPARTA files found in {0:s}'.format(path_raw))
        return
    lam = 500.e-9 # wavelength at which r0 is given
    sec_atmos=[] # list of strings in unix format for time of the atmospheric parameters (seeing,r0,strehl,tau0,windSpeed)
    r0=[]# list of r0
    windSpeed=[] # list of wind speed
    strehl=[] # list of strehl
    file_atmos = [] # list of fits files names
    sec_VisLoop=[] # list of strings in unix format for time of
    flux_VisLoop=[] # list of fluxes for the visible WFS
    sec_IRLoop=[] # list of strings in unix format for time of
    flux_IRLoop=[] # list of fluxes for the IR DTTS images
    sec_img_DTTS=[] # list of strings in unix format for time of
    cube_DTTS = [] # list of cubes of images of the DTTS
    name=[] # list of OB names (length=the number of sparta files)
    date_start_str=[] # list of the starting time of each OB
    coords = [] # list of astropy.coordinates  (length=the number of sparta files)
    altitude = [] # list of altitudes (length=the number atmosph timestamps)
    airmass = [] # list of airmasses (length=the number atmosph timestamps)
    # Reading the SPARTA files
    for i in range(num_files):
        hdu_list=fits.open(files[i])
        header = hdu_list[0].header
        try:
#            if header['HIERARCH ESO DPR TYPE'] != 'OBJECT,AO':
            if header['HIERARCH ESO DPR TYPE'] != 'OBJECT,AO' or \
                    header['HIERARCH ESO OBS PROG ID'] == 'Maintenance' or \
                    len(hdu_list)<2:
                if debug:
                    print('Bad file {0:1}'.format(files[i][files[i].index('SPHER'):]))
                continue
        except: #if these keywords are not in the header we skip this file
            continue
        name = np.append(name,header['HIERARCH ESO OBS NAME']) # OB name
        wfs_mode = header['HIERARCH ESO AOS VISWFS MODE']
        gain = int(wfs_mode[wfs_mode.index('GAIN_')+5:wfs_mode.index('_FREQ')])
        freq = int(wfs_mode[wfs_mode.index('FREQ_')+5:wfs_mode.index('Hz')])
        date_start_str_tmp = header['DATE'] # starting date (temporary because earlier times might be contained in the file)
        date_start_tmp = Time(date_start_str_tmp) # converted to a Time object
        ra = header['RA']*u.degree
        dec = header['DEC']*u.degree
        coords_J2000 = SkyCoord(ra,dec)
        coords = np.append(coords,coords_J2000)
        if debug:
            print('Reading {0:1}'.format(files[i][files[i].index('SPHER'):]))
            print('   AO mode: {0:s}'.format(wfs_mode))
        AtmPerfParams = hdu_list[1].data # We read the atmospheric data
        if len(AtmPerfParams["Sec"]) > 0:
            sec_atmos = np.append(sec_atmos, AtmPerfParams["Sec"]) # AtmPerfParams["Sec"] is a list hence this syntax
            r0 = np.append(r0,AtmPerfParams["R0"])
            windSpeed = np.append(windSpeed,AtmPerfParams["WindSpeed"])
            strehl = np.append(strehl,AtmPerfParams["StrehlRatio"])
            file_atmos = np.append(file_atmos,np.repeat(files[i],len(AtmPerfParams["Sec"])))
            date_start_tmp = np.min([date_start_tmp,Time(AtmPerfParams["Sec"][0],format='unix')])
            obstimes = Time(AtmPerfParams["Sec"],format='unix')
            for obstime in obstimes:
                current_coords_altaz = coords_J2000.transform_to(coordinates.AltAz(obstime=obstime,location=location))
                altitude=np.append(altitude,current_coords_altaz.alt.value)
                airmass = np.append(airmass,current_coords_altaz.secz.value)
            if debug:
                print('   {0:3d} atmospheric parameters'.format(len(AtmPerfParams["Sec"])))
        VisLoopParams=hdu_list[2].data
        tmp_flux_VisLoop = [f for f in VisLoopParams['Flux_avg'] if f > 1.]
        tmp_sec_VisLoop = [VisLoopParams["Sec"][i] for i in range(len(VisLoopParams["Sec"])) if VisLoopParams['Flux_avg'][i] > 1.]
        if len(tmp_flux_VisLoop) > 0:
            flux_VisLoop = np.append(flux_VisLoop,tmp_flux_VisLoop)
            sec_VisLoop = np.append(sec_VisLoop,tmp_sec_VisLoop)
            date_start_tmp = np.min([date_start_tmp,Time(tmp_sec_VisLoop[0],format='unix')])
            if debug:
                print('   {0:3d} visible loop parameters'.format(len(tmp_flux_VisLoop)))
        IRLoopParams=hdu_list[3].data
        if len(IRLoopParams['Flux_avg']) != len(IRLoopParams["Sec"]):
            print('Problem: {0:d} IR fluxes with {1:d} time stamps'.format(len(IRLoopParams['Flux_avg']),len(IRLoopParams["Sec"])))
        tmp_flux_IRLoop =  [f for f in IRLoopParams['Flux_avg'] if f > 1.]
        tmp_sec_IRLoop =  [IRLoopParams["Sec"][i] for i in range(len(IRLoopParams["Sec"])) if IRLoopParams['Flux_avg'][i] > 1.]
        if len(tmp_flux_IRLoop) > 0:
            flux_IRLoop = np.append(flux_IRLoop,tmp_flux_IRLoop)
            sec_IRLoop = np.append(sec_IRLoop,tmp_sec_IRLoop)
            date_start_tmp = np.min([date_start_tmp,Time(tmp_sec_IRLoop[0],format='unix')])
            if debug:
                print('   {0:3d} IR loop parameters'.format(len(tmp_flux_IRLoop)))
        data_DTTS = hdu_list['IRPixelAvgFrame'].data
        if len(data_DTTS['Sec'])>0:
            sec_img_DTTS = np.append(sec_img_DTTS,data_DTTS['Sec'])
            cube_DTTS = np.append(cube_DTTS,data_DTTS['Pixels'])
            if debug:
                print('   {0:3d} DTTS images'.format(len(data_DTTS['Sec'])))
        if len(data_DTTS['Pixels']/(32*32)) != len(data_DTTS['Sec']):
            print('Problem: {0:d} DTTS time stamps but {1:d} images'.format(len(data_DTTS['Sec']),len(data_DTTS['Pixels']/(32*32))))

        hdu_list.close()
        # now date_start_tmp is the true starting date so we can append it to the list
        date_start_str = np.append(date_start_str,date_start_tmp.iso)

    # We create the Time objects after converting the list of strings using the unix format
    time_atmos=Time(sec_atmos,format='unix')
    time_atmos.format='isot'
    time_VisLoop=Time(sec_VisLoop,format='unix')
    time_VisLoop.format='isot'
    time_IRLoop=Time(sec_IRLoop,format='unix')
    time_IRLoop.format='isot'
    time_DTTS = Time(sec_img_DTTS,format='unix')
    time_DTTS.format='isot'

    # We reformat the DTTS cube
    cube_DTTS = np.resize(cube_DTTS,(len(cube_DTTS)//(32*32),32,32))

    # We double check that the atmospheric values are valid
    bad_strehl = np.logical_or(np.asarray(strehl)>0.98, np.asarray(strehl)<0.05)
    if np.any(bad_strehl):
        if debug:
            print('   {0:d} bad strehl measurements were detected:'.format(np.sum(bad_strehl)),strehl[bad_strehl])
        strehl[bad_strehl]=np.nan
    bad_r0 = np.logical_or(np.asarray(r0)>0.9, np.asarray(r0)<=0.)
    if np.any(bad_r0):
        if debug:
            print('   {0:d} bad r0 measurements were detected:'.format(np.sum(bad_r0)),r0[bad_r0])
        r0[bad_r0]=np.nan
    bad_windSpeed = np.logical_or(np.asarray(windSpeed)>50, np.asarray(windSpeed)<=0)
    if np.any(bad_windSpeed):
        if debug:
            print('   {0:d} bad equivalent wind velocity measurements were detected:'.format(np.sum(bad_windSpeed)),windSpeed[bad_windSpeed])
            windSpeed[bad_windSpeed]=np.nan

    print('We read in total {0:d} atmospheric parameters among which {1:d=.0f} are valid.'.format(len(time_atmos),np.sum(np.isfinite(strehl)*1.)))
    print('We read in total {0:d} VisLoop parameters'.format(len(time_VisLoop)))
    print('We read in total {0:d} IRLoop parameters'.format(len(time_IRLoop)))
    # We compute tau0 and the seeing from r0
    tau0 = (0.314*np.asarray(r0)/np.asarray(windSpeed))
    seeing = (np.rad2deg(lam/np.asarray(r0))*3600)


    # we store in time_max the time of the last data, and make sure at least one of the
    # arrays (time_atmos,time_VisLoop or time_IRLoop) are populated.
    try:
        time_max = np.max([np.max(time_atmos),np.max(time_IRLoop),np.max(time_VisLoop)])
    except ValueError:
        try:
            time_max = np.max([np.max(time_atmos),np.max(time_VisLoop)])
        except ValueError:
            try:
                time_max = np.max(time_atmos)
            except ValueError:
                print('There are no relevent data in the sparta files')
                return

    # We create new arrays with unique OB names and dates
    new_date_start_str = []
    new_date_end_str = []
    new_name = []
    simbad_dico = {'DEC':[],'RA':[],'simbad_DEC_ICRS':[],'simbad_DEC_current':[],\
                   'simbad_FLUX_H':[],'simbad_FLUX_J':[],'simbad_FLUX_K':[],\
                    'simbad_FLUX_V':[],'simbad_FLUX_R':[],'simbad_FLUX_G':[],'simbad_FLUX_I':[],\
                    'simbad_ID_HD':[],'simbad_MAIN_ID':[],\
                    'simbad_OTYPE':[],'simbad_OTYPE_3':[],'simbad_OTYPE_V':[],\
                    'simbad_PMDEC':[],'simbad_PMRA':[],'simbad_RA_ICRS':[],\
                    'simbad_RA_current':[],'simbad_SP_TYPE':[],\
                    'simbad_separation_RADEC_ICRSJ2000':[],'simbad_separation_RADEC_current':[]}
    for i,namei in enumerate(name):
        if i==0: #the first OB
            simbad_dico_tmp = query_simbad(Time(date_start_str[i]),coords[i],name=name[i])
            if simbad_dico_tmp is not None:
                new_name.append(namei)
                new_date_start_str.append(date_start_str[i])
                for simbad_key in simbad_dico.keys():
                    if simbad_key in simbad_dico_tmp.keys():
                        simbad_dico[simbad_key].append(simbad_dico_tmp[simbad_key])
                    else:
                        simbad_dico[simbad_key].append(np.nan)
                continue
        if namei != name[i-1]: # in case this is the same OB repeated
            txtfilename = files[i].replace('.fits','.NL.txt')
            if os.path.isfile(txtfilename):
                try:
                    with open(txtfilename, "r") as myfile:
                        lines = myfile.readlines()
                    for line in lines:
                        if line.startswith('Target:'):
                            OB_targetname = line[line.index('Target:')+7:].strip()
                            print('Target name: {0:s}'.format(OB_targetname))
                except Exception as e:
                    print('Problem while reading {0:s}. No target name found'.format(txtfilename))
                    print(e)
            else:
                OB_targetname = name[i]
            simbad_dico_tmp = query_simbad(Time(date_start_str[i]),coords[i],name=OB_targetname)
            if simbad_dico_tmp is not None:
                new_name.append(namei)
                new_date_start_str.append(date_start_str[i])
                for simbad_key in simbad_dico.keys():
                    if simbad_key in simbad_dico_tmp.keys():
                        simbad_dico[simbad_key].append(simbad_dico_tmp[simbad_key])
                    else:
                        simbad_dico[simbad_key].append(np.nan)

    for i,namei in enumerate(name[0:-1]):
            if namei != name[i+1]:
                new_date_end_str.append(date_start_str[i+1])
    new_date_end_str.append(time_max.iso)

    # We print the info for each target.
    # We first create arrays (same dimension as new name) to store the median values and later save it.
    start_ob_list = []
    end_ob_list = []
    median_strehl_list=[]
    median_seeing_list = []
    median_tau0_list = []
    median_windSpeed_list = []
    median_r0_list = []
    median_flux_VisLoop_list = []
    print('                                                        Strehl Seeing Tau0    Wind   r0     WFS flux   V mag')
    for i,start_tmp_str in enumerate(new_date_start_str):
        end_tmp_str = new_date_end_str[i]
        start_tmp = Time(start_tmp_str,out_subfmt='date_hm')
        end_tmp = Time(end_tmp_str,out_subfmt='date_hm')
        id_atmos = np.logical_and(time_atmos>start_tmp,time_atmos<end_tmp)
        id_VisWFS =  np.logical_and(time_VisLoop>start_tmp,time_VisLoop<end_tmp)
        try:
            median_strehl = np.nanmedian(strehl[id_atmos])
            median_seeing = np.nanmedian(seeing[id_atmos])
            median_tau0= np.nanmedian(tau0[id_atmos])
            median_windSpeed= np.nanmedian(windSpeed[id_atmos])
            median_r0= np.nanmedian(r0[id_atmos])
            median_flux_VisLoop = np.nanmedian(flux_VisLoop[id_VisWFS])
            print('{0:s} from {1:s} to {2:s} {3:3.1f}%   {4:2.1f}"  {5:04.1f}ms {6:4.1f}m/s {7:3.1f}cm {8:3.1e}ADU {9:3.1f}'.format(\
                  new_name[i].ljust(35)[0:35],\
                   start_tmp.value[11:],end_tmp.value[11:],median_strehl*100,\
                   median_seeing,median_tau0*1000,median_windSpeed,\
                   median_r0*100,median_flux_VisLoop,simbad_dico['simbad_FLUX_V'][i]))
            start_ob_list.append(start_tmp)
            end_ob_list.append(end_tmp)
            median_strehl_list.append(median_strehl)
            median_seeing_list.append(median_seeing)
            median_tau0_list.append(median_tau0)
            median_windSpeed_list.append(median_windSpeed)
            median_r0_list.append(median_r0)
            median_flux_VisLoop_list.append(median_flux_VisLoop)
        except:
            print('{0:s} from {1:s} to {2:s} ---%   --"   ---ms ----m/s  ---cm   ---ADU ----'.format(new_name[i].ljust(35),\
                   start_tmp.value[11:],end_tmp.value[11:]))

    time_file = Time(new_date_start_str)
    nb_obs = len(new_name)

    # we store here the starting date and the ending date
    time_min = time_file[0]
    current_datetime = time_max.datetime
    if current_datetime.hour > 12:
        current_night = current_datetime.date()
    else:
        current_night_datetime = current_datetime - timedelta(days=1)
        current_night = current_night_datetime.date()

    data = [new_name,[str(st) for st in start_ob_list], [str(st) for st in end_ob_list],
                      median_strehl_list, median_seeing_list,
                      median_tau0_list, median_windSpeed_list, median_r0_list,
                      median_flux_VisLoop_list, list(simbad_dico['simbad_FLUX_V'])]
    names = ['OB_name','start_UT','end_UT','strehl_sparta','seeing_los_sparta','tau0_los_sparta',
             'windSpeed_los_sparta','r0_los_sparta','sparta_flux_VisLoop','V_mag']
    file_name = os.path.join(path_output,'summary_{0:s}.csv'.format(str(current_night)))
    ascii.write(data, file_name, names=names,format='csv')#,overwrite=True

    pd_simbad = pd.DataFrame(simbad_dico)
    pd_simbad.to_csv(os.path.join(path_output,'simbad_{0:s}.csv'.format(str(current_night))),index=False)

    # We convert the visWFS flux in photons per subaperture. visWFS is the flux on the whole pupil made of 1240 subapertures.
    nb_subapertures = 1240
    ADU2ph = 270000/2**14#ph/ADU#17 # from Jeff email
    flux_VisLoop = ADU2ph * np.asarray(flux_VisLoop)/gain/nb_subapertures  # in photons per subaperture per frame

    #We interpolate the visWFS flux at the time where the strehl is known
    if len(flux_VisLoop) > 1:
        flux_VisLoop_function = interp1d(time_VisLoop.mjd,flux_VisLoop,kind='linear',bounds_error=False,fill_value=flux_VisLoop[0])
        flux_VisLoop_interpolated = flux_VisLoop_function(time_atmos.mjd)
    else:
        flux_VisLoop_interpolated = flux_VisLoop

    atmos_param_df = pd.DataFrame({'date':time_atmos,'tau0_los_sparta':tau0,\
                                   'seeing_los_sparta':seeing,\
                                   'wind_speed_los_sparta':windSpeed,\
                                   'r0_los_sparta':r0,'strehl_sparta':strehl,\
                                   'interpolated_flux_VisLoop[#photons/aperture/frame]':flux_VisLoop_interpolated,\
                                   'airmass_sparta':airmass,'altitude_sparta':altitude,\
                                   'sparta_file':file_atmos})
    # we compute the zenith seeing: seeing(zenith) = seeing(AM) AM^(3/5)
    atmos_param_df['seeing_zenith_sparta'] = atmos_param_df['seeing_los_sparta']/np.power(atmos_param_df['airmass_sparta'],3./5.)
    atmos_param_df['r0_zenith_sparta'] = atmos_param_df['r0_los_sparta']*np.power(atmos_param_df['airmass_sparta'],3./5.)
    atmos_param_df['tau0_zenith_sparta'] = atmos_param_df['tau0_los_sparta']*np.power(atmos_param_df['airmass_sparta'],3./5.)
    atmos_param_df.to_csv(os.path.join(path_output,'sparta_atmospheric_params_{0:s}.csv'.format(str(current_night))),index=False)

    IRLoop_df = pd.DataFrame({'date':time_IRLoop,'flux_IRLoop_ADU':flux_IRLoop})
    IRLoop_df.to_csv(os.path.join(path_output,'sparta_IR_DTTS_{0:s}.csv'.format(str(current_night))),index=False)
    VisLoop_df = pd.DataFrame({'date':time_VisLoop,
                               'flux_VisLoop[#photons/aperture/frame]':flux_VisLoop,
                               'Frame rate [Hz]':freq})
    VisLoop_df.to_csv(os.path.join(path_output,'sparta_visible_WFS_{0:s}.csv'.format(str(current_night))),index=False)

    # We convert the DTTS time stamps in an dataframe (later we could add more informations
    #directly retrievde from the image...)
    DTTS_df = pd.DataFrame({'date':time_DTTS})
    DTTS_df.to_csv(os.path.join(path_output,'DTTS_cube_{0:s}.csv'.format(str(current_night))))

    primary_hdu = fits.PrimaryHDU(cube_DTTS)
    col1 = fits.Column(name='Sec', format='D',array=sec_img_DTTS)
    cols = fits.ColDefs([col1])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    thdulist = fits.HDUList([primary_hdu, tbhdu])
    thdulist.writeto(os.path.join(path_output,'DTTS_cube_{0:s}.fits'.format(str(current_night))),clobber=True)

    if debug:
        print('Writing atmospheric_params_{0:s}.csv ...'.format(str(current_night)))
        print('Writing visible_WFS_{0:s}.csv ...'.format(str(current_night)))
        print('Writing IR_DTTS_{0:s}.csv ...'.format(str(current_night)))
        print('Writing DTTS_cube_{0:s}.csv ...'.format(str(current_night)))
        print('Writing DTTS_cube_{0:s}.fits ...'.format(str(current_night)))

    # Now we query the ASM database to get the tau0 from the MASS-DIMM
    start_date_asm_str = time_min.iso.replace(' ','T')
    end_date_asm_str = time_max.iso.replace(' ','T')
    print('Querying mass-dimm data')
    request_asm_str = ['wget','-O',os.path.join(path_output,'mass_dimm_{0:s}.csv'.format(str(current_night))),\
                       'http://archive.eso.org/wdb/wdb/asm/mass_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&tab_fwhm=1&tab_fwhmerr=0&tab_tau=1&tab_tauerr=0&tab_tet=0&tab_teterr=0&tab_alt=0&tab_alterr=0&tab_fracgl=1&tab_turbfwhm=1&tab_tau0=1&tab_tet0=0&tab_turb_alt=0&tab_turb_speed=1'.format(start_date_asm_str,end_date_asm_str)]
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
        mass_df = pd.read_csv(os.path.join(path_output,'mass_dimm_{0:s}.csv'.format(str(current_night))),skiprows=1,skipfooter=5,engine='python')
        if debug:
            if len(mass_df.keys())<2:
                print('No data to be read in the mass-dimm file.')
                raise IOError('Empty data in {0:s}'.format(os.path.join(path_output,'mass_dimm_{0:s}.csv'.format(str(current_night)))))
            else:
                print('The MASS-DIMM file contains {0:d} values.'.format(len(mass_df)))
        mass_df.rename(columns={'Date time': 'date',\
                        'MASS Tau0 [s]':'MASS_tau0',\
                        'MASS-DIMM Cn2 fraction at ground':'MASS-DIMM_fracgl',\
                        'MASS-DIMM Tau0 [s]':'MASS-DIMM_tau0',\
                        'MASS-DIMM Turb Velocity [m/s]':'MASS-DIMM_turb_speed',\
                        'MASS-DIMM Seeing ["]':'MASS-DIMM_seeing',\
                        'Free Atmosphere Seeing ["]':'MASS_freeatmos_seeing'}, inplace=True)
        time_mass_dimm_asm = Time(list(mass_df['date']),format='isot',scale='utc')
        mass_df.to_csv(os.path.join(path_output,'mass_dimm_{0:s}.csv'.format(str(current_night))),index=False)
    except Exception as e:
        time_mass_dimm_asm=None
        if debug:
            print(e)
            print("The plot won't contain any MASS-DIMM data.")

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

    # Now we read the DIMM file
    print('Querying dimm data')
    try:
        dimm_df = pd.read_csv(os.path.join(path_output,'dimm_{0:s}.csv'.format(str(current_night))),skiprows=1,skipfooter=5,engine='python')
        if debug:
            if len(dimm_df.keys())<2:
                print('No data to be read in the dimm file.')
                raise IOError('Empty data in {0:s}'.format(os.path.join(path_output,'dimm_{0:s}.csv'.format(str(current_night)))))
            else:
                print('The DIMM file contains {0:d} values.'.format(len(dimm_df)))
        dimm_df.rename(columns={'Date time': 'date',\
                        'DIMM Seeing ["]':'dimm_seeing'}, inplace=True)
        time_dimm_asm = Time(list(dimm_df['date']),format='isot',scale='utc')
        dimm_df.to_csv(os.path.join(path_output,'dimm_{0:s}.csv'.format(str(current_night))),index=False)
    except Exception as e:
        time_dimm_asm=None
        if debug:
            print(e)
            print("The plot won't contain any DIMM data.")

    # Now we query the old dimm in case data were taken before 2016-04-04T10:08:39
    if time_min<Time('2016-04-05 00:00:00'):
        print('Querying old dimm data')
        request_asm_str = ['wget','-O',os.path.join(path_output,'old_dimm_{0:s}.csv'.format(str(current_night))),\
                           'http://archive.eso.org/wdb/wdb/asm/historical_ambient_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&tab_fwhm=1&tab_airmass=0&tab_rfl=0&tab_tau=1&tab_tet=0&top=1000000'.format(\
                           start_date_asm_str,end_date_asm_str)]
        output,error = subprocess.Popen(request_asm_str,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
        if debug:
            print(' '.join(request_asm_str))
            print(output.decode('UTF8'))
        if error != None:
            print('Error during the request of the ASM OLD DIMM database:')
            print(error)
        # Now we read the old DIMM file
        try:
            old_dimm_df = pd.read_csv(os.path.join(path_output,'old_dimm_{0:s}.csv'.format(str(current_night))),skiprows=1,skipfooter=5,engine='python')
            if debug:
                if len(old_dimm_df.keys())<2:
                    print('No data to be read in the old dimm file.')
                    raise IOError('Empty data in {0:s}'.format(os.path.join(path_output,'old_dimm_{0:s}.csv'.format(str(current_night)))))
                else:
                    print('The old DIMM file contains {0:d} values.'.format(len(old_dimm_df)))
            old_dimm_df.rename(columns={'Date time': 'date',\
                        'DIMM Seeing ["]':'old_dimm_seeing',\
                        'Tau0 [s]':'old_dimm_tau0'}, inplace=True)
            time_old_dimm_asm = Time(list(old_dimm_df['date']),format='isot',scale='utc')
            old_dimm_df.to_csv(os.path.join(path_output,'old_dimm_{0:s}.csv'.format(str(current_night))),index=False)
        except Exception as e:
            time_old_dimm_asm=None
            if debug:
                print(e)
                print("The plot won't contain any old DIMM data.")

    # Now we query the ASM database to get the seeing from the SLODAR
    print('Querying slodar data')
    request_asm_str = ['wget','-O',os.path.join(path_output,'slodar_{0:s}.csv'.format(str(current_night))),\
            'http://archive.eso.org/wdb/wdb/asm/slodar_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&tab_cnsqs_uts=1&tab_fracgl300=1&tab_fracgl500=1&tab_hrsfit=1&tab_fwhm=1'.format(start_date_asm_str,end_date_asm_str)]
    output,error = subprocess.Popen(request_asm_str,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
    if debug:
        print(' '.join(request_asm_str))
        print(output.decode('UTF8'))
    if error != None:
        print('Error during the request of the ASM SLODAR database:')
        print(error)

    # Now we read the SLODAR file
    try:
        slodar_df = pd.read_csv(os.path.join(path_output,'slodar_{0:s}.csv'.format(str(current_night))),skiprows=1,skipfooter=5,engine='python')
        if debug:
            if len(slodar_df.keys())<2:
                print('No data to be read in the slodar file.')
                raise IOError('Empty data in {0:s}'.format(os.path.join(path_output,'slodar_{0:s}.csv'.format(str(current_night)))))
            else:
                print('The slodar file contains {0:d} values.'.format(len(slodar_df)))
        slodar_df.rename(columns={'Date time': 'date','Cn2 above UTs [10**(-15)m**(1/3)]':'Cn2_above_UT',\
                          'Cn2 fraction below 300m':'slodar_GLfrac_300m',\
                          'Cn2 fraction below 500m':'slodar_GLfrac_500m',\
                          'Surface layer profile [10**(-15)m**(1/3)]':'slodar_surface_layer',\
                          'Seeing ["]':'slodar_seeing'}, inplace=True)
        wave_nb=2*np.pi/lam
        time_slodar_asm = Time(list(slodar_df['date']),format='isot',scale='utc')
        slodar_df['slodar_r0_above_UT'] = np.power(0.423*(wave_nb**2)*slodar_df['Cn2_above_UT']*1.e-15,-3./5.)
        slodar_df['slodar_seeing_above_UT']= np.rad2deg(lam/slodar_df['slodar_r0_above_UT'])*3600.
        slodar_df['slodar_Cn2_total'] = np.power(slodar_df['slodar_seeing']/2.e7,1./0.6) # in m^1/3
        slodar_df['slodar_surface_layer_fraction'] = slodar_df['slodar_surface_layer']*1.e-15 / slodar_df['slodar_Cn2_total']
        slodar_df.to_csv(os.path.join(path_output,'slodar_{0:s}.csv'.format(str(current_night))),index=False)
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

    # Now we query the telescope seeing
    print('Querying SPHERE header data')
    request_sphere = ['wget','-O',os.path.join(path_output,'sphere_ambi_{0:s}.csv'.format(str(current_night))),\
            'http://archive.eso.org/wdb/wdb/eso/sphere/query?wdbo=csv&night={0:s}..{1:s}&tab_prog_id=0&tab_dp_id=0&tab_ob_id=0&tab_exptime=0&tab_dp_cat=0&tab_tpl_start=0&tab_dp_type=0&tab_dp_tech=0&tab_seq_arm=0&tab_ins3_opti5_name=0&tab_ins3_opti6_name=0&tab_ins_comb_vcor=0&tab_ins_comb_iflt=0&tab_ins_comb_pola=0&tab_ins_comb_icor=0&tab_det_dit1=0&tab_det_seq1_dit=0&tab_det_ndit=0&tab_det_read_curname=0&tab_ins2_opti2_name=0&tab_det_chip_index=0&tab_ins4_comb_rot=0&tab_ins1_filt_name=0&tab_ins1_opti1_name=0&tab_ins1_opti2_name=0&tab_ins4_opti11_name=0&tab_tel_ia_fwhm=1&tab_tel_ia_fwhmlin=1&tab_tel_ia_fwhmlinobs=1&tab_tel_ambi_windsp=0&tab_night=1&tab_fwhm_avg=0&top=1000'.format(start_date_asm_str,end_date_asm_str)]
    output,error = subprocess.Popen(request_sphere,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
    if debug:
        print(' '.join(request_sphere))
        print(output.decode('UTF8'))
    if error != None:
        print('Error during the request of the SPHERE database:')
        print(error)

    try:
        sphere_df = pd.read_csv(os.path.join(path_output,'sphere_ambi_{0:s}.csv'.format(str(current_night))),skiprows=1,skipfooter=5,engine='python') # 1st line is bank
        if debug:
            if len(sphere_df.keys())<2:
                print('No data to be read in the SPHERE header file.')
                raise IOError('Empty data in {0:s}'.format(os.path.join(path_output,'sphere_ambi_{0:s}.csv'.format(str(current_night)))))
            else:
                print('The SPHERE header file contains {0:d} values.'.format(len(sphere_df)))
        sphere_df.rename(columns={'DATE OBS': 'date'}, inplace=True) # To make it compatible with the other csv files.
        sphere_keys_to_drop= ['Release Date','Object','RA','DEC','Target Ra Dec','Target l b','OBS Target Name']
        for sphere_key_to_drop in sphere_keys_to_drop:
            sphere_df.drop(sphere_key_to_drop, axis=1, inplace=True)
        time_sphere = Time(list(sphere_df['date']),format='iso',scale='utc')
        sphere_df.to_csv(os.path.join(path_output,'sphere_ambi_{0:s}.csv'.format(str(current_night))),index=False)
    except Exception as e:
        if debug:
            print(e)
            print("The plot won't contain any data from the SPHERE science files headers.")

    # Now we query the meteo tower to get the wind speed, direction and temperature
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

    # Now we query the lathpro
    print('Querying Lathpro data')
    request_lathpro = ['wget','-O',os.path.join(path_output,'lathpro_{0:s}.csv'.format(str(current_night))),\
            'http://archive.eso.org/wdb/wdb/asm/lhatpro_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&tab_integration=0&tab_lwp0=0'.format(start_date_asm_str,end_date_asm_str)]
    output,error = subprocess.Popen(request_lathpro,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
    if debug:
        print(' '.join(request_lathpro))
        print(output.decode('UTF8'))
    if error != None:
        print('Error during the request of the Lathpro database:')
        print(error)
    try:
        lathpro_df = pd.read_csv(os.path.join(path_output,'lathpro_{0:s}.csv'.format(str(current_night))),skiprows=1,skipfooter=5,engine='python') # 1st line is bank
        if debug:
            if len(lathpro_df.keys())<2:
                print('No data to be read in the Lathpro file.')
                raise IOError('Empty data in {0:s}'.format(os.path.join(path_output,'lathpro_{0:s}.csv'.format(str(current_night)))))
            else:
                print('The Lathpro file contains {0:d} values.'.format(len(lathpro_df)))
        lathpro_df.rename(columns={'Date time': 'date','IR temperature [Celsius]':'lathpro_IR_temperature[Celsius]',\
                          'Precipitable Water Vapour [mm]':'lathpro_pwv[mm]'}, inplace=True)
        lathpro_df.to_csv(os.path.join(path_output,'lathpro_{0:s}.csv'.format(str(current_night))),index=False)
    except Exception as e:
        if debug:
            print(e)
            print("The plot won't contain any data from the Lathpro")

    # Now we query the ECMWF data
    print('Querying ECMWF data')
    pd_ecmwf = request_ecmwf(time_min,time_max)
    if pd_ecmwf is not None:
        pd_ecmwf.to_csv(os.path.join(path_output,'ecmwf_{0:s}.csv'.format(str(current_night))),index=False)
        time_ecmwf = Time(list(pd_ecmwf['date']))#,format='isot',scale='utc')

    if plot:
        majorFormatter = mpl.dates.DateFormatter('%H:%M')
        plt.close(1)
        fig = plt.figure(1, figsize=(12,15))
        plt.rcParams.update({'font.size':14})

        gs = gridspec.GridSpec(4,2, height_ratios=[1,1,1,1],)
        gs.update(left=0.1, right=0.95, bottom=0.1, top=0.98, wspace=0.2, hspace=0.3)

        ax1 = plt.subplot(gs[0,0]) # Area for the seeing
        ax2 = plt.subplot(gs[0,1]) # Area for the r0
        ax3 = plt.subplot(gs[1,0]) # Area for the Flux
        ax4 = plt.subplot(gs[1,1]) # Area for the Strehl
        ax5 = plt.subplot(gs[2:4,0]) # Area for the combined plot
        ax6 = plt.subplot(gs[2,1]) # Area for the GL frac
        ax7 = plt.subplot(gs[3,1]) # Area for the wind speed

        # Plot the seeing in the graph 1
        try:
            ax1.plot_date(time_atmos.plot_date,atmos_param_df['seeing_zenith_sparta'],'.', color='darkorange',markeredgecolor='none',label='sparta')
        except Exception as e:
            if debug:
                print('No Sparta seeing available for the plot 1: {0:s}'.format(str(e)))
        try:
            ax1.plot_date(time_mass_dimm_asm.plot_date,mass_df['MASS-DIMM_seeing'],'.', color='palevioletred',\
                          markeredgecolor='none',label='MASS-DIMM')
        except Exception as e:
            if debug:
                print('No MASS-DIMM seeing available for the plot 1: {0:s}'.format(str(e)))
        try:
            ax1.plot_date(time_mass_dimm_asm.plot_date,mass_df['MASS_freeatmos_seeing'],'.', color='lime',markeredgecolor='none',label='MASS')
        except Exception as e:
            if debug:
                print('No MASS free atmosphere seeing available for the plot 1: {0:s}'.format(str(e)))
        try:
            ax1.plot_date(time_dimm_asm.plot_date,dimm_df['dimm_seeing'],'.', color='dimgrey',markeredgecolor='none',label='DIMM')
        except Exception as e:
            if debug:
                print('No DIMM data available for the plot 1: {0:s}'.format(str(e)))
        try:
            ax1.plot_date(time_old_dimm_asm.plot_date,old_dimm_df['old_dimm_seeing'],'.', color='dimgrey',markeredgecolor='none',label='old DIMM')
        except Exception as e:
            if debug:
                print('No old DIMM data available for the plot 1: {0:s}'.format(str(e)))
        try:
            # I commented the SLODAR alone since only the slodar above UT matters
            #ax1.plot_date(time_slodar_asm.plot_date,asm_slodar_file['seeing'],'.', color='darkkhaki',markeredgecolor='none',label='SLODAR')
            ax1.plot_date(time_slodar_asm.plot_date,slodar_df['slodar_seeing_above_UT'],'.', color='magenta',\
                          markeredgecolor='none',label='SLODAR above UT')
        except Exception as e:
            if debug:
                print('No SLODAR data available for the plot 1: {0:s}'.format(str(e)))
        try:
#            ax1.plot_date(time_sphere.plot_date,sphere_df['TEL IA FWHM'],'.', color='rosybrown',markeredgecolor='none',label='TEL.IA.FWHM')
            ax1.plot_date(time_sphere.plot_date,sphere_df['TEL IA FWHMLIN'],'.', color='rosybrown',markeredgecolor='none',label='TEL.IA.FWHMLIN')
        except Exception as e:
            if debug:
                print('No TEL IA FWHM data available for the plot 1: {0:s}'.format(str(e)))
        ax1.set_ylabel('Seeing (arcsec)')
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
        # add filling for even OBs
        min_y,max_y = ax1.get_ybound()
        for i in range(0,nb_obs-2,2):
            ax1.fill_between(time_file[i:i+2].plot_date,0,max_y, facecolor='blue', alpha=0.1)
        if np.mod(nb_obs,2)==0: #then you add the last one which is defined
            ax1.fill_between(time_file[nb_obs-2:nb_obs].plot_date,0,max_y, facecolor='blue', alpha=0.1)
        else:
            ax1.fill_between([time_file[-1].plot_date,time_atmos[-1].plot_date],0,100, facecolor='blue', alpha=0.2)
        ax1.set_ylim(min_y,max_y)
        ax1.set_xlim(time_min.plot_date,time_max.plot_date)
        ax1.grid()
        ax1.legend(frameon=False,loc='best',fontsize=10)
        ax1.xaxis.set_major_formatter(majorFormatter)

        # Plot the tau0 in the graph 2.
        try:
            ax2.plot_date(time_atmos.plot_date,atmos_param_df['tau0_zenith_sparta']*1000,'.', color='darkgreen',markeredgecolor='none',label='sparta')
        except Exception as e:
            if debug:
                print('No Sparta data available for the plot 2: {0:s}'.format(str(e)))
        try:
            ax2.plot_date(time_mass_dimm_asm.plot_date,mass_df['MASS-DIMM_tau0']*1000.,'.',color='palevioletred',label='MASS-DIMM',markeredgecolor='none')
            ax2.plot_date(time_mass_dimm_asm.plot_date,mass_df['MASS_tau0']*1000.,'.',color='dimgrey',label='MASS',markeredgecolor='none')
        except Exception as e:
            if debug:
                print('No MASS-DIMM data available for the plot 2: {0:s}'.format(str(e)))
        try:
            ax2.plot_date(time_old_dimm_asm.plot_date,old_dimm_df['old_dimm_tau0']*1000.,'.',color='palevioletred',label='old DIMM',markeredgecolor='none')
        except Exception as e:
            if debug:
                print('No old DIMM data available for the plot 2: {0:s}'.format(str(e)))

        ax2.set_ylabel('$\\tau_0$ (ms)')
        for tick in ax2.get_xticklabels():
            tick.set_rotation(45)
        # add filling for even OBs
        min_y,max_y = ax2.get_ybound()
        for i in range(0,nb_obs-2,2):
            ax2.fill_between(time_file[i:i+2].plot_date,min_y,max_y, facecolor='blue', alpha=0.1)
        if np.mod(nb_obs,2)==0: #then you add the last one which is defined
            ax2.fill_between(time_file[nb_obs-2:nb_obs].plot_date,min_y,max_y, facecolor='blue', alpha=0.1)
        else:
            ax2.fill_between([time_file[-1].plot_date,time_atmos[-1].plot_date],min_y,max_y, facecolor='blue', alpha=0.1)
        ax2.set_xlim(time_min.plot_date,time_max.plot_date)
        ax2.set_ylim(min_y,max_y)
        ax2.grid()
        ax2.legend(frameon=False,loc='best',fontsize=10)
        ax2.xaxis.set_major_formatter(majorFormatter)

        # plot the DDT flux in the graph 3
        ax3.plot_date(time_IRLoop.plot_date,flux_IRLoop,\
            '.', color='red',markeredgecolor='none',label='IR DTTS')
        ax3.plot_date(time_VisLoop.plot_date,flux_VisLoop,\
            '.', color='blue',markeredgecolor='none',label='Vis WFS')
        ymin = np.percentile(np.append(flux_IRLoop,flux_VisLoop),10)/10
        ymax = np.percentile(np.append(flux_IRLoop,flux_VisLoop),90)*10
        ax3.set_ylim(ymin,ymax)
        ax3.set_yscale("log", nonposy='clip')
        ax3.set_ylabel('Flux in photons/aperture')
        ax3.legend(frameon=False,loc='best')
        for tick in ax3.get_xticklabels():
            tick.set_rotation(45)
        # add filling for even OBs
        for i in range(0,nb_obs-2,2):
            ax3.fill_between(time_file[i:i+2].plot_date,0,ax3.get_ybound()[1], facecolor='blue', alpha=0.1)
        if np.mod(nb_obs,2)==0: #then you add the last one which is defined
            ax3.fill_between(time_file[nb_obs-2:nb_obs].plot_date,0,ax3.get_ybound()[1], facecolor='blue', alpha=0.1)
        else:
            ax3.fill_between([time_file[-1].plot_date,time_atmos[-1].plot_date],0,ax3.get_ybound()[1], facecolor='blue', alpha=0.1)
        ax3.grid()
        ax3.set_xlim(time_min.plot_date,time_max.plot_date)
        ax3.xaxis.set_major_formatter(majorFormatter)

        # plot the strehl in the graph 4
        ax4.plot_date(time_atmos.plot_date,atmos_param_df['strehl_sparta']*100,'.', color='darkorchid',markeredgecolor='none')
        ax4.set_ylabel('Strehl (%)')
        ax4.set_ylim(0,100)
        for i in range(0,nb_obs-2,2):
            ax4.fill_between(time_file[i:i+2].plot_date,0,100, facecolor='blue', alpha=0.1)
        if np.mod(nb_obs,2)==0: #then you add the last one which is defined
            ax4.fill_between(time_file[nb_obs-2:nb_obs].plot_date,0,100, facecolor='blue', alpha=0.1)
        else:
            ax4.fill_between([time_file[-1].plot_date,time_atmos[-1].plot_date],0,100, facecolor='blue', alpha=0.1)
        for tick in ax4.get_xticklabels():
            tick.set_rotation(45)
        ax4.grid()
        ax4.set_xlim(time_min.plot_date,time_max.plot_date)
        ax4.xaxis.set_major_formatter(majorFormatter)

        # plot the strehl, seeing and flux in the graph 5
        med_lin_flux = np.median(flux_VisLoop_interpolated)
        size_strehl = flux_VisLoop_interpolated/med_lin_flux*20
        ax5.plot_date(time_atmos.plot_date,seeing,'.', color='darkorange',markeredgecolor='none',label='seeing (")')
        ax5.scatter(time_atmos.plot_date,strehl, color='darkorchid',s=size_strehl,marker='s',alpha=0.3)#,edgecolor='black')
        ax5.plot_date(time_atmos.plot_date,tau0*100,'.', color='darkgreen',markeredgecolor='none',label='$\\tau_0$ (sx100)')
        min_y,max_y = ax5.get_ybound()
        for tick in ax5.get_xticklabels():
            tick.set_rotation(45)
        ax5.set_xlabel('Night of {0:s}'.format(str(current_night)))
        ax5.set_ylabel('Seeing in arcsec, $\\tau_0/10$ in ms and Strehl (size prop. to WFS flux)')
        for i in range(0,nb_obs-2,2):
            ax5.fill_between(time_file[i:i+2].plot_date,min_y,max_y, facecolor='blue', alpha=0.1)
        if np.mod(nb_obs,2)==0: #then you add the last one which is defined
            ax5.fill_between(time_file[nb_obs-2:nb_obs].plot_date,min_y,max_y, facecolor='blue', alpha=0.1)
        else:
            ax5.fill_between([time_file[-1].plot_date,time_atmos[-1].plot_date],min_y,max_y, facecolor='blue', alpha=0.1)
        for i,time_filei in enumerate(time_file):
            ax5.text(time_filei.plot_date, max_y-(max_y-min_y)*0.3,new_name[i], fontsize=10,rotation=90)#min_y+(max_y-min_y)/5.
        for tick in ax5.get_xticklabels():
            tick.set_rotation(45)
        ax5.grid()
        ax5.set_ylim(min_y,np.max([2,max_y]))
        ax5.set_xlim(time_min.plot_date,time_max.plot_date)
        ax5.xaxis.set_major_formatter(majorFormatter)

        # Plot the ground layer and surface layer fraction in the graph 6
        try:
            ax6.plot_date(time_mass_dimm_asm.plot_date,np.asarray(mass_df['MASS-DIMM_fracgl'])*100.,'.',color='palevioletred',label='MASS-DIMM',markeredgecolor='none')
        except Exception as e:
            if debug:
                print('No MASS-DIMM data available for the plot 6: {0:s}'.format(str(e)))
        try:
            ax6.plot_date(time_slodar_asm.plot_date,slodar_df['slodar_surface_layer_fraction']*100,'.', color='black',markeredgecolor='none',label='SLODAR surface layer')
            ax6.plot_date(time_slodar_asm.plot_date,slodar_df['slodar_GLfrac_500m']*100,'.', color='red',markeredgecolor='none',label='SLODAR 500m')
            ax6.plot_date(time_slodar_asm.plot_date,slodar_df['slodar_GLfrac_300m']*100,'.', color='blue',markeredgecolor='none',label='SLODAR 300m')
        except Exception as e:
            if debug:
                print('No SLODAR data available for the plot 6: {0:s}'.format(str(e)))
        ax6.set_ylabel('Ground layer fraction (%)')
        for tick in ax6.get_xticklabels():
            tick.set_rotation(45)
        # add filling for even OBs
        min_y,max_y = [0,100] #ax6.get_ybound()
        for i in range(0,nb_obs-2,2):
            ax6.fill_between(time_file[i:i+2].plot_date,min_y,max_y, facecolor='blue', alpha=0.1)
        if np.mod(nb_obs,2)==0: #then you add the last one which is defined
            ax6.fill_between(time_file[nb_obs-2:nb_obs].plot_date,min_y,max_y, facecolor='blue', alpha=0.1)
        else:
            ax6.fill_between([time_file[-1].plot_date,time_atmos[-1].plot_date],min_y,max_y, facecolor='blue', alpha=0.1)
        ax6.set_xlim(time_min.plot_date,time_max.plot_date)
        ax6.set_ylim(min_y,max_y)
        ax6.grid()
        ax6.legend(frameon=False,loc='best',fontsize=10)
        ax6.xaxis.set_major_formatter(majorFormatter)

        # Plot the wind speed in the graph 7
        ax7.plot_date(time_atmos.plot_date,windSpeed,'.', color='darkgreen',markeredgecolor='none',label='sparta')
        try:
            ax7.plot_date(time_mass_dimm_asm.plot_date,mass_df['MASS-DIMM_turb_speed'],'.',color='palevioletred',label='MASS',markeredgecolor='none')
        except Exception as e:
            if debug:
                print('No MASS-DIMM data available for the plot 7: {0:s}'.format(str(e)))
#        try:
#            ax7.plot_date(time_sphere.plot_date,sphere_df['TEL AMBI WINDSP'],'.', color='rosybrown',markeredgecolor='none',label='TEL.AMBI.WINDSPD')
#        except Exception as e:
#            if debug:
#                print('No SPHERE data available for the plot 7: {0:s}'.format(str(e)))
        try:
            ax7.plot_date(time_asm.plot_date,asm_df['windspeed_30m'],'.',color='rosybrown',label='ASM 30m',markeredgecolor='none')
        except Exception as e:
            if debug:
                print('No ASM data available for the plot 7: {0:s}'.format(str(e)))
        try:
            ax7.plot_date(time_ecmwf.plot_date,pd_ecmwf['ecmwf_200mbar_windspeed[m/s]']*0.4,'.',color='blue',label='200mbar * 0.4',markeredgecolor='none')
        except Exception as e:
            if debug:
                print('No ECMWF data available for the plot 7: {0:s}'.format(str(e)))

        ax7.set_ylabel('Wind speed (m/s)')
        for tick in ax7.get_xticklabels():
            tick.set_rotation(45)
        # add filling for even OBs
        min_y,max_y = ax7.get_ybound()
        for i in range(0,nb_obs-2,2):
            ax7.fill_between(time_file[i:i+2].plot_date,min_y,max_y, facecolor='blue', alpha=0.1)
        if np.mod(nb_obs,2)==0: #then you add the last one which is defined
            ax7.fill_between(time_file[nb_obs-2:nb_obs].plot_date,min_y,max_y, facecolor='blue', alpha=0.1)
        else:
            ax7.fill_between([time_file[-1].plot_date,time_atmos[-1].plot_date],min_y,max_y, facecolor='blue', alpha=0.1)
        ax7.set_xlim(time_min.plot_date,time_max.plot_date)
        ax7.set_ylim(min_y,max_y)
        ax7.grid()
        ax7.legend(frameon=False,loc='best',fontsize=10)
        ax7.xaxis.set_major_formatter(majorFormatter)

        fig.savefig(os.path.join(path_output,'sparta_plot_{0:s}.pdf'.format(str(current_night))))
        #plt.close(1)


    return

def convert_keyword_coord(keyword_coord):
    """
    Convert a keyword of type -124700.06 into a string "-12:47:00.06" readable
    by astropy.coordinates
    Input:
        - keyword_coord of type float (example -124700.06 as retrived from
        header['HIERARCH ESO TEL TARG ALPHA'] or header['HIERARCH ESO TEL TARG DEC'])
    Output:
        - formatted string directly readable by astropy.coordinates (example: "-12:47:00.06")
    """
    if type(keyword_coord) != float:
        raise TypeError('The argument {0} is not a float'.format(keyword_coord))
    if keyword_coord<0:
        keyword_coord_str = '{0:012.4f}'.format(keyword_coord) #0 padding with 6 digits (11-4-1-1) for the part before the point.
        keyword_formatted = '{0:s}:{1:s}:{2:s} '.format(keyword_coord_str[0:3],
                keyword_coord_str[3:5],keyword_coord_str[5:])
    else:
        keyword_coord_str = '{0:011.4f}'.format(keyword_coord) #0 padding with 6 digits (11-4-1) for the part before the point.
        keyword_formatted = '{0:s}:{1:s}:{2:s} '.format(keyword_coord_str[0:2],
                keyword_coord_str[2:4],keyword_coord_str[4:])
    return keyword_formatted


if __name__ == "__main__":
#    date = sys.argv[1]
#    path_raw = os.path.join('/data-ut3/raw',date)
#    path_sparta = '/diska/home/astro3/PrivateData/jmilli/sparta'
#    path_output = os.path.join(path_sparta,date)
#    plot_sparta_data(path_raw=path_raw,path_output=path_output,debug=False)
    debug=False
    path_raw=None
    path_output=None
    try:
        opts, args = getopt.getopt(sys.argv[1:],"i:o:d:",["ifolder=","ofolder=","debug="])
    except Exception as e:
        print(e)
        print('Syntax error')
        print('plot_sparta_data.py -i <inputfolder> -o <outputfolder> debug=True/False')
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
        date = sys.argv[1]
        path_raw = os.path.join('/data-ut3/raw',date)
        path_sparta = '/diska/home/astro3/PrivateData/jmilli/sparta'
        path_output = os.path.join(path_sparta,date)
    plot_sparta_data(path_raw=path_raw,path_output=path_output,debug=debug)



