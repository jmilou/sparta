#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 01:48:04 2017
@author: jmilli
"""

import pandas as pd
import os
import numpy as np
import subprocess
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
import astropy.coordinates as coord
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import ICRS, FK5 #,FK4, Galactic
Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source" # Select early Data Release 3
path_data = os.path.join(os.path.dirname(os.path.abspath(__file__)),'sparta_data')

def query_dimm(path,start_date='2017-04-28T00:00:00.00',\
               end_date='2017-05-01T12:00:00.00'):
    """
    Query the ASM dimm archive (new DIMM available from 2016-04-02 onwards),
    saves the result in a csv file and returns the table as a panda data frame.
    We limit the search to 1 000 000 lines. If more is expected, pay attention
    Input:
        - path: path where to save the csv table
        - start_date: the date (and time) for the query start, in the isot format
                    (for instance '2017-04-28T00:00:00.00')
        - end_date: the date (and time) for the query end, in the isot format
                    (for instance '2017-04-28T00:00:00.00')                    
    """
    filename = 'dimm_query_{0:s}_{1:s}.csv'.format(start_date,end_date)
    request_asm_str = ['wget','-O',os.path.join(path,filename),\
                       'http://archive.eso.org/wdb/wdb/asm/dimm_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&tab_fwhm=1&tab_rfl=0&tab_rfl_time=0&top=1000000'.format(\
                       start_date,end_date)]
    output,error = subprocess.Popen(request_asm_str,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
    print(' '.join(request_asm_str))
    print(output.decode('ISO-8859-1'))
    df = pd.read_csv(os.path.join(path,filename),skiprows=1,skipfooter=5,\
                     parse_dates=True, index_col='Date time',\
                     infer_datetime_format=True)
    return df

def query_old_dimm(path,start_date='2015-04-28T00:00:00.00',\
               end_date='2015-05-01T12:00:00.00'):
    """
    Query the ASM old dimm archive (old DIMM available up to 2016-04-01),
    saves the result in a csv file and returns the table as a panda data frame.
    We limit the search to 1 000 000 lines. If more is expected, pay attention
    Input:
        - path: path where to save the csv table
        - start_date: the date (and time) for the query start, in the isot format
                    (for instance '2015-04-28T00:00:00.00')
        - end_date: the date (and time) for the query end, in the isot format
                    (for instance '2015-04-28T00:00:00.00')                    
    """
    filename = 'old_dimm_query_{0:s}_{1:s}.csv'.format(start_date,end_date)
    request_asm_str = ['wget','-O',os.path.join(path,filename),\
                       'http://archive.eso.org/wdb/wdb/asm/historical_ambient_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&tab_fwhm=1&tab_airmass=0&tab_rfl=0&tab_tau=1&tab_tet=0&top=1000000'.format(\
                        start_date,end_date)]    
    output,error = subprocess.Popen(request_asm_str,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
    print(' '.join(request_asm_str))
    print(output.decode('ISO-8859-1'))
    df = pd.read_csv(os.path.join(path,filename),skiprows=1,skipfooter=5,\
                     parse_dates=True, index_col='Date time',\
                     infer_datetime_format=True)
    return df

def query_mass(path,start_date='2017-04-28T00:00:00.00',\
               end_date='2017-05-01T12:00:00.00'):
    """
    Query the ASM mass-dimm archive (available from April 2016 onwards, with a gap 
     between 2017-02-01 (noon) and 2017-05-19 (noon) (UT)),
    saves the result in a csv file and returns the table as a panda data frame.
    We limit the search to 1 000 000 lines. If more is expected, pay attention
    Input:
        - path: path where to save the csv table
        - start_date: the date (and time) for the query start, in the isot format
                    (for instance '2017-04-28T00:00:00.00')
        - end_date: the date (and time) for the query end, in the isot format
                    (for instance '2017-04-28T00:00:00.00')                    
    """
    filename = 'mass_query_{0:s}_{1:s}.csv'.format(start_date,end_date)
    request_asm_str = ['wget','-O',os.path.join(path,filename),\
                       'http://archive.eso.org/wdb/wdb/asm/mass_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&tab_fwhm=1&tab_fwhmerr=0&tab_tau=1&tab_tauerr=0&tab_tet=1&tab_teterr=0&tab_alt=1&tab_alterr=0&tab_fracgl=1&tab_turbfwhm=1&tab_tau0=1&tab_tet0=1&tab_turb_alt=1&tab_turb_speed=1&top=1000000'.format(\
                       start_date,end_date)]
    output,error = subprocess.Popen(request_asm_str,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
    print(' '.join(request_asm_str))
    print(output.decode('ISO-8859-1'))
    df = pd.read_csv(os.path.join(path,filename),skiprows=1,skipfooter=5,\
                     parse_dates=True, index_col='Date time',\
                     infer_datetime_format=True)
    return df

def query_slodar(path,start_date='2017-04-28T00:00:00.00',\
               end_date='2017-05-01T12:00:00.00'):
    """
    Query the ASM slodar archive (available from April 2016 onwards,
    saves the result in a csv file and returns the table as a panda data frame.
    We limit the search to 1 000 000 lines. If more is expected, pay attention
    Input:
        - path: path where to save the csv table
        - start_date: the date (and time) for the query start, in the isot format
                    (for instance '2017-04-28T00:00:00.00')
        - end_date: the date (and time) for the query end, in the isot format
                    (for instance '2017-04-28T00:00:00.00')                    
    """
    filename = 'slodar_query_{0:s}_{1:s}.csv'.format(start_date,end_date)
    request_slodar_str = ['wget','-O',os.path.join(path,filename),\
                       'http://archive.eso.org/wdb/wdb/asm/slodar_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&tab_cnsqs_uts=1&tab_fracgl300=1&tab_fracgl500=1&tab_fwhm=1&tab_hrsfit=1&top=1000000'.format(\
                       start_date,end_date)]
    output,error = subprocess.Popen(request_slodar_str,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
    print(' '.join(request_slodar_str))
    print(output.decode('ISO-8859-1'))
    try:
        slodar_df = pd.read_csv(os.path.join(path,filename),skiprows=1,skipfooter=5,\
                         parse_dates=True, index_col='Date time',\
                         infer_datetime_format=True)

        if len(slodar_df.keys())<2:
            print('No data to be read in the slodar file.')
            raise IOError('Empty data in {0:s}'.format(os.path.join(path,filename)))
        else:
            print('The slodar file contains {0:d} values.'.format(len(slodar_df)))
        slodar_df.rename(columns={'Seeing ["]':'Seeing [arcsec]'}, inplace=True)
        lam = 500.e-9 # wavelength at which r0 is given
        wave_nb=2*np.pi/lam
        slodar_df['r0 above UT [m]'] = np.power(0.423*(wave_nb**2)*slodar_df['Cn2 above UTs [10**(-15)m**(1/3)]']*1.e-15,-3./5.)
        slodar_df['Seeing above UT [arcsec]']= np.rad2deg(lam/slodar_df['r0 above UT [m]'])*3600.
        slodar_df['Surface layer fraction'] = slodar_df['Surface layer profile [10**(-15)m**(1/3)]'] / slodar_df['Cn2 above UTs [10**(-15)m**(1/3)]']
        slodar_df.to_csv(os.path.join(path,filename))            
        return slodar_df
    except ValueError as e:
        print(e)
        print('There is likely no data to be read in the slodar file {0:s}'.format(os.path.join(path,filename)))
        return
    except Exception as e:
        print("Problem during the request")
        print(type(e))
        print(e)        
        return

def query_ecmwf_jetstream(path,start_date='2017-04-28T00:00:00.00',\
               end_date='2017-05-01T12:00:00.00'):
    """
    Query the ESO ECMWF archive.
    It saves the result in a csv file and returns the table as a panda data frame.
    This service was maintained until 2020-12-31 and no data is available in
    2021 or later.
    Input:
        - path: path where to save the csv table
        - start_date: the date (and time) for the query start, in the isot format
                    (for instance '2017-04-28T00:00:00.00')
        - end_date: the date (and time) for the query end, in the isot format
                    (for instance '2017-04-28T00:00:00.00')        
    Output:
        panda table with the following keys: 
            - 'date', in the iso format (e.g. 2018-10-27 06:00:00.000)
            - 'ECMWF jetstream windspeed [m/s]' 
    """
    filename = 'ecmwf_jetstream_query_{0:s}_{1:s}.txt'.format(start_date,end_date)    
    request_asm_str = ['wget','-O',os.path.join(path,filename),\
                   'http://www.eso.org/asm/api/?from={0:s}&to={1:s}Z&fields=asmmetnow-jsspeed'.format(\
                       start_date,end_date)]
    output,error = subprocess.Popen(request_asm_str,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
    print(' '.join(request_asm_str))
    print(output.decode('ISO-8859-1'))    
    dico={'date':[],'ECMWF jetstream windspeed [m/s]':[]}
    try:
        with open (os.path.join(path,filename), "r") as myfile:
            txt_string=myfile.read() #.replace('\n', '')
        txt_string = txt_string[txt_string.index('[[')+2:txt_string.index(']]')].split('],[')
        for txt in txt_string:        
            date_str,speed_str = txt.split(',')
            time_tmp = Time(int(date_str)/1000.,format='unix')
            time_tmp.format='iso'
            dico['date'].append(time_tmp)
            dico['ECMWF jetstream windspeed [m/s]'].append(float(speed_str))
    except ValueError as e:
        print(e)
        print('There is likely no data to be read in the ECMWF file {0:s}'.format(os.path.join(path,filename)))
        return
    except Exception as e:
        print("Problem during the request")
        print(type(e))
        print(e)        
        return
    pd_ecmwf = pd.DataFrame(dico)
    pd_ecmwf.to_csv(os.path.join(path,filename.replace('.txt','.csv')),index=False)
    return pd_ecmwf

def interpolate_date(dates_input,param_input,dates_output,plot=True,kind='linear',fill_value="extrapolate"):
    """
    Given a list of astropy.time.Time and a list of parameters with the same
    length, it interpolates the date at the given output dates
    Input:
        - dates_input an object astropy.time.Time object or list of these objects
        - param_input a list of floats
        - plot: boolean to plot the interpolation or not
        - kind : str or int, optional
            Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'
            where 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of first, second or third order) or as an integer
            specifying the order of the spline interpolator to use.
            Default is 'linear'.
    """    
    interp_function = interp1d(dates_input.mjd,param_input,\
                               kind=kind,bounds_error=False,fill_value=fill_value)
    param_output = interp_function(dates_output.mjd) 
#    dates_output.format='iso'
#    dates_input.format='iso'
    if plot:
        plt.close(1)
        # Create a Figure environment, with a number and size (in inches ... sigh!).
        fig = plt.figure(1, figsize=(7,7))
        ax = plt.subplot() # Create a drawing area ... more on this later !
        # Plot the data
        curve = ax.plot_date(dates_input.plot_date,param_input,'-', color='darkgreen',\
                      label='input')
        pts = ax.plot_date(dates_output.plot_date,param_output,'or',\
                      label='interpolated values')
        # Define the limits, labels, ticks as required
        ax.set_xlabel('Date')
        ax.set_ylabel('Parameter to interpolate')
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.grid(True)
        plt.show()
#        plt.savefig('', bbox_inches = 'tight')
    return param_output
    
    
def query_meteo(path,start_date='2017-04-28T00:00:00.00',\
               end_date='2017-05-01T12:00:00.00'):
    """
    Query the ASM meteo archive,
    saves the result in a csv file and returns the table as a panda data frame.
    Input:
        - path: path where to save the csv table
        - start_date: the date (and time) for the query start, in the isot format
                    (for instance '2017-04-28T00:00:00.00')
        - end_date: the date (and time) for the query end, in the isot format
                    (for instance '2017-04-28T00:00:00.00')   
    Output: a panda data frame with the columns
        Date time
        Wind Direction at 30m (0/360) [deg]
        Wind Direction at 10m (0/360) [deg]
        Wind Speed at 30m [m/s]
        Wind Speed at 10m [m/s]
        Wind Speed U at 20m [m/s]
        Wind Speed V at 20m [m/s]
        Wind Speed W at 20m [m/s]
    """
    filename = 'meteo_query_{0:s}_{1:s}'.format(start_date,end_date)
    request_asm_str = ['wget','-O',os.path.join(path,filename),\
                   'http://archive.eso.org/wdb/wdb/asm/meteo_paranal/query?wdbo=csv&start_date={0:s}..{1:s}&tab_press=0&tab_presqnh=0&tab_temp1=0&tab_temp2=0&tab_temp3=0&tab_temp4=0&tab_tempdew1=0&tab_tempdew2=0&tab_tempdew4=0&tab_dustl1=0&tab_dustl2=0&tab_dusts1=0&tab_dusts2=0&tab_rain=0&tab_rhum1=0&tab_rhum2=0&tab_rhum4=0&tab_wind_dir1=1&tab_wind_dir1_180=0&tab_wind_dir2=1&tab_wind_dir2_180=0&tab_wind_speed1=1&tab_wind_speed2=1&tab_wind_speedu=1&tab_wind_speedv=1&tab_wind_speedw=1&top=5000000'.format(\
                       start_date,end_date)]
    output,error = subprocess.Popen(request_asm_str,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()
    print(' '.join(request_asm_str))
    print(output.decode('ISO-8859-1'))
    df = pd.read_csv(os.path.join(path,filename),skiprows=1,skipfooter=5,engine='python')
    return df

def query_simbad(date,coords,name=None,debug=True,limit_G_mag=15):
    """
    Function that tries to query Simbad to find the object. 
    It first tries to see if the star name (optional argument) is resolved 
    by Simbad. If not it searches for the pointed position (ra and
    dec) in a cone of radius 10 arcsec. If more than a star is detected, it 
    takes the closest from the (ra,dec).
    Input:
        - date: an astropy.time.Time object (e.g. date = Time(header['DATE-OBS'])
        - name: a string with the name of the source.
        - coords: a SkyCoord object. For instance, if we extract the keywords 
            of the fits files, we should use
            coords = coord.SkyCoord(header['RA']*u.degree,header['DEC']*u.degree)
            coord.SkyCoord('03h32m55.84496s -09d27m2.7312s', ICRS)
    Output:
        - a dictionary with the most interesting simbad keywords.
    """
    search_radius = 10*u.arcsec # we search in a 10arcsec circle.
    search_radius_alt = 210*u.arcsec # in case nothing is found, we enlarge the search
        # we use 210 arcsec because Barnard star (higher PM star moves by 10arcsec/yr --> 210 arcsec in 21yrs)
    customSimbad = Simbad()
    customSimbad.add_votable_fields('flux(V)','flux(R)','flux(G)','flux(I)','flux(J)','flux(H)',\
                                    'flux(K)','id(HD)','sp','otype','otype(V)','otype(3)',\
                                   'propermotions','ra(2;A;ICRS;J2000;2000)',\
                                 'dec(2;D;ICRS;J2000;2000)',\
                                 'ra(2;A;FK5;J{0:.3f};2000)'.format(date.jyear),\
                                 'dec(2;D;FK5;J{0:.3f};2000)'.format(date.jyear))
    # First we do a cone search around he coordinates
    search = customSimbad.query_region(coords,radius=search_radius)
    
    if search is None and name is None:
        # If the cone search failed and no name is provided we cannot do anything more
        print('No star identified for the RA/DEC pointing. Enlarging the search to {0:.0f} arcsec'.format(search_radius_alt.value))
        search = customSimbad.query_region(coords,radius=search_radius_alt)
        if search is None:
            print('No star identified for the RA/DEC pointing. Stopping the search.')
            return None
        else:
            validSearch = search[search['FLUX_G']<limit_G_mag]
            nb_stars = len(validSearch)                
        
    elif search is None and name is not None:
        # If the cone search failed but a name is provided, we query that name
        print('No star identified within {0:.0f} arcsec of the RA/DEC pointing. Querying the target name {1:s}'.format(search_radius.to(u.arcsec).value,name))
        # get the star from target name
        simbad_dico = get_dico_star_properties_from_simbad_target_name_search(name,customSimbad)
        if 'simbad_FLUX_V' in simbad_dico.keys():  
            nb_stars = -1 # nothing else to be done! 
            print('Star {0:s} identified using the target name'.format(simbad_dico['simbad_MAIN_ID']))
        else:
            print('No star corresponding to the target name {0:s}. Enlarging the search to {1:.0f} arcsec'.format(name,search_radius_alt.value))
            search = customSimbad.query_region(coords,radius=search_radius_alt)
            if search is None:
                print('No star identified for the RA/DEC pointing. Stopping the search.')
                return None
            else:
                validSearch = search[search['FLUX_G']<limit_G_mag]
                nb_stars = len(validSearch)                
    else:
        # If the cone search returned some results, we count the valid candidates.
        nb_stars = len(search)
        validSearch = search[search['FLUX_G']<limit_G_mag]
        nb_stars = len(validSearch)    
        
    if nb_stars==0:
        print('No star identified for the pointing position. Querying the target name')
        # get the star from target name if we have it in the text file.
        simbad_dico = get_dico_star_properties_from_simbad_target_name_search(name,customSimbad)
        # if we found a star, we add the distance between ICRS coordinates and pointing
        if 'simbad_RA_ICRS' in simbad_dico.keys() and 'simbad_DEC_ICRS' in simbad_dico.keys():
            coords_ICRS_str = ' '.join([simbad_dico['simbad_RA_ICRS'],simbad_dico['simbad_DEC_ICRS']])
            coords_ICRS = coord.SkyCoord(coords_ICRS_str,frame=ICRS,unit=(u.hourangle,u.deg))
            sep_pointing_ICRS = coords.separation(coords_ICRS).to(u.arcsec).value
            simbad_dico['simbad_separation_RADEC_ICRSJ2000']=sep_pointing_ICRS
        # if we found a star, we add the distance between Simbad current coordinates and pointing
        if 'simbad_RA_current' in simbad_dico.keys() and 'simbad_DEC_current' in simbad_dico.keys():
            coords_current_str = ' '.join([simbad_dico['simbad_RA_current'],simbad_dico['simbad_DEC_current']])
            coords_current = coord.SkyCoord(coords_current_str,frame=ICRS,unit=(u.hourangle,u.deg))
            sep_pointing_current = coords.separation(coords_current).to(u.arcsec).value
            simbad_dico['simbad_separation_RADEC_current']=sep_pointing_current
    elif nb_stars>0:
        if nb_stars ==1:
            i_min=0
            print('One star found: {0:s} with G={1:.1f}'.format(\
                  validSearch['MAIN_ID'][i_min],validSearch['FLUX_G'][i_min]))
        else:
            print('{0:d} stars identified within {1:.0f} or {2:.0f} arcsec. Querying the target name'.format(nb_stars,search_radius.value,search_radius_alt.value)) 
            # First we query the target name
            simbad_dico = get_dico_star_properties_from_simbad_target_name_search(name,customSimbad)
            if ('simbad_MAIN_ID' in simbad_dico):
                # the star was resolved and we assume there is a single object corresponding to the search 
                i_min=0
            else:
                print('Target not resolved or not in the list. Selecting the closest star.')
                sep_list = []
                for key in validSearch.keys():
                    if key.startswith('RA_2_A_FK5_'):
                        key_ra_current_epoch = key
                    elif key.startswith('DEC_2_D_FK5_'):
                        key_dec_current_epoch = key
                for i in range(nb_stars):
                    ra_i = validSearch[key_ra_current_epoch][i]
                    dec_i = validSearch[key_dec_current_epoch][i]
                    coord_str = ' '.join([ra_i,dec_i])
                    coords_i = coord.SkyCoord(coord_str,frame=FK5,unit=(u.hourangle,u.deg))
                    sep_list.append(coords.separation(coords_i).to(u.arcsec).value)
                i_min = np.argmin(sep_list)
                min_sep = np.min(sep_list)
                print('The closest star is: {0:s} with G={1:.1f} at {2:.2f} arcsec'.format(\
                  validSearch['MAIN_ID'][i_min],validSearch['FLUX_G'][i_min],min_sep))
        simbad_dico = populate_simbad_dico(validSearch,i_min)
    simbad_dico['DEC'] = coords.dec.to_string(unit=u.degree,sep=' ')
    simbad_dico['RA'] = coords.ra.to_string(unit=u.hourangle,sep=' ')
    # if we found a star, we add the distance between ICRS coordinates and pointing
    if 'simbad_RA_ICRS' in simbad_dico.keys() and 'simbad_DEC_ICRS' in simbad_dico.keys():
        coords_ICRS_str = ' '.join([simbad_dico['simbad_RA_ICRS'],simbad_dico['simbad_DEC_ICRS']])
        coords_ICRS = coord.SkyCoord(coords_ICRS_str,frame=ICRS,unit=(u.hourangle,u.deg))
        sep_pointing_ICRS = coords.separation(coords_ICRS).to(u.arcsec).value
        simbad_dico['simbad_separation_RADEC_ICRSJ2000']=sep_pointing_ICRS
    # if we found a star, we add the distance between Simbad current coordinates and pointing
    if 'simbad_RA_current' in simbad_dico.keys() and 'simbad_DEC_current' in simbad_dico.keys():
        coords_current_str = ' '.join([simbad_dico['simbad_RA_current'],simbad_dico['simbad_DEC_current']])
        coords_current = coord.SkyCoord(coords_current_str,frame=ICRS,unit=(u.hourangle,u.deg))
        sep_pointing_current = coords.separation(coords_current).to(u.arcsec).value
        simbad_dico['simbad_separation_RADEC_current']=sep_pointing_current
        print('Distance between the current star position and pointing position: {0:.1f}arcsec'.format(sep_pointing_current))
    # if we found a star with no R magnitude but with known V mag and spectral type, we compute the R mag.
    if 'simbad_FLUX_V' in simbad_dico.keys() and 'simbad_SP_TYPE' in simbad_dico.keys() and 'simbad_FLUX_R' not in simbad_dico.keys():
        color_VminusR = color(simbad_dico['simbad_SP_TYPE'],filt='V-R')
        if np.isfinite(color_VminusR) and np.isfinite(simbad_dico['simbad_FLUX_V']):
            simbad_dico['simbad_FLUX_R'] = simbad_dico['simbad_FLUX_V'] - color_VminusR
    return simbad_dico

def get_dico_star_properties_from_simbad_target_name_search(name,customSimbad):
    """
    Method not supposed to be used outside the query_simbad method
    Returns a dictionary with the properties of the star, after querying simbad
    using the target name
    If no star is found returns an empty dictionnary
    """
    simbad_dico = {}
    simbadsearch = customSimbad.query_object(name)
    if simbadsearch is None:
        # failure
        return simbad_dico
    else:
        # successful search
        return populate_simbad_dico(simbadsearch,0)

def populate_simbad_dico(simbad_search_list,i):
    """
    Method not supposed to be used outside the query_simbad method
    Given the result of a simbad query (list of simbad objects), and the index of 
    the object to pick, creates a dictionary with the entries needed.
    """    
    simbad_dico = {}
    for key in simbad_search_list.keys():
        if key in ['MAIN_ID','SP_TYPE','ID_HD','OTYPE','OTYPE_V','OTYPE_3']: #strings
            if not simbad_search_list[key].mask[i]:
                simbad_dico['simbad_'+key] = simbad_search_list[key][i]
        elif key in ['FLUX_V', 'FLUX_R', 'FLUX_G','FLUX_I', 'FLUX_J', 'FLUX_H', 'FLUX_K','PMDEC','PMRA']: #floats
            if not simbad_search_list[key].mask[i]:
                simbad_dico['simbad_'+key] = float(simbad_search_list[key][i])
        elif key.startswith('RA_2_A_FK5_'): 
            simbad_dico['simbad_RA_current'] = simbad_search_list[key][i]      
        elif key.startswith('DEC_2_D_FK5_'): 
            simbad_dico['simbad_DEC_current'] = simbad_search_list[key][i]
        elif key=='RA':
            simbad_dico['simbad_RA_ICRS'] = simbad_search_list[key][i]
        elif key=='DEC':
            simbad_dico['simbad_DEC_ICRS'] = simbad_search_list[key][i]     
    return simbad_dico
    
#ra = 10.*u.degree
#dec = -24*u.degree
#testCoord = coord.SkyCoord(ra,dec)
###03 32 55.84496 -09 27 29.7312
#testCoord = coord.SkyCoord('03h32m55.84496s -09d27m29.7312s', ICRS)
##testCoord = coord.SkyCoord('03h32m55.84496s -09d27m12.7312s', ICRS)
#date = Time('2017-01-01T02:00:00.0')
#test=query_simbad(date,testCoord,name='eps Eri')     

def read_color_table():
    """
    Read the csv file Johnson_color_stars.txt, built from the website
    http://www.stsci.edu/~inr/intrins.html and that gives the Johnson colors 
    of stars depending on their spectral type.
    """
    tablename=os.path.join(path_data,'Johnson_color_stars.txt')
    tab = pd.read_csv(tablename)
    return tab

def extract_spectral_type_code(sp_type_str):
    """
    Function that uses the spectral type given by Simbad (for instance
    G2IV, F5V, G5V+DA1.0, M1.4, M1.5, or F0IV) and that returns 
    the code from 0=B0.0 to 49=M4.0 (same convention as in 
    http://www.stsci.edu/~inr/intrins.html)    
    """
    if not isinstance(sp_type_str,str):
        if isinstance(sp_type_str,unicode):
            sp_type_str=str(sp_type_str)
        else:
            print('Argument is not a string. Returning.')
            print(sp_type_str)
            return
    spectral_type_letter = (sp_type_str[0:1]).upper()
    if spectral_type_letter=='B':
        offset_code = 0.
    elif spectral_type_letter=='A':
        offset_code = 10.
    elif spectral_type_letter=='F':
        offset_code = 20.
    elif spectral_type_letter=='G':
        offset_code = 30.
    elif spectral_type_letter=='K':
        offset_code = 40.
    elif spectral_type_letter=='M':
        offset_code = 48.
    else:
        print('The spectral letter extracted from {0:s} is not within B, A, F, G, K, M. Returning'.format(spectral_type_letter))
        return
    if sp_type_str[2:3] == '.':
        spectral_type_number = float(sp_type_str[1:4])
    else:
        try:
            spectral_type_number = float(sp_type_str[1:2])
        except:
            spectral_type_number = 0
            print('The spectral type {0:s} could not not be accurately determined.'.format(sp_type_str))
    return offset_code+spectral_type_number
    

def color(sp_type_str,filt='V-R'):
    """
    Reads the table from http://www.stsci.edu/~inr/intrins.html and returns the 
    color of the star corresponding to the spectral type given in input.
    Input:
        - sp_type_str: a string representing a spectral type, such as that returned 
            by a simbad query. Some examples of such strings are
            G2IV, F5V, G5V+DA1.0, M1.4, M1.5, or F0IV
        - filt: a string representing a color, to be chose between
            U-B, B-V, V-R, V-I, V-J, V-H, V-K, V-L, V-M, V-N. By default it is 
            V-R. 
    """
    table = read_color_table()
    if filt not in table.keys():
        print('The color requested "{0:s}" is not in the color table.'.format(filt))
        return
    code = extract_spectral_type_code(sp_type_str)
    if not isinstance(code,(int, float)):
        return np.nan
    interp_function = interp1d(table['Code'],table[filt],bounds_error=True)
    try:
        col = interp_function(code)
    except ValueError as e:
        print('ValueError: {0:s}'.format(str(e)))
        print('The spectral type code {0:.1f} is out of range'.format(code))
        print('Returning NaN')   
        return np.nan
    return float(col)

def query_atmospheric_transparency(start_date='2017-04-28T00:00:00.00',\
               end_date='2017-05-01T12:00:00.00'):
    """
    For a given start and end date, returns the sub-table with the atmospheric
    transparency entered by the weather officer in Paranal. The data are available 
    from 2011-11-15 to 2023-02-09.    

    Parameters
    ----------
    start_date : str, optional
        start date for the query. The default is '2017-04-28T00:00:00.00'.
    end_date : str, optional
        end date for the query. The default is '2017-05-01T12:00:00.00'.

    Raises
    ------
    ValueError
        If start date or end date is out of the possible range, raise a ValueError.

    Returns
    -------
    panda DataFrame
        Panda Dataframe with the following columns: ['author', 'weather_category', 'comment']
        and a datetime index. The first entry of the table is earlier than start_date
        and the last entry is later than end_date.
    """
    tablename = os.path.join(path_data,'Paranal_weather_observations_2011_to_2023.csv')
    atm_transparency_table = pd.read_csv(tablename,parse_dates=['observation_timestamp'],\
                             index_col='observation_timestamp',infer_datetime_format=True)
    dt_start_time = Time(start_date).to_datetime()
    dt_end_time = Time(end_date).to_datetime()
    if dt_start_time<atm_transparency_table.index[0]:
        raise ValueError('The start date {0:s} is smaller than the earliest possible date {1:s}'.format(str(dt_start_time),str(atm_transparency_table.index[0])))
    if dt_end_time>atm_transparency_table.index[-1]:
        raise ValueError('The end date {0:s} is larger than the latest possible date {1:s}'.format(str(dt_end_time),str(atm_transparency_table.index[-1])))
    atm_transparency_table.index[-1]
    index_after_start = np.argmax((atm_transparency_table.index - dt_start_time).total_seconds()>0)
    index_after_end = np.argmax((atm_transparency_table.index - Time(end_date).to_datetime()).total_seconds()>0)
    return atm_transparency_table.iloc[index_after_start-1:index_after_end+1]

if __name__ == "__main__":

    # test the color function
    sp_type_str = 'M1.4'
    c=color(sp_type_str)
    print(c)

    # test of the query_ecmwf_jetstream function
    path_ecmwf = '/Users/millij/Documents/atmospheric_parameters/ECMW_forecast/test' 
    start_date = '2020-12-27T00:00:00'
    end_date = '2020-12-28T00:00:00'
    pd_ecmwf = query_ecmwf_jetstream(path_ecmwf,start_date,end_date)

    start_date = '2020-12-27T00:00:00'
    end_date = '2020-12-28T00:00:00'
    pd_transparency = query_atmospheric_transparency(start_date,end_date)
    
    # test of the query_slodar function
    path_slodar = '/Users/millij/Documents/atmospheric_parameters/SLODAR/test' 
    start_date = '2018-10-27T00:00:00'
    end_date = '2018-11-11T00:00:00'
    pd_slodar = query_slodar(path_slodar,start_date,end_date)

    # test of the query_simbad functions
    date_test = Time('2020-07-14T00:04:20.200')
    # test_coordinates = coord.SkyCoord.from_name('Lacaille 8760')
    test_coordinates = coord.SkyCoord.from_name('HIP 87937')
    # test_coordinates = coord.SkyCoord('17h57m48.4997s +04d44m36.111s', frame=ICRS)
    simbad_dico = query_simbad(date_test,test_coordinates,name=None,debug=True)
    print(simbad_dico)

    # other test    
    target_name = 'SCrA'
    coords = coord.SkyCoord(285.286528*u.degree,-36.95604*u.degree)
    date = Time('2018-07-05T08:29:35.41')    
    simbad_dico = query_simbad(date,coords,name=target_name,limit_G_mag=15)
    # query_simbad(date,coords,name=None,debug=True,limit_G_mag=15):

    # radius = u.Quantity(12.0, u.arcsec)
    # j = Gaia.cone_search_async(test_coordinates, radius)
    # r = j.get_results()
    # r.pprint()        
        
    