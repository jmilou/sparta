#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:11:33 2017

@author: jmilli

This script reads data from the ECMWF saved as a csv file. It includes utilities 
to find properties of the wind in altitude.
"""

import os,sys
import numpy as np
#import matplotlib.pyplot as plt
#import glob
from astropy.time import Time
from datetime import datetime
from scipy.interpolate import interp1d
from astropy import units as u
import pandas as pd

path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'sparta_data')
filename = 'ecmwf_profile_200mbar_2015-nov2018.csv'
pd_ecmwf= pd.read_csv(os.path.join(path,filename),na_values=-9999.0)#,sep='\t')
#pd_ecmwf= pd.read_csv(os.path.join(path,filename),na_values=-9999.0,sep='\t')
#pd_ecmwf.head()
#"year	month	day	hour_UT	geopotential height (m)	temperature (degree)	humidity (%)	wind velocity (m/s)	wind direction (degree from north clockwise)"
datetime_list = []
for i in range(len(pd_ecmwf)):
    datetime_list.append(datetime(pd_ecmwf['year'][i],pd_ecmwf['month'][i],pd_ecmwf['day'][i],pd_ecmwf['hour_UT'][i],0,0))
time_list = Time(datetime_list)
time_list.format='isot'

pd_ecmwf['date']=time_list
pd_ecmwf.drop('year', axis=1, inplace=True)
pd_ecmwf.drop('month', axis=1, inplace=True)
pd_ecmwf.drop('day', axis=1, inplace=True)
pd_ecmwf.drop('hour_UT', axis=1, inplace=True)
pd_ecmwf.rename(columns={'geopotential height (m)': 'ecmwf_200mbar_geopotential_height[m]',\
                         'temperature (degree)':'ecmwf_200mbar_temp[deg]',\
                         'humidity (%)':'ecmwf_200mbar_rel_hum[%]',\
                         'wind velocity (m/s)':'ecmwf_200mbar_windspeed[m/s]',\
                         'wind direction (degree from north clockwise)':'ecmwf_200mbar_winddir[deg]'}, inplace=True)

interp_function_windspeed = interp1d(time_list.mjd,pd_ecmwf['ecmwf_200mbar_windspeed[m/s]'],\
                               kind='linear',bounds_error=False,fill_value="extrapolate")

unwrap_winddir = np.rad2deg(np.unwrap(np.deg2rad(pd_ecmwf['ecmwf_200mbar_winddir[deg]'])))
interp_function_winddir = interp1d(time_list.mjd,unwrap_winddir,\
                               kind='linear',bounds_error=False,fill_value="extrapolate")

def request_ecmwf(date_start,date_end):
    """
    Reads a csv file that contains predictions of the ECMWF andchecks whether 
    predictions are available for the given starting and ending dates. Returns a 
    data frame with the predictions within those dates.
    Input:
        -date_start: a astropy.time.Time object for the starting time
        -date_end: a astropy.time.Time object for the ending time
    Output:
        - a panda dataframe with keys in ['ecmwf_200mbar_geopotential_height[m]',
        'ecmwf_200mbar_temp[deg]','ecmwf_200mbar_windspeed[m/s]',
        'ecmwf_200mbar_winddir[deg]','date']  
    """
    min_start = np.min(np.abs(time_list-date_start))
    min_end = np.min(np.abs(time_list-date_end))
    id_min_start = np.argmin(np.abs(time_list-date_start))
    id_min_end   = np.argmin(np.abs(time_list-date_end))
    if id_min_start<1:        
        print('No ECMFW forecast available for the time {0:s}. The closest ECMWF data has {1:.1f} days difference.'.format(date_start.isot,min_start.to(u.day).value))
        return
    elif id_min_end>len(pd_ecmwf)-2:
        print('No ECMFW forecast available for the time {0:s}. The closest ECMWF data has {1:.1f} days difference.'.format(date_end.isot,min_end.to(u.day).value))
        return        
    elif min_start> 30*u.hour: 
        print('No ECMFW forecast available for the starting time {0:s}. The closest ECMWF data has {1:.1f} days difference.'.format(date_start.isot,min_start.to(u.day).value))
        return
    elif  min_end> 30*u.hour:     
        print('No ECMFW forecast available for the ending time {0:s}. The closest ECMWF data has {1:.1f} days difference.'.format(date_end.isot,min_end.to(u.day).value))
        return 
    else:
        id_min_start = id_min_start-1
        id_min_end = id_min_end+1
        return pd_ecmwf[id_min_start:id_min_end]
    
def interpolate_ecmwf(date,output='windspeed',verbose=True):
    """
    Given an object astropy.time.Time, it checks whether
    there is a forecast from the ECMWF and interpolates the wind
    Input:
        - date an object astropy.time.Time object or list of these objects
        - output: default is 'windspeed'. Can be also 'winddir'
    """    
    if output=='windspeed':
        interp_function = interp_function_windspeed
    elif output=='winddir':
        interp_function = interp_function_winddir
    else:
        print("The type of output of interpolate_ecmwf was not understood: {0:s}. It should be 'windspeed' or 'winddir'".format(output)) 
        return
    if isinstance(date,Time)==False:
        print('The input date must be an astropy.time.Time object')
        return
    if date.size>1:
        param_output = np.ndarray((date.size))
        param_output.fill(np.nan)
        for i,d in enumerate(date):
            min_time_ellapsed = np.min(np.abs(time_list-d))
            if min_time_ellapsed > 30*u.hour :
                if verbose:
                    print('No ECMFW forecast available for {0:s}. The closest ECMWF data has {1:.1f} days difference.'.format(d.isot,min_time_ellapsed.to(u.day).value))
            else:
                param_output[i] = interp_function(d.mjd)
        return param_output
    else:            
        min_time_ellapsed = np.min(np.abs(time_list-date))
        if min_time_ellapsed > 30*u.hour :
            if verbose:
                print('No ECMFW forecast available for {0:s}. The closest ECMWF data has {1:.1f} days difference.'.format(date.isot,min_time_ellapsed.to(u.day).value))
            return np.nan
        else:
            return interp_function(date.mjd)

if __name__ == "__main__":
#    ws = interpolate_ecmwf(Time(list(['2015-07-20 23:40:50.300','2015-03-20 23:40:50.300'])))
#    ws1 = interpolate_ecmwf(Time('2015-03-20 23:40:50.300'))
#    ws2 = interpolate_ecmwf(Time('2015-07-20 23:40:50.300'))
#    print(ws)
#    print(ws1)
#    print(ws2)
#    ws3 = interpolate_ecmwf(Time('2017-07-20 23:40:50.300'))
#    wd1 = interpolate_ecmwf(Time(list(['2014-07-20 23:40:50.300','2015-03-20 23:40:50.300'])),output='winddir')
#    print(wd1)
#    pd_test = request_ecmwf(Time('2017-02-17 23:00:00.00'),Time('2017-02-18 12:00:00.000'))
#    print(pd_test)  
    
    pd_test = request_ecmwf(Time('2017-11-18 23:00:00.00'),Time('2017-11-19 12:00:00.000'))
    print(pd_test)  
    print(np.mean(pd_test['ecmwf_200mbar_windspeed[m/s]']))
        