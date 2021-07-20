#!/opt/anaconda3/envs/python37/bin/python

import argparse
# from glob import glob
# import os
import numpy as np
# import subprocess
import time
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
# import sys
# from astropy.io import ascii,fits
# import os,sys
import sparta_utilities as sparta
import glob
from astropy.time import Time
# import matplotlib.pyplot as plt
# import numpy as np
from scipy import signal
# from pathlib import Path
from query_eso_archive import query_dimm,query_mass
"""

"""

wl=600e-9 # AO wavelength
D=8 #pupil diameter
peak_to_valley_arcsec = 2.6
peak_to_valley_m = np.sin(np.deg2rad(peak_to_valley_arcsec/3600))*D #the ptv of the TTM is 2.6 arcsec
# print('The stroke of the ITTM mirror is {0:.1f} microns'.format(peak_to_valley_m*1e6))
    
# visloop_recorder_file = Path('/Volumes/MILOU_1TB/sparta_data/2019-07-27_low_tau0/2019-07-28T00:18:42-VisLoopRecorder/2019-07-28T00:18:42-VisLoopRecorder.fits')
# 
def analyze_tiptilt(visloop_recorder_file):
    start_time = time.time()
    path_parent = visloop_recorder_file.parent
    VisLoopRecorder_HDU = fits.open(visloop_recorder_file)      
    VisLoopRecorder_HDU.info()
    
    VisLoopRecorder = VisLoopRecorder_HDU[1].data
    
    seconds_VisLoopRecorder = VisLoopRecorder["Seconds"]+VisLoopRecorder["USeconds"]/1.e6
    time_ellapsed_array = seconds_VisLoopRecorder - seconds_VisLoopRecorder[0] #np.diff(seconds_VisLoopRecorder)
    # time_VisLoopRecorder = Time(seconds_VisLoopRecorder,format='unix')
        
    size_VisLoopRecorder = len(time_ellapsed_array)    
    print('There are {0:6d} time stamps'.format(size_VisLoopRecorder))
    
    ellapsed_time_VisLoopRecorder = seconds_VisLoopRecorder[-1]-seconds_VisLoopRecorder[0]
    print('Total ellapsed time: {0:5.1f}s'.format(ellapsed_time_VisLoopRecorder))
    delta_t = ellapsed_time_VisLoopRecorder/size_VisLoopRecorder
    print('Time between 2 time stamps: {0:4.2f}ms'.format(delta_t*1000))
    freq = 1./delta_t
    print('Sampling frequency of {0:6.1f} Hz'.format(freq))

    start_date = Time(seconds_VisLoopRecorder[0],format='unix')
    end_date = Time(seconds_VisLoopRecorder[-1],format='unix')    
    pd_dimm = query_dimm(str(path_parent),start_date=start_date.iso,end_date=end_date.iso)
    pd_mass = query_mass(str(path_parent),start_date=start_date.iso,end_date=end_date.iso)
    mean_dimm_seeing = np.mean(pd_dimm['DIMM Seeing ["]'])
    mean_mdimm_seeing = np.mean(pd_mass['MASS-DIMM Seeing ["]'])
    mean_tau0_ms = np.mean(pd_mass['MASS-DIMM Tau0 [s]'])*1000
    expected_jitter_arcsec = seeing_to_jitter(mean_mdimm_seeing)
    turb_velocity = np.mean(0.314*(500e-9/np.deg2rad(pd_mass['MASS-DIMM Seeing ["]']/3600))/(pd_mass['MASS-DIMM Tau0 [s]']))
    cross_over_frequency = 0.7*turb_velocity/D
    # pd_mass['MASS-DIMM Turb Velocity [m/s]']


    ITTM_Positions = VisLoopRecorder['ITTM_Positions']

    tip_rms = np.std(ITTM_Positions[:,0])
    tilt_rms = np.std(ITTM_Positions[:,1])
    # jitter_to_seeing(tip_rms_arcsec)
    # jitter_to_seeing(tilt_rms_arcsec)
    mean_jitter = np.mean([tip_rms,tilt_rms])
    scaling_factor_tt = expected_jitter_arcsec/mean_jitter
    print('To match a seeing of {0:.2f}arcsec, the conversion factor from TT unit to arcsec is {1:.2f} arcsec'.format(mean_mdimm_seeing,scaling_factor_tt))
    print('According to Jeff, this value should be close to 2.6 arcsec')
    
    
    
    plt.close(0)
    fig = plt.figure(0, figsize=(7,3))
    ax=plt.subplot(111)
    ax.plot(time_ellapsed_array,ITTM_Positions[:,0], label='tip')
    ax.plot(time_ellapsed_array,ITTM_Positions[:,1], label='tilt')
    ax.set_xlabel('Time ellapsed in s')
    ax.set_ylabel('Position')
    ax.legend(frameon=False,loc='best')
    ax.grid()
    ax.set_title('Seeing {0:.2f}", $\\tau_0$ {1:.1f}ms'.format(mean_mdimm_seeing,mean_tau0_ms))
    plt.tight_layout()
    fig.savefig(path_parent.joinpath('{0:s}_ITTM_position_vs_time.pdf'.format(start_date.iso)))

    plt.close(1)
    fig = plt.figure(1, figsize=(7,3))
    ax=plt.subplot(111)
    ax.plot(time_ellapsed_array,ITTM_Positions[:,0]*scaling_factor_tt, label='tip')
    ax.plot(time_ellapsed_array,ITTM_Positions[:,1]*scaling_factor_tt, label='tilt')
    ax.set_xlabel('Time ellapsed in s')
    ax.set_ylabel('Position in arcsec')
    ax.legend(frameon=False,loc='best')
    ax.grid()
    ax.set_title('Seeing {0:.2f}", $\\tau_0$ {1:.1f}ms (TT pos to ": {2:.2f}")'.format(mean_mdimm_seeing,mean_tau0_ms,scaling_factor_tt))
    plt.tight_layout()
    fig.savefig(path_parent.joinpath('{0:s}_ITTM_position_arcsec_vs_time.pdf'.format(start_date.iso)))
    
    
    f_tip, PSD_tip = signal.periodogram(ITTM_Positions[:,0], freq)
    f_tilt, PSD_tilt = signal.periodogram(ITTM_Positions[:,1], freq)
    plt.close(2)
    fig = plt.figure(2, figsize=(7,3))
    ax = plt.subplot(111)
    ax.loglog(f_tip, PSD_tip,label='tip',alpha=0.6)
    ax.loglog(f_tilt, PSD_tilt,label='tilt',alpha=0.6)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('PSD [$Hz^{-1}$]')
    ax.legend(frameon=False,loc='best')
    ax.grid()
    plt.tight_layout()
    fig.savefig(path_parent.joinpath('{0:s}_PSD_tiptilt.pdf'.format(start_date.iso)))
    
    min_freq_for_fit = 10. #minimum frequency for which we want to start the fit
    max_freq_for_fit = 1000 #maximum frequency for which we want to start the fit
    freq_array = np.array([min_freq_for_fit,max_freq_for_fit])
    log_freq_tip=np.log(f_tip)
    log_PSD_tip=np.log(PSD_tip)
    id_tip_for_fit = np.logical_and(f_tip>min_freq_for_fit,f_tip<max_freq_for_fit)
    coefficients_tip = np.polyfit(log_freq_tip[id_tip_for_fit], log_PSD_tip[id_tip_for_fit], 1)
    linear_fit_function_tip = np.poly1d(coefficients_tip)
    log_freq_tilt=np.log(f_tilt)
    log_PSD_tilt=np.log(PSD_tilt)
    id_tilt_for_fit = np.logical_and(f_tilt>min_freq_for_fit,f_tilt<max_freq_for_fit)
    coefficients_tilt = np.polyfit(log_freq_tilt[id_tilt_for_fit], log_PSD_tilt[id_tilt_for_fit], 1)
    linear_fit_function_tilt = np.poly1d(coefficients_tilt)
    
    
    plt.close(3)
    # fig = plt.figure(3, figsize=(7,7))
    fig, ax = plt.subplots(2,1,num=3,figsize=(7,7))
    ax[0].scatter(f_tip, PSD_tip,s=1,color='skyblue',alpha=0.3,label='tip')
    ax[0].plot(freq_array, np.exp(linear_fit_function_tip(np.log(freq_array))),\
            color='black',label='fit: {0:.1e}*f^({1:.2f})'.format(np.exp(coefficients_tip[1]),\
                                    coefficients_tip[0]), linewidth=2)            
    ax[1].scatter(f_tilt, PSD_tilt,s=1,color='darkorange',alpha=0.3,label='tilt')
    ax[1].plot(freq_array, np.exp(linear_fit_function_tilt(np.log(freq_array))),\
            color='black',label='fit: {0:.1e}*f^({1:.2f})'.format(np.exp(coefficients_tilt[1]),\
                                    coefficients_tilt[0]), linewidth=2)            
    ax[0].set_ylabel('Tip PSD [$Hz^{-1}$]')
    ax[1].set_ylabel('Tilt PSD [$Hz^{-1}$]')
    for a in ax:
        a.axvline(cross_over_frequency,linestyle=':',color='grey',label='Greenwood cross-over freq')
        a.set_yscale('log')
        a.set_xscale('log')
        a.set_xlabel('Frequency [Hz]')
        a.grid()
        a.set_xlim([f_tip[1],f_tip[-1]])
        a.set_ylim(np.percentile(PSD_tip,[0.01,99.99]))
        a.legend(frameon=False)    
    ax[0].set_title('Kolmogorov says -8/3=-2.76 slope')
    plt.tight_layout()
    fig.savefig(path_parent.joinpath('{0:s}_PSD_tiptilt_with_fit.pdf'.format(start_date.iso)))
        
    VisLoopRecorder_HDU.close()                
    end_time = time.time()
    print('Tip/tilt data analyzed in {0:.1f}s'.format(end_time-start_time))
    return

def jitter_to_seeing(jitter_arcsec):
    """
    Converts a jitter  in arcsec into a seeing in arcsec using the formula in
    the field guide to AO by Tyson
    """
    jitter_rad = np.deg2rad(jitter_arcsec/3600)
    r0 = np.power(jitter_rad**2/0.182/wl**2*np.power(D,1/3.),-3/5)
    seeing_arcsec = np.rad2deg(500e-9/r0)*3600
    return seeing_arcsec

def seeing_to_jitter(seeing_arcsec):
    """
    Converts a seeing  in arcsec into a jitter in arcsec using the formula in
    the field guide to AO by Tyson
    """
    r0=500e-9/np.deg2rad(seeing_arcsec/3600)
    jitter_rad = np.sqrt(0.182*wl**2*np.power(D,-1/3.)*np.power(r0,-5/3.))
    jitter_arcsec = np.rad2deg(jitter_rad)*3600
    return jitter_arcsec
    
if __name__ == "__main__":
    # We parse the input
    parser = argparse.ArgumentParser(description='Type the *_VisLoopRecorder.fits file(s) you want to process')

    parser.add_argument('files', type=str, help='VisLoopRecorder.fits you want to process', nargs='*')
    parser.add_argument('-tt', help='analyze tip-tilt', action='store_true') #if used True (we download) else False: no download
    args = parser.parse_args()
    visloop_recorder_files = args.files
    analyze_tt = args.tt   
    if analyze_tt:
        for visloop_recorder_file in visloop_recorder_files:
            analyze_tiptilt(Path(visloop_recorder_file))