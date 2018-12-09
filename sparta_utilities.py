#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:16:07 2017

@author: jmilli
"""
from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.colors import LogNorm,SymLogNorm
from astropy.time import Time
import pandas as pd
path_sparta = os.path.join(os.path.dirname(os.path.abspath(__file__)),'sparta_data')

def DMvect2map(v,visu=True):
    """
    Takes in input a 1D vector of 1377 elements and map it to the DM actuator map
    """
    if len(v) != 1377:
        raise IOError('The input vector must have 1377 elements (currently {0:d})'.format(len(v)))
    DM_mask = fits.getdata(os.path.join(path_sparta,'DMMap.fits'))
    DM_map = np.ones_like(DM_mask,dtype=float)
    DM_map[DM_mask==0] = np.nan
    DM_map[DM_mask>0] = v
    DM_map = np.fliplr(np.rot90(DM_map,-3))
    if visu:
        fig, ax = plt.subplots()
        im = ax.imshow(DM_map, cmap='CMRmap', origin='lower',interpolation='nearest',\
                       vmin=np.nanmin(v),vmax=np.nanmax(v))
        fig.colorbar(im)
        ax.set_title('Map of the DM')
    return DM_map

def save_unix_times(array_seconds,filename=None):
    """
    Functions that takes in input an array with the times in the unix format 
    (in seconds) and converts it to the iso format and save the result as 
    a csv file and returns a panda array.
    """
    TimesList = Time(array_seconds,format='unix')
    TimesList.format = 'iso'
    date_iso_str = [str(s) for s in TimesList]
    pd_time = pd.DataFrame({'Date_ISO':date_iso_str,'Date_unix':array_seconds})
    if filename is not None:
        pd_time.to_csv(filename)
    return pd_time    

def DMpositions2cube(v,path='.',subtractBias=True,name='HODM_cube.fits',binning=20):
    """
    Similar to DMvect2map but it acts on a 2d array of size N by 1377 and makes 
    a cube. Optionnaly you subtract the median cube. The cube is saved in a fits file.
    """
    nb_time_stamps = v.shape[0]
    if v.shape[1] != 1377:
        raise IOError('The input array must be 2d with 1377 elements in the 2nd dimension (currently {0:d})'.format(v.shape[1]))
    v = v-np.median(v,axis=0)
    DM_mask = fits.getdata(os.path.join(path_sparta,'DMMap.fits'))
    DM_cube = np.ndarray((int(nb_time_stamps/binning),DM_mask.shape[0],DM_mask.shape[1]),dtype=float)
    for i in range(int(nb_time_stamps/binning)):
        DM_map_tmp = np.ones_like(DM_mask,dtype=float)
        DM_map_tmp[DM_mask==0] = np.nan
        DM_map_tmp[DM_mask>0] = np.mean(v[i*binning:(i+1)*binning,:],axis=0)
        DM_map_tmp = np.fliplr(np.rot90(DM_map_tmp,-3))
        DM_cube[i,:,:] = DM_map_tmp
    fits.writeto(os.path.join(path,name),DM_cube,overwrite=True)
    return DM_cube

def SHvect2map(v,visu=True):
    """
    Takes in input a 1D vector of 1240 elements and map it to the SH WFS map
    """
    if len(v) != 1240:
        raise IOError('The input vector must have 1240 elements (currently {0:d})'.format(len(v)))
    SH_mask = fits.getdata(os.path.join(path_sparta,'shack.grid.fits'))
    SH_map = np.ones_like(SH_mask,dtype=float)
    SH_map[SH_mask==0] = np.nan
    SH_map[SH_mask>0] = v
    SH_map = np.fliplr(np.rot90(SH_map,-3))
    if visu:
        fig, ax = plt.subplots()
        im = ax.imshow(SH_map, cmap='CMRmap', origin='lower',interpolation='nearest',\
                       vmin=np.nanmin(v),vmax=np.nanmax(v))
        fig.colorbar(im)
        ax.set_title('Map of the Shack-Hartmann')
    return SH_map    
    
def SHslopes2map(slopes,visu=True):
    """
    Takes in input a 1D vector of 2480 elements, map it to the SH WFS and returns
    2 maps of the slopes in x and y
    """
    if len(slopes) != 2480:
        raise IOError('The input vector must have 2480 elements (currently {0:d})'.format(len(slopes)))
    mapx = np.ndarray((40,40),dtype=float)*np.nan
    mapy = np.ndarray((40,40),dtype=float)*np.nan
    shackgrid = fits.getdata(os.path.join(path_sparta,'shack.grid.fits'))
    mapx[shackgrid>0] = slopes[np.arange(1240,dtype=int)*2]
    mapy[shackgrid>0] = slopes[np.arange(1240,dtype=int)*2+1]
    mapx = np.fliplr(np.rot90(mapx,-3))
    mapy = np.fliplr(np.rot90(mapy,-3))
    if visu: 
        fig, ax = plt.subplots(1,2)
        im = ax[0].imshow(mapx, cmap='CMRmap', origin='lower',interpolation='nearest',\
            vmin=np.nanmin(slopes),vmax=np.nanmax(slopes))
        ax[0].set_title('SH slopes X')
        ax[1].imshow(mapy, cmap='CMRmap', origin='lower',interpolation='nearest',\
            vmin=np.nanmin(slopes),vmax=np.nanmax(slopes))
        ax[1].set_title('SH slopes Y')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    return mapx,mapy

def DMposition2phase(HODM_Position,HODM_Position_bias=None,visu=True):
    """
    From the voltage vector applied to the DM, this function reconstructs the 
    corresponding wavefront in nm by using the influence matrix of the instrument. 
    Input:
        - HODM_Position: a vector or 1377 elements representing the voltage 
            applied to the DM
        - HODM_Position_bias: the bias of the voltage (typically the median of
            the sequence). This is optionnal and if nothing is provided, uses 0
        - visu: boolean to display the image     
    """
    if len(HODM_Position) != 1377:
        raise IOError('The input vector must have 1377 elements (currently {0:d})'.format(len(HODM_Position)))    
    IMF = fits.getdata(os.path.join(path_sparta,'SAXO_DM_IFM.fits')) #shape (1377, 240, 240)
    # The matrix needs to be normalised to allow conversion into optical wavefront errors:
    # influence matrix normalization = defoc meca in rad @ 632 nm
    rad_632_to_nm_opt = 1. / 2. / np.pi * 632 * 2
    IMF   = IMF * rad_632_to_nm_opt
    IMF   = IMF.reshape(1377, 240*240).T # shape (57600, 1377)
    if HODM_Position_bias is None:
        HODM_Position_bias = np.zeros((1377))
    HODM_Position_NoBias = HODM_Position - HODM_Position_bias
    phase = (IMF @ HODM_Position_NoBias).reshape((240, 240))
    if visu:        
        fig, ax = plt.subplots()
        im = ax.imshow(phase, cmap='CMRmap', origin='lower',interpolation='nearest',\
                       vmin=np.nanmin(phase),vmax=np.nanmax(phase))
        fig.colorbar(im,orientation='vertical', label='OPD [nm]')
        ax.set_title('Corrected phase')        
    return phase    

def DMpositions2phase(HODM_Positions,HODM_Position_bias=None,name='corrected_phase_cube.fits',\
                      path='.',binning=40):
    """
    This function is similar to the DMPosition2phase except that it takes in 
    input  a sequence of voltages instead of a single vector, and therefore 
    produces a cube of wavefronts. 
    The option to rebin the sequence was introduced in order to keep the size
    of the output cube reasonable. 
    Input:
        - HODM_Positions: an array or shape (nframes,1377) representing the sequence 
            of nframes voltages applied to the DM
        - HODM_Position_bias: the bias of the voltage. This is optionnal 
            and if nothing is provided, uses 0        
        - name: the name of the file to save
        - path: path where the file is saved
        - binning: the binning factor to apply (40 by default). If no binning 
            is desired, use binning=1
    """
    if HODM_Positions.ndim != 2 and HODM_Positions.shape[1] != 1377:
        raise IOError('The input vector must be a 2D array of shape (nframes,1377) (currently',HODM_Positions.shape,')')    
    nframes = HODM_Positions.shape[0]
    IMF = fits.getdata(os.path.join(path_sparta,'SAXO_DM_IFM.fits')) #shape (1377, 240, 240)
    # The matrix needs to be normalised to allow conversion into optical wavefront errors:
    # influence matrix normalization = defoc meca in rad @ 632 nm
    rad_632_to_nm_opt = 1. / 2. / np.pi * 632 * 2
    IMF   = IMF * rad_632_to_nm_opt
    IMF   = IMF.reshape(1377, 240*240).T # shape (57600, 1377)
    if HODM_Position_bias is None:
        HODM_Position_bias = np.zeros((1377))
    if binning ==1:
        HODM_Positions_NoBias = HODM_Positions - HODM_Position_bias
        phase = (HODM_Positions_NoBias @ IMF.T).reshape((nframes, 240, 240))        
    elif binning>1:
        HODM_Positions_binned = np.ndarray((int(nframes/binning),1377),dtype=float)
        for i in range(int(nframes/binning)):
            HODM_Positions_binned[i,:] = np.mean(HODM_Positions[i*binning:(i+1)*binning,:],axis=0)
        HODM_Positions_NoBias = HODM_Positions_binned - HODM_Position_bias
        phase = (HODM_Positions_NoBias @ IMF.T).reshape((int(nframes/binning), 240, 240))   
    else:
        raise IOError('The binning factor must be an integer greater or equal to 1.')           
    fits.writeto(os.path.join(path,name),phase,overwrite=True)
    return phase    

def ResidualSlopes2ResidualPhase(Gradients,S2M,M2V,visu=True):
    """
    From the voltage vector applied to the DM, this function reconstructs the 
    corresponding wavefront in nm by using the influence matrix of the instrument. 
    Input:
        - Gradients: a vector or 2480 elements representing the slopes measured
            by the WFS
        - S2M: the slopes to modes matrix (shape (988, 2480))
        - M2V: the modes to voltage matrix (shape (1377, 988))
        - visu: boolean to display the image     
    """
    if len(Gradients) != 2480:
        raise IOError('The input Gradients vector must have 2480 elements (currently {0:d})'.format(len(Gradients)))    
    if S2M.ndim!=2 or S2M.shape[1]!=2480:
        raise IOError('The input S2M matrix must have a shape (988, 2480) (currently',S2M.shape,')')
    if M2V.ndim!=2 or M2V.shape[0]!=1377:
        raise IOError('The input M2V matrix must have a shape (1377, 988) (currently',M2V.shape,')')
    IMF = fits.getdata(os.path.join(path_sparta,'SAXO_DM_IFM.fits')) #shape (1377, 240, 240)
    # The matrix needs to be normalised to allow conversion into optical wavefront errors:
    # influence matrix normalization = defoc meca in rad @ 632 nm
    rad_632_to_nm_opt = 1. / 2. / np.pi * 632 * 2
    IMF   = IMF * rad_632_to_nm_opt
    IMF   = IMF.reshape(1377, 240*240).T # shape (57600, 1377)
    mode  = S2M @ Gradients
    volt  = M2V @ mode
    res_turbulence = (IMF @ volt).reshape((240, 240))
    # compute theoretical Strehl
    wave   = 1600
    err    = res_turbulence[res_turbulence != 0].std()
    sigma  = 2*np.pi*err/wave
    strehl = np.exp(-sigma**2)
    print('Residual OPD:     {0:6.1f} nm'.format(err))
    print('Estimated Strehl: {0:6.1f}%'.format(strehl*100))
    if visu:        
        fig, ax = plt.subplots()
        im = ax.imshow(res_turbulence, cmap='CMRmap', origin='lower',interpolation='nearest',\
                       vmin=np.nanmin(res_turbulence),vmax=np.nanmax(res_turbulence))
        fig.colorbar(im,orientation='vertical', label='OPD [nm]')
        ax.set_title('Corrected phase')        
    return res_turbulence    
    
def ResidualSlopesSequence2ResidualPhase(Gradients,S2M,M2V,name='residual_phase_cube.fits',\
                      path='.',binning=40):
    """
    Same functions as ResidualSlopes2ResidualPhase but applies it to a sequence 
    of slopes (gradients) instead of a single vector. 
    Input:
        - Gradients: a sequence of vectors of 2480 elements representing the slopes measured
            by the WFS
        - S2M: the slopes to modes matrix (shape (988, 2480))
        - M2V: the modes to voltage matrix (shape (1377, 988))
        - name: the name of the file to save
        - path: path where the file is saved
        - binning: the binning factor to apply (40 by default). If no binning 
            is desired, use binning=1
    """
    if Gradients.ndim != 2 and Gradients.shape[1] != 2480:
        raise IOError('The input vector must be a 2D array of shape (nframes,2480) (currently',Gradients.shape,')')    
    if S2M.ndim!=2 or S2M.shape[1]!=2480:
        raise IOError('The input S2M matrix must have a shape (988, 2480) (currently',S2M.shape,')')
    if M2V.ndim!=2 or M2V.shape[0]!=1377:
        raise IOError('The input M2V matrix must have a shape (1377, 988) (currently',M2V.shape,')')
    IMF = fits.getdata(os.path.join(path_sparta,'SAXO_DM_IFM.fits')) #shape (1377, 240, 240)
    # The matrix needs to be normalised to allow conversion into optical wavefront errors:
    # influence matrix normalization = defoc meca in rad @ 632 nm
    rad_632_to_nm_opt = 1. / 2. / np.pi * 632 * 2
    IMF   = IMF * rad_632_to_nm_opt
    IMF   = IMF.reshape(1377, 240*240).T # shape (57600, 1377)

    nframes = Gradients.shape[0]
    mode  = Gradients @ S2M.T
    volt  = mode @ M2V.T
    res_turbulence = (volt @ IMF.T).reshape((nframes, 240, 240))

    if binning ==1:
        nframes = Gradients.shape[0]
        slopes = Gradients
    elif binning>1:
        slopes = np.ndarray((int(Gradients.shape[0]/binning),2480),dtype=float)
        for i in range(int(nframes/binning)):
            slopes[i,:] = np.mean(Gradients[i*binning:(i+1)*binning,:],axis=0)
        nframes = int(Gradients.shape[0]/binning)
    else:
        raise IOError('The binning factor must be an integer greater or equal to 1.')           
    mode  = slopes @ S2M.T
    volt  = mode @ M2V.T
    res_turbulence = (volt @ IMF.T).reshape((nframes, 240, 240))
    fits.writeto(os.path.join(path,name), res_turbulence, overwrite=True)
    return res_turbulence

    
def locateSubapertures(subapertureIndices):
    """
    Takes in input a list of subapertures indices and plots their location on the SH
    It returns their position on a 40x40 map
    """
    SH_vect = np.zeros(1240,dtype=int)
    SH_vect[subapertureIndices] = 1
    SH_mask = SHvect2map(SH_vect)
    return SH_mask
    
def locateActuator(actuatorIndices):
    """
    Takes in input a list of actuator indices and plots their location on the DM
    It returns their position on a 40x40 map
    """
    DM_vect = np.zeros(1377,dtype=int)
    DM_vect[actuatorIndices] = 1
    DM_mask = DMvect2map(DM_vect)
    return DM_mask

def update_sparta_keyword(files,keys=['HIERARCH ESO DPR TYPE'],values=['OBJECT,AO'],inplace=False):
    """
    Funtion used to update the primary header of a sparta file (or any file containing 
    a primary HDU and additional table in extensions).
    Input:
        - files: string containing the file neame  that might ontain a wildcard, like
        SPHERE_GEN_SPARTA*.fits
        - keys: a list of keys ot update (for instance HIERARCH ESO DPR TYPE)
        - values: a list of new values for these keys
        -inplace: boolean to indicate if one wants to replace the existing files
        or if one wantes to save updated files in a new file
    """
    filenames = glob.glob(files)
    for filename in filenames:
        if inplace:
            new_filename=filename
        else:
            new_filename=filename.replace('.fits','_updated.fits')
        hdu_list = fits.open(filename)
        for i,key in enumerate(keys):
            hdu_list[0].header[key] = values[i]
            prihdu = fits.PrimaryHDU(data=hdu_list[0].data,header=hdu_list[0].header)
            hdulist = fits.HDUList([prihdu])
            for i in np.arange(1,7):
                tbhdu = fits.BinTableHDU(hdu_list[i].data,header=hdu_list[i].header)
            hdulist.append(tbhdu)
            hdulist.writeto(new_filename,clobber=True,output_verify='ignore')   
        hdu_list.close()
    return
  

def plot_DTTS_image(cube,indices=None,path='.',name=None,text=None,minpx=None,maxpx=None):
    """
    Saves a plot of a DTTS image with the correct pixel scale.
    Input:
        - cube: a 32x32 cube of DTTS images. It can also be a single image
        - indices: a list of indices from the cube. It can be a single index 
        if one wants to plot only a singgle image of the cube. If it is not specified,
        all images of the cube are plotted.
        - path: the path where the image must be saved. If the path does not exist,
        it is created automatically
        - text: the text to write in the image (None to write the index of the image,
            or any string, or empty string no to write anything.
    """
    if cube.ndim ==3:
        if cube.shape[1]!=32 or cube.shape[1]!=32:
            raise ValueError('The DTTS images must be 32 by 32 pixels')
    elif cube.ndim==2:
        cube = cube.reshape((1,32,32))
    nframes,ysize,xsize = cube.shape
    # From David email, the DTTS is at f/D=40.38 with pixels of 18 microns 
    # hence 11.5 mas/pix
    px = 0.0115
    if os.path.exists(path)==False:
        os.mkdir(path)
    if indices is None:
        indices = np.arange(nframes)
    for i in indices:
        img = cube[i,:,:]
        if minpx is None:
            minval=np.nanmin(img)
        else:
            minval=minpx
        if maxpx is None:
            maxval=np.nanmax(img)
        else:
            maxval=maxpx
        fig, ax = plt.subplots()
        im = ax.imshow(img, cmap='CMRmap', origin='lower',\
                       interpolation='nearest',\
                       norm=SymLogNorm(vmin=minval, vmax=maxval,linthresh=0.1))
        im.set_extent([-xsize/2*px,(xsize/2-1)*px,-ysize/2*px,(ysize/2-1)*px])
        ax.set_xlabel('Distance in arcsec')
        ax.set_ylabel('Distance in arcsec')
        if text is None:
            ax.text(0,0.15,"Frame {0:04d}".format(i),\
                    horizontalalignment='center',fontsize=18,color='white')
        else:
            ax.text(0,0.15,text,horizontalalignment='center',\
                    fontsize=18,color='white')
        fig.colorbar(im)
        if name is None:
            fig.savefig(os.path.join(path,'DTTS_image_{0:04d}.pdf'.format(i)))
        else:
            fig.savefig(os.path.join(path,name))
        plt.close(fig)

def plot_irdis_image(path,image,name='image.pdf',minpx=None,maxpx=None,\
                     text=None,dpi=70,arrowdir=None,arrow_length=0.7,\
                     logNorm=False,pixel_size=0.01225):
    """
    Plot the image, optionnally with one or 2 arrows to represent the 
    wind directions.
    Input:
        - path: the path where the image is saved
        - image: a 2D image (numpy array)
        - name: the name of the image to save, with the extension (*.pdf, 
                or *png for a movie)
        - minpx, maxpx: the minimum and maximum flux value of the colour scale
        - text: a text 
        - dpi: the resolution (for high quality movies)
        - arrowdir: the direction of the arrow in degrees (0=north, 90=East).
            Typically, wind_dir - derotation_angle gives the correct direction 
            on the detetor.
            It can be either a float or a list of 2 directions if you want to plot 
            2 arrows (e.g. surface and altitude wind)
        - arrow_lenght: the length in arcsec
        - pixel_size: the pixek size in arcsec/px
    """
    ysize,xsize = image.shape
    if pixel_size is None:
        pixel_size = 0.01225
    fig, ax = plt.subplots()
    if minpx is None:
        minpx=np.nanmin(image)
    if maxpx is None:
        maxpx=np.nanmin(image)
    if logNorm:
        im = ax.imshow(image, cmap='CMRmap', origin='lower',interpolation='nearest',\
            norm=SymLogNorm(vmin=minpx, vmax=maxpx,linthresh=1))
    else:
        im = ax.imshow(image, cmap='CMRmap', origin='lower',interpolation='nearest',\
            vmin=minpx,vmax=maxpx)
    if arrowdir is not None:
        if isinstance(arrowdir,(int,float)):
            dy = arrow_length*np.cos(np.deg2rad(arrowdir))
            dx = -arrow_length*np.sin(np.deg2rad(arrowdir))
            ax.annotate("", xy=(dx, dy), xytext=(0, 0),xycoords='data',\
                        arrowprops=dict(arrowstyle="->",linewidth = 2.,color = 'black'))
        elif isinstance(arrowdir,list):
            colors=['black','red']
            for i_arrows in range(2):
                dy = arrow_length*np.cos(np.deg2rad(arrowdir[i_arrows]))
                dx = -arrow_length*np.sin(np.deg2rad(arrowdir[i_arrows]))
                ax.annotate("", xy=(dx, dy), xytext=(0, 0),xycoords='data',\
                            arrowprops=dict(arrowstyle="->",linewidth = 2.,color =colors[i_arrows]))                
    if text is not None:
        ax.text(0,-1,text,horizontalalignment='center',fontsize=18,color='white')
    #fig.subplots_adjust(right=0.8)
    #cbar = fig.colorbar(im)
    im.set_extent([-xsize/2*pixel_size,(xsize/2-1)*pixel_size,-ysize/2*pixel_size,(ysize/2-1)*pixel_size])
    ax.set_xlabel('Distance in arcsec')
    ax.set_ylabel('Distance in arcsec')
    cbar = fig.colorbar(im, orientation='vertical')
    cbar.set_label('Flux (arbitrary unit)', rotation=90)
    fig.savefig(os.path.join(path,name),dpi = dpi)
    plt.close()        

def IRPixelRecorder2cube(BinTableHDU,save=None):
    """
    Converts a header data unit coming from a Sparta binary table IRPixelRecorder
    into a cube of images.
    Input:
        - IRPixelRecorder: the Binary table contained in the 1st extension
        of the IRPixelRecorder file (for instance 
        2018-09-11T02-38-22-IRPixelRecorder.fits)
        - save: the complete filename of the cube to save
    Output:
        - the cube of images 
    """
    nframes = BinTableHDU.shape[0]
    nx = BinTableHDU[0][2]
    ny = BinTableHDU[0][1]
    arraysize = len(BinTableHDU[0][3])
    cube=np.ndarray((nframes,ny,nx))
    for i in range(nframes):
        cube_tmp = np.reshape(BinTableHDU[i][3],(arraysize//(nx*ny), ny, nx))
        cube[i,:,:] = cube_tmp[0,:,:]
    if save is not None:
        fits.writeto(save, cube, overwrite=True)
    return cube    
    
if __name__ == '__main__':
    
    path_test = '/Users/jmilli/Documents/internships/cpannetier/LWE_detection'
    cube_DTTS= fits.getdata(os.path.join(path_test,'DTTS_cube_2015-10-03.fits'))
    plot_DTTS_image(cube_DTTS,indices=[2,100,300,499],path=path_test)

