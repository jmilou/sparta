#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:22:00 2022



@author: millij
"""

import numpy as np
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy.coordinates import name_resolve
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import ICRS, FK5 #,FK4, Galactic

def query_simbad(date,coords,name=None,limit_G_mag=15,metadata=None):
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
            coords = SkyCoord(header['RA']*u.degree,header['DEC']*u.degree)
            SkyCoord('03h32m55.84496s -09d27m2.7312s', ICRS)
        - limit_G_mag: the limiting G magnitude beyond which we consider the star too 
            faint to be the correct target (optional, by default 15)
        - metadata : any additional information in the form of a dictionnary that
            one wants to pass in the ouptut dictionnary 
    Output:
        - a dictionary with the most interesting simbad keywords and the original 
            RA,DEC coordinates from the pointing position.
    """
    search_radius = 10*u.arcsec # we search in a 10arcsec circle.
    search_radius_alt = 220*u.arcsec # in case nothing is found, we enlarge the search
        # we use 210 arcsec because Barnard star (higher PM star moves by 10arcsec/yr --> 220 arcsec in 22yrs)

    if coords.ndim>0: # if coords is an array of SkyCoord we only take the 2st element to avoid issues with arrays.
        coords = coords[0]

    # The output of the function is simbad_dico.
    # We first populate it with the initial values to pass 
    simbad_dico = {}
    simbad_dico['RA'] = coords.ra.to_string(unit=u.hourangle,sep=' ')
    simbad_dico['DEC'] = coords.dec.to_string(unit=u.degree,sep=' ')
    simbad_dico['DATE'] = date.iso
    if type(metadata) is dict:
        for key,val in metadata.items():
            simbad_dico[key] = val           

    if name is not None:
        # here we can handle special cases where the object name is 47 Tuc for instance
        if np.logical_and('47' in name,'tuc' in name.lower()):
            name = 'Gaia EDR3 4689637789368594944'
        elif np.logical_and('3603' in name,'ngc' in name.lower()):
            name ='HD 97950B'
            # to be checked ! I'm not sure this is the AO star...
            print('NGC 3603 case not implemented yet')
        elif np.logical_and('6380' in name,'ngc' in name.lower()):
            name = 'Gaia EDR3 5961801153907816832'
        elif np.logical_and('theta' in name.lower(),'ori' in name.lower()):
            # I still have to find the coordinate of theta Ori B1 which is the other astrometric calibrator often used. 
            name = 'tet01 Ori B'        

        # then we try to resolve the name of the object directly.
        try:
            object_coordinate = SkyCoord.from_name(name.strip())
            separation = object_coordinate.separation(coords)
            print('Object - Pointing separation is {0:.2f}'.format(separation))
            if separation < search_radius_alt:
                print('The object found is likely the target')
                name = name.strip()
            else:
                print('The object found is likely not the target.')
                name = None
        except name_resolve.NameResolveError as e:
            print('Object {} not recognized'.format(name.strip()))
            print(e)
            name = None

        # at this point we have done our best to have a valid name recognized by 
        # Simbad. If this is the case, then name is a valid string. Otherwise, name is None.

    customSimbad = Simbad()
    customSimbad.add_votable_fields('flux(G)','flux(J)','flux(H)',\
                                    'flux(K)','id(HD)','sp','otype','otype(V)','otype(3)',\
                                   'propermotions','ra(2;A;ICRS;J2000;2000)',\
                                 'dec(2;D;ICRS;J2000;2000)',\
                                 'ra(2;A;FK5;J{0:.3f};2000)'.format(date.jyear),\
                                 'dec(2;D;FK5;J{0:.3f};2000)'.format(date.jyear))

    if name is not None:
        search = customSimbad.query_object(name)
        validSearch = search[search['FLUX_G']<limit_G_mag]
        nb_stars = len(validSearch)                
        # at this point normally there is one single star found
        if nb_stars == 1:
            simbad_dico = populate_simbad_dico(validSearch,0,simbad_dico)
            # we add the distance between pointing and current position in the dictionnary
            simbad_dico  = add_separation_between_pointing_current_position(coords,simbad_dico)
            return simbad_dico
        else:
            print('Something went wrong, there are {0:d} valid stars '.format(nb_stars))
            return None

    else: # in this case no name is provided
    
        # First we do a cone search around the coordinates
        search = customSimbad.query_region(coords,radius=search_radius)
        if search is not None:
            validSearch = search[search['FLUX_G']<limit_G_mag]
            nb_stars = len(validSearch)                
            if nb_stars==0:
                search = None
        if search is  None:
            # If the cone search failed and no name is provided we cannot do anything more
            print('No star identified for the RA/DEC pointing. Enlarging the search to {0:.0f} arcsec'.format(search_radius_alt.value))
            search = customSimbad.query_region(coords,radius=search_radius_alt)
            if search is None:
                print('No star identified for the RA/DEC pointing. Stopping the search.')
                return simbad_dico
            else:
                validSearch = search[search['FLUX_G']<limit_G_mag]
                nb_stars = len(validSearch)                
                        
        if nb_stars==0:
            print('No star identified for the RA/DEC pointing. Stopping the search.')
            return simbad_dico
        elif nb_stars>0:
            if nb_stars ==1:
                i_min=0
                print('One star found: {0:s} with G={1:.1f}'.format(\
                      validSearch['MAIN_ID'][i_min],validSearch['FLUX_G'][i_min]))
            else:
                print('{0:d} stars identified within {1:.0f} or {2:.0f} arcsec'.format(nb_stars,search_radius.value,search_radius_alt.value)) 
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
                    coords_i = SkyCoord(coord_str,frame=FK5,unit=(u.hourangle,u.deg))
                    sep_list.append(coords.separation(coords_i).to(u.arcsec).value)
                i_min = np.argmin(sep_list)
                min_sep = np.min(sep_list)
                print('The closest star is: {0:s} with G={1:.1f} at {2:.2f} arcsec'.format(\
                  validSearch['MAIN_ID'][i_min],validSearch['FLUX_G'][i_min],min_sep))
        simbad_dico = populate_simbad_dico(validSearch,i_min,simbad_dico)
        simbad_dico = add_separation_between_pointing_current_position(coords,simbad_dico)
        return simbad_dico

def populate_simbad_dico(simbad_search_list,i,simbad_dico):
    """
    Method not supposed to be used outside the query_simbad method
    Given the result of a simbad query (list of simbad objects), and the index of 
    the object to pick, creates a dictionary with the entries needed.
    """    
    for key in simbad_search_list.keys():
        if key in ['MAIN_ID','SP_TYPE','ID_HD','OTYPE','OTYPE_V','OTYPE_3']: #strings
            simbad_dico['simbad_'+key] = simbad_search_list[key][i]
        elif key in ['FLUX_G', 'FLUX_J', 'FLUX_H', 'FLUX_K','PMDEC','PMRA']: #floats
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

def add_separation_between_pointing_current_position(coords,simbad_dico):
    """
    Input: 
        - coords: a SkyCoord object. For instance, if we extract the keywords 
            of the fits files, we should use
            coords = SkyCoord(header['RA']*u.degree,header['DEC']*u.degree)
            SkyCoord('03h32m55.84496s -09d27m2.7312s', ICRS)
        - simbad_dico is a dictionnary containing the keys 
            ['simbad_MAIN_ID',
             'simbad_SP_TYPE',
             'simbad_ID_HD',
             'simbad_OTYPE',
             'simbad_OTYPE_V',
             'simbad_OTYPE_3',
             'simbad_FLUX_G',
             'simbad_FLUX_J',
             'simbad_FLUX_H',
             'simbad_FLUX_K',
             'simbad_PMDEC',
             'simbad_PMRA',
             'simbad_simbad_RA_current',
             'simbad_simbad_DEC_current',
             'simbad_simbad_RA_ICRS',
             'simbad_simbad_DEC_ICRS']
    The function adds the keys simbad_separation_RADEC_ICRSJ2000 and simbad_separation_RADEC_current
    corresponding to the distance between pointing and ICRS and current coordinates    
    It returns the updated dictionnary
    """
    if 'simbad_RA_ICRS' in simbad_dico.keys() and 'simbad_DEC_ICRS' in simbad_dico.keys():
        coords_ICRS_str = ' '.join([simbad_dico['simbad_RA_ICRS'],simbad_dico['simbad_DEC_ICRS']])
        coords_ICRS = SkyCoord(coords_ICRS_str,frame=ICRS,unit=(u.hourangle,u.deg))
        sep_pointing_ICRS = coords.separation(coords_ICRS).to(u.arcsec).value
        simbad_dico['simbad_separation_RADEC_ICRSJ2000'] = sep_pointing_ICRS
    # if we found a star, we add the distance between Simbad current coordinates and pointing
    if 'simbad_RA_current' in simbad_dico.keys() and 'simbad_DEC_current' in simbad_dico.keys():
        coords_current_str = ' '.join([simbad_dico['simbad_RA_current'],simbad_dico['simbad_DEC_current']])
        coords_current = SkyCoord(coords_current_str,frame=ICRS,unit=(u.hourangle,u.deg))
        sep_pointing_current = coords.separation(coords_current).to(u.arcsec).value
        simbad_dico['simbad_separation_RADEC_current']=sep_pointing_current
        print('Distance between the current star position and pointing position: {0:.1f}arcsec'.format(simbad_dico['simbad_separation_RADEC_current']))
    return simbad_dico

if __name__ == "__main__":
    """
    This is just an example of how the script can be used
    """

    ra = 10.*u.degree
    dec = -24*u.degree
    testCoord = SkyCoord(ra,dec)
    date = Time('2017-01-01T02:00:00.0')
    test=query_simbad(date,testCoord,name='eps Eri',limit_G_mag=15)
    print(test)
    
    testCoord = SkyCoord(['09 23 47.1 +20 21 52.034'], frame=ICRS, unit=(u.hourangle, u.deg))
    date = Time('2017-07-08T02:00:00.0')
    test=query_simbad(date,testCoord,name='beta Pictoris',limit_G_mag=15)
    print(test)
