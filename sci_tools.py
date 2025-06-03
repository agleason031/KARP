#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 17:01:27 2025

@author: agleason
"""
from scipy.ndimage import uniform_filter
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from astropy import units as u
from astropy.stats import mad_std
import numpy as np
import ccdproc

#returns correction needed for earth's motion
def heliocentric_correction(objRA, objDEC, otime):
    # Find heliocentric correction velocity
    # lam = lam * (1+v)/c in km/s
    # Take the observed date and the location of APO
    # Calculate the heliocentric velocity of earth relative to the star
    # This is the velocity, v that we include in the code
    
    # Megan's heliocentric function
    #EX: helioCorrect("05 02 58.72 -70 49 44.7", "Las Campanas Observatory", "2017-10-26 05:03")
    #EX: t = Time("2015-06-30 23:59:60.500")
    coord = str(objRA + " " + objDEC) # Get Coordinates
    site = EarthLocation.of_site('Apache Point Observatory') # Get the site (Apache point observatory)
    t =  Time(otime)
    sc = SkyCoord(coord, unit=(u.hourangle, u.deg)) # Make a SkyCoord Object for our star
    heliocorr = sc.radial_velocity_correction('heliocentric', obstime=t, location=site) # Heliocentric radial velocity correction
    #print("heliocorr",heliocorr)
    # Map to v_helio
    correction = heliocorr.to(u.km/u.s)
    #print("correction",correction)
    correction = str(correction)
    correction = correction.replace('km / s', '')
    return(float(correction))

#creates and returns bias and flat images
def get_cal_images(blist, flist):
    # Make a master bias from the input bias image
    print("Making Master Bias")
    masterbias = ccdproc.combine(blist, method='median', unit='adu',sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,mem_limit=350e6)
    
    # Make a master flat from the flat images
    print("Making Master Flat")
    masterflat = ccdproc.combine(flist, method='median', unit='adu',sigma_clip=False,mem_limit=350e6)
  
    # Subtract Master Bias from Master Flat
    masterflatDEbias = ccdproc.subtract_bias(masterflat,masterbias)
    
    # Smooth master flatdebias:
    np_mfdb = np.asarray(masterflatDEbias) # Convert to a numpy array        
    smooth_mf_mb = uniform_filter(np_mfdb, size=5)
    
    # Make a Final Flat by dividing mf_mb by smooth_mf_mb
    final_flat = np_mfdb/smooth_mf_mb
    # Y top of 0th order line 2025ish
    # Y bottom of 0th order line 1995ish
    # Go ten pixels on either side
    # X=165-835
    xZo = np.arange(165,835,1)
    yZo = np.arange(1985,2035,1)
    
    # Remove zeorth orther line
    for i in range(len(xZo)):
        for k in range(len(yZo)):
            final_flat[yZo[k], xZo[i]] = 1
      
    # Remove other bad features
    xOther1 = np.arange(165,835,1)
    yOther1 = np.arange(2074,2083,1)
    
    for i in range(len(xOther1)):
        for k in range(len(yOther1)):
            final_flat[yOther1[k], xOther1[i]] = 1 # Set the flat value of these small areas to 1
            # Note that this minimally affects the final flat as these areas are relativly small
            # So we still get a representative final flat image
    
    for i in range(len(xOther1)):
        for k in range(len(yOther1)):
            final_flat[yOther1[k], xOther1[i]] = 1
            
    return masterbias, final_flat
            
    