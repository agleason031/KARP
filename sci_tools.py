#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 17:01:27 2025

@author: agleason
"""
from scipy.ndimage import uniform_filter
from scipy.optimize import curve_fit
from scipy.integrate import quad
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from astropy import units as u
from astropy.stats import mad_std
from astropy.nddata import CCDData
from functions import G
import numpy as np
import ccdproc
import csv

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
def get_cal_images(blist, flist, verbose, grapher):
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
            
    if (verbose == True):
        grapher.plot_masterbias(masterbias)
        grapher.plot_final_flat(final_flat)
            
    final_flat_write = CCDData(final_flat, unit="adu")
    final_flat_write.write(grapher.target_dir+"OUT/"+grapher.objid+"_final_flat.fits", overwrite = True)
    
    print("Final Flat Made")
        
    return masterbias, final_flat
            
def fit_metal_lines(gwave, med_comb, sig_final, grapher):
    #find equivalent widths of metal lines
    metal_lines = [3820.425, 3933.66, 4045.812, 4063.594, 4226.728, 4260.474, 4271.76, 4307.902, 4383.545, 4404.75, 4957.596, 5167.321, 5172.684, 5183.604, 5269.537, 5328.038]
    metal_mask = [4, 3, 4, 4, 3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4]
    metal_fit = []
    metal_results = []
    
    #iterate over metal lines       
    for i, line in enumerate(metal_lines):
        ew_sum = 0
        		
        mask = (gwave > line - metal_mask[i]) & (gwave < line + metal_mask[i])
        flux_vals = med_comb[mask]
        wave_vals = gwave[mask]
        wave_err = sig_final[mask]
        p0 = [-np.min(flux_vals), line, 2.0, 1.0]
        bounds = [(-np.inf,0,1, 0.9),(0,np.inf,np.inf, 1.1)]
           
        #attempt fit
        try:
            popt, pcov = curve_fit(G, wave_vals, flux_vals, p0=p0, bounds=bounds, maxfev=5000)
            metal_fit.append(popt)
        except RuntimeError:
            print("Curve fit failed")
        
        #define function used for fitting    
        line_fit = lambda x: G(x, popt[0], popt[1], popt[2], popt[3])
        integrated, err_integrated = quad(line_fit, line - metal_mask[i], line + metal_mask[i])
        equi_width = (metal_mask[i] * 2 * popt[3]) - integrated
        
        #riemann sum equivalent width
        for val in flux_vals:
            ew_sum += 0.5 * (popt[3] - val)
        
        #error calculation
        error = np.sqrt(np.sum(np.square(wave_err * 0.5)))
        sys = .002 * metal_mask[i] * 2
        adopt_err = np.sqrt(error**2 + sys**2)
        
        #create csv of data output
        metal_results.append([line, equi_width, ew_sum, error, sys, (equi_width + ew_sum) / 2, adopt_err, metal_mask[i]])
        
        print("------------------")
        print(f"Metal line at {line}")
        print(f"EW_g is {equi_width:.4f} and EW is {ew_sum:.4f}")
        print(f"Error is {error:.4f} and sys is {sys}")
        print(f"Adopted is {(equi_width+ew_sum)/2:.4f} with error {adopt_err:.4f}")
         
    # Write results to CSV after the loop
    csv_path = grapher.target_dir + f"OUT/{grapher.objid}_metal_lines.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Line (A)", "EW_g", "EW_riemann", "Error", "Sys", "Adopted_EW", "Adopted_Error, Width of Mask"
        ])
        writer.writerows(metal_results)
    print(f"Metal line results written to {csv_path}")

    #make graphs of fits around lines
    grapher.metal_line_plt(metal_fit, gwave, mask, med_comb, metal_lines)
    
    #make big plot of all metal lines
    grapher.metal_line_big_plt(metal_fit, metal_lines, metal_mask, gwave, med_comb)
    print(f"Big summary plot saved to {grapher.target_dir}OUT/{grapher.objid}_metal_lines_all.png")

def fit_vel_lines(gwave, med_comb, grapher):
    # Measure the radial velocity of the star based on these 7 absorption lines
    print("Measuring radial velocity of the star")
    # Line names: H theta, H eta, H zeta, H delta, H gamma, H alpha, Ca K, Don;t use: Hbeta, HeI(sdB) 4471.5, HeI 4173, HeI 4922, HeI 5016
    vel_lines = [3797.91,3835.40,3889.06,4101.73,4340.47,6562.79,3933.66] # In angstroms ,,4861.35,4471.5,4713,4922
    vel_mask = [5,5,5,5,5,5,10,10,10,10,10] # half-width of lines to fit in angstroms
    vel_fit = [] # The fit parameters for the velocity fits
    vel_fit_linenum = [] # Which of the seven lines we just fit
    
    # Fit a gaussian to the core of an absorption feature
    for i in range(len(vel_lines)):
        # We want the flux values around the line centers
       
        mask = (gwave > vel_lines[i] - vel_mask[i]) & (gwave < vel_lines[i] + vel_mask[i])
        flux_vals = med_comb[mask]
        wave_vals = gwave[mask]
        
        p0 = [-np.min(flux_vals), vel_lines[i], 2.0, 1] # Guess that the minimum flux is the peak of the fit
        # and that the offset is 1 (accounting for normilization), and that the center is the vel_lines value
        bounds = [(-np.inf,0,1,0.5),(0,np.inf,np.inf,1.1)] # a, mu, sig, bck
        try:
            # Curve fit takes, x (locations), y (values)
            popt, _ = curve_fit(G, wave_vals, flux_vals, bounds=bounds, p0=p0, maxfev=5000) # ignore covariance matrix spat out from curve_fit
            vel_fit.append(popt)  # [a, mu, sigma, c] returend from curve_fit
            vel_fit_linenum.append(i) # Wavelength values that where succesfully fit
        except RuntimeError:
            print("Curve fit failed at maxfev=1000")
            
    # Now convert the measured center of the absorption feature from angstroms to km/s
    # Line names: H theta, H eta, H zeta, H delta, H gamma, H alpha, Ca K
    line_velocities_karp = [] # The velocites of the absorption lines that we get from KARP
    for i in range(len(vel_fit)):
        mu= vel_fit[i][1]
        line_velocities_karp.append(((float(mu)-float(vel_lines[i]))/float(vel_lines[i]))*3*10**5)
    
    #plot the region around graphs
    grapher.plot_vel_lines(vel_fit, vel_fit_linenum, vel_lines, gwave, med_comb, vel_mask)
    
    # average the velocities of the 7 lines:
    radial_vel = np.mean(line_velocities_karp)
    radial_vel_err = np.std(line_velocities_karp)
    print("Radial velocity:",radial_vel,"+/-", radial_vel_err)        
    # Look for HI line, 4471.5