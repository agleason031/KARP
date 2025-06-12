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
from numba import njit, prange
import numpy as np
import ccdproc
import csv
import inspect
import os

_calibration_cache = {}

def get_var_name(var):
    frame = inspect.currentframe().f_back  # Get the caller's frame
    local_vars = frame.f_locals
    for name, value in local_vars.items():
        if value is var:
            return name
    return None  # If no match is found

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
    #get images from cache if already created
    key = grapher.objid
    bias_path = grapher.target_dir + "OUT/" + grapher.objid + "_master_bias.fits"
    flat_path = grapher.target_dir + "OUT/" + grapher.objid + "_final_flat.fits"
    
    if key in _calibration_cache:
        return _calibration_cache[key]
    
    # ✅ Load from disk if both exist
    if os.path.exists(bias_path) and os.path.exists(flat_path):
        print("Loading disk calibration files...")
        masterbias = CCDData.read(bias_path, unit='adu')
        final_flat = CCDData.read(flat_path, unit='adu')
        _calibration_cache[key] = (masterbias, final_flat)
        return masterbias, final_flat
    
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
        grapher.plot_image(masterbias, get_var_name(masterbias))
        grapher.plot_image(final_flat, get_var_name(final_flat))
    
    masterbias.write(bias_path, overwrite=True)
    CCDData(final_flat, unit="adu").write(flat_path, overwrite=True)
    
    print("Final Flat Made")
    
    _calibration_cache[key] = (masterbias, final_flat)
    return masterbias, final_flat
            
def fit_metal_lines(gwave, med_comb, sig_final, metal_mask, radial_vel, grapher):
    #find equivalent widths of metal lines
    metal_lines = [3820.425, 3933.66, 4045.812, 4063.594, 4226.728, 4260.474, 4271.76, 4307.902, 4383.545, 4404.75, 4957.596, 5167.321, 5172.684, 5183.604, 5269.537, 5328.038]
    metal_fit = []
    metal_results = []
    center_shifts = [((radial_vel / (3*10**5)) * line) for line in metal_lines]
    masks = [(gwave > metal_lines[i] - metal_mask[i] + center_shifts[i]) & (gwave < metal_lines[i] + metal_mask[i] + center_shifts[i]) for i in range(len(metal_lines))]
    
    #makes list of bin sizes at different wavelengths
    gwave_dif = []
    for i in range(len(gwave)):
        try:
            gwave_dif.append((gwave[i+1]-gwave[i-1])/2)
        except IndexError:
            gwave_dif.append(0.5)
    gwave_dif = np.array(gwave_dif)
    
    #iterate over metal lines       
    for i, line in enumerate(metal_lines):        	
        flux_vals = med_comb[masks[i]]
        if len(flux_vals) == 0:
            print(f"Line {line} is out of range")
            metal_fit.append([0, 0, 0, 1])
            continue
        wave_dif = gwave_dif[masks[i]]
        wave_vals = gwave[masks[i]]
        wave_err = sig_final[masks[i]]
        p0 = [max(-1+np.min(flux_vals), -0.01), line + center_shifts[i], 2.0, 1.0]
        bounds = [(-np.inf,0,1, 0.9),(0,np.inf,np.inf, 1.1)]
           
        #attempt fit
        try:
            popt, pcov = curve_fit(G, wave_vals, flux_vals, p0=p0, bounds=bounds, maxfev=1000)
            metal_fit.append(popt)
        except RuntimeError:
            print("------------------")
            print("Curve fit failed")
            print(f"Metal line at {line}")
            metal_fit.append([0, 0, 0, 1])
            continue
        
        #define function used for fitting    
        line_fit = lambda x: G(x, popt[0], popt[1], popt[2], popt[3])
        integrated, err_integrated = quad(line_fit, line - metal_mask[i] + center_shifts[i], line + metal_mask[i] + center_shifts[i])
        equi_width = (metal_mask[i] * 2 * popt[3]) - integrated
        
        #riemann sum equivalent width
        ew_sum = 0
        for j in range(len(flux_vals)):
            ew_sum += -wave_dif[j] * (popt[3] - flux_vals[j])
        
        #error calculation
        error = np.sqrt(np.sum(np.square(wave_err * wave_dif)))
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
    grapher.metal_line_plt(metal_fit, gwave, masks, med_comb, metal_lines, center_shifts)
    
    #make big plot of all metal lines
    grapher.metal_line_big_plt(metal_fit, metal_lines, masks, gwave, med_comb, center_shifts)
    print(f"Big summary plot saved to {grapher.target_dir}OUT/{grapher.objid}_metal_lines_all.png")

def fit_vel_lines(gwave, med_comb, grapher):
    # Measure the radial velocity of the star based on these 7 absorption lines
    print("Measuring radial velocity of the star")
    # Line names: H theta, H eta, H zeta, H delta, H gamma, H alpha, Ca K, Don;t use: Hbeta, HeI(sdB) 4471.5, HeI 4173, HeI 4922, HeI 5016
    vel_lines = [3797.91,3835.40,3889.06,4101.73,4340.47,6562.79,3933.66] # In angstroms ,,4861.35,4471.5,4713,4922
    vel_mask = [10,10,10,10,10,10,10,10,10,10,10] # half-width of lines to fit in angstroms
    vel_fit = [] # The fit parameters for the velocity fits
    vel_fit_linenum = [] # Which of the seven lines we just fit
    
    # Fit a gaussian to the core of an absorption feature
    for i in range(len(vel_lines)):
        # We want the flux values around the line centers
       
        mask = (gwave > vel_lines[i] - vel_mask[i]) & (gwave < vel_lines[i] + vel_mask[i])
        flux_vals = med_comb[mask]
        if len(flux_vals) == 0:
            print(f"Line {vel_lines[i]} is out of range")
            continue
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
        line_velocities_karp.append(((float(mu)-float(vel_lines[vel_fit_linenum[i]]))/float(vel_lines[vel_fit_linenum[i]]))*3*10**5)
    
    #plot the region around graphs
    grapher.plot_vel_lines(vel_fit, vel_fit_linenum, vel_lines, gwave, med_comb, vel_mask)
    
    # average the velocities of the 7 lines:
    radial_vel = np.mean(line_velocities_karp)
    radial_vel_err = np.std(line_velocities_karp)
    print("Radial velocity:",radial_vel,"+/-", radial_vel_err)        
    # Look for HI line, 4471.5
    return radial_vel
    
@njit
def fit_flux(i, Fnorm_Es, wavelengths_np, flux_masked, include, removed_index,
    flux_raw_cor1_np, flux_masked_global, sky_raw1, norm_line_width, rm_param=1.3):
    
    if Fnorm_Es[i] > rm_param:
        return np.full(8, np.nan)  # Skip

    lam_left = max(0, i - norm_line_width)
    lam_right = min(len(wavelengths_np), i + norm_line_width)
    
    wavelength = wavelengths_np[i]
    sky = sky_raw1[i]
    flux = flux_raw_cor1_np[i]

    wave_range = []
    flux_values = []
    if removed_index[i] == 0:
        # Build boolean mask
        for k in range(lam_left, lam_right):
            if include[k] and removed_index[k] == 0:
                wave_range.append(wavelengths_np[k])
                flux_values.append(flux_raw_cor1_np[k])
        if len(wave_range) < 2:
            return np.full(8, np.nan)
        #Linear fit
        ml, bl = linear_fit_numba(np.array(wave_range), np.array(flux_values))
    else:
        for k in range(lam_left, lam_right):
            val = flux_masked_global[k]
            if not np.isnan(val):
                wave_range.append(wavelengths_np[k])
                flux_values.append(val)
        if len(wave_range) < 2:
            return np.full(8, np.nan)
        ml, bl = linear_fit_numba(np.array(wave_range), np.array(flux_values))
        
    fitted_value = ml * wavelength + bl
    flux_m = flux_masked[i]
    
    if not np.isnan(flux_m):
        return np.array([fitted_value, sky, flux, wavelength, flux_m, np.nan, wavelength, np.nan])
    else:
        return np.array([fitted_value, sky, flux, wavelength, np.nan, flux, np.nan, wavelength])

@njit
def fit_all_fluxes(Fnorm_Es, wavelengths_np, flux_masked, include, removed_index,
                   flux_raw_cor1_np, flux_masked_global, sky_raw1, norm_line_width):
    n = len(Fnorm_Es)
    results = np.full((n, 8), np.nan)
    for i in prange(n):
        results[i] = fit_flux(i, Fnorm_Es, wavelengths_np, flux_masked, include, removed_index,
                                    flux_raw_cor1_np, flux_masked_global, sky_raw1, norm_line_width)
    return results

@njit
def linear_fit_numba(x, y):
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    sx = np.sum(x)
    sy = np.sum(y)
    sxx = np.sum(x * x)
    sxy = np.sum(x * y)
    denom = n * sxx - sx * sx
    if denom == 0:
        return np.nan, np.nan
    m = (n * sxy - sx * sy) / denom
    b = (sxx * sy - sx * sxy) / denom
    return m, b