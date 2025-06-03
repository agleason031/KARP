#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
KARP

KOSMOS Astronomical Reduction Pipeline

Created by Chase L. Smith, Max Moe

><(((º>
"""

# Importing relevant packages
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter
from datetime import timedelta
from astropy.table import Table
from astropy.nddata import CCDData
from astropy.stats import mad_std
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import argparse
import ccdproc
import warnings
import glob
import time
import os
import pygame
import csv
import grapher
import functions
import sci_tools

#-------------------------------
#constant definitions

#sound related
pygame.mixer.init(frequency=44100, size=-16, channels=1)
# Generate sine wave
duration = 0.07 # seconds
freq = 777  # Hz
freq1 = 600
sample_rate = 44100
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
waveform = np.sin(2 * np.pi * freq * t)
waveform1 = np.sin(2 * np.pi * freq1 * t)
# Convert to 16-bit signed integers
audio = np.int16(waveform * 32767).tobytes()
audio1 = np.int16(waveform1 * 32767).tobytes()
# Create sound objects
sound = pygame.mixer.Sound(buffer=audio)
sound1 = pygame.mixer.Sound(buffer=audio1)


#----------------------------------
#function definitions

def run_fit(args):
    row, clr, a_width, buff_width, bckrnd_width = args
    return functions.fit_cent_gaussian(row, clr, a_width, buff_width, bckrnd_width)

def nan_aware_smooth(arr, window=100):
    """
    Replace NaNs in an array by computing a moving average
    over a fixed window, ignoring NaNs in the computation.
    """
    valid = ~np.isnan(arr) # Boolean mask of valid values
    smoothed = uniform_filter1d(np.nan_to_num(arr), size=window, mode='nearest')  # Sum of values (zeros where NaN)
    norm = uniform_filter1d(valid.astype(float), size=window, mode='nearest')     # Count of valid values
    with np.errstate(invalid='ignore'): # Ignore divide-by-zero warnings
        result = smoothed / norm  # True local mean (ignoring NaNs)
        result[~valid] = result[~valid] # Replace only NaNs
    return result

# Take parameters from the KparamGXXXXXX.txt file
def take_bait(file):
    """
    Parameters
    ----------
    file : text file at some location
        Input parameters for KARP to run
    Returns
    -------
    params : Python readable parameters of various formats
    """
    params = {} # output params and their values
    try:
        with open(file, 'r') as file: # Open the file
            for line in file: # Read in each line
                line = line.strip() # removes whitespace
                if line and not line.startswith("#"): # ignore comments (lines that start with #)
                    key, value = line.split("=", 1) # Split at equals sign
                    params[key.strip()] = value.strip() # Make each param equal its value
    # Error checking:
    except FileNotFoundError:
        print(f"Error: KARP could not find file: {file}")
        return None
    except ValueError:
        print(f"Value error in line: {line}")
        return None
    # return values
    return params

#input string true or false and returns bool
def string_to_bool(s):
    return {"true": True, "false": False}.get(s.lower())

def y_lam(y):
    # Vectorize array
    y = np.array(y)
    # Wavelength as a function of pixel with helocentric velocity correction
    return ((a*(y/2000)**3)+(b*(y/2000)**2)+c*(y/2000)+d)*(1+(v_helio/(3*10**5)))

# Helper function to format image filenames
def format_fits_filename(dloc, num):
    return f"{dloc}k.{int(num):04d}.fits"

#------------------------------------

if __name__ == "__main__":
    # Let the user know the program has started
    print("KARP is swimming! (Code is running)")
    print("><(((º>")
    print("- ><(((º>")
    print("- - ><(((º>")
    print("- - - ><(((º>")
    print("- - - - ><(((º>")
    print("- - - - - ><(((º>")
    starttime = time.perf_counter()
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Take the parameter file from the command line
    parser = argparse.ArgumentParser(prog="KARP")
    parser.add_argument('-kparam', '--kp') # Location of our data
    args = parser.parse_args()
    
    
    #get input parameters
    params = take_bait(str(args.kp))
    
    # assign relevant values
    dloc = str(params["dloc"]) # Location to store output data
    objid = str(params["objid"]) # Object ID
    objRA = str(params["objRA"]) # Object RA in the form "05 02 58.72"
    objDEC = str(params["objDEC"]) # Object DEC in the form "-70 49 44.7"
    otime = str(params["obsTime"]) # The time the object was observed, listed as t = Time("2015-06-30 23:59:60.500")
    scinum = int(params["scinum"]) # Sci image number that KARP is currently reducing
    argnum = int(params["argnum"]) # The image number of the Argon spectral calibration image
    fstart = int(params["fstart"]) # The first flat image, assuming the flats are ordered sequentially
    fstop = int(params["fstop"]) # The last flat image, assuming the flats are ordered sequentially
    bstart = int(params["bstart"]) # The first bias image assuming the bias images are ordered sequentially
    bstop = int(params["bstop"]) # The last bias image, assuming the bias images are ordered sequentially
    appw = int(np.round(float(params["appw"])-1)/2) # So an appature width inputu of 13 is two 6 pixel sides, and if you put 14 it rounds to 13
    buffw = int(params["buffw"]) # Width of the buffer in pixels (on either side of the appature, so 4 is two, 4 pix buffers su,etrically spaced around the center line)
    bckw = int(params["bckw"]) # The width of the background in pixels, (on either side of the buffer)
    clmin = int(params["clmin"]) # The minimum center line trace value that KARP will fit
    clmax = int(params["clmax"]) # The minimum center line trace value that KARP will fit
    # (Be warry of signal beyond this line, KARP will try it's best to have the line be within these bounds)
    norm_line_width = int(params["norm_line_width"])  # When normalizing spectra, this is the one-half width of the line KARP will try and fit at each point
    norm_line_boxcar = int(params["norm_line_boxcar"]) # When normalizing the spectra, 
    target_dir = str(params["target_dir"])
    skip_red = string_to_bool(params["skip_red"])
    comb_spec = string_to_bool(params["comb_spec"])
    fit_lines = string_to_bool(params["fit_lines"])
    
    print("KARP has taken the bait! (Input file):")
    print("------------")
    print(params) # Print out parameters
    print("------------")
    
    # Target directory should be something like /d/ori1/csmit/metalpoor/RED_G093440
    # Syntax of target directory is RED_G123456, with RED=reduced
    # Test input: -data_loc /d/ori1/csmit/metalpoor/Chase_G093440/Chase_G093440/ -sci_im_num 61 -arg_im_num 64 -f_start 1 -f_stop 7 -b_start 9 -b_stop 17 -appw 13 -buffw 2 -bckw 7 -cline_min 479 -cline_max 492 -target_dir /d/ori1/csmit/metalpoor/RED_G093440/
    
    if skip_red == False:
        print("KARP is doing reduction!")
    elif skip_red == True:
        print("KARP is SKIPPING reduction!")
    
    # initialize value so KARP runs empty values if skip_red=True is run by accident
    flux_raw_cor1 = []
    wavelengths = []
    # Global variables
    fim = np.arange(fstart,fstop,1)
    bim = np.arange(bstart,bstop,1)
    
    # Make a global "OUT" folder to keep finalized outputs
    os.makedirs(str(target_dir)+"OUT", exist_ok=True) # Initializes a directory OUT files
    # Make a output text file detaling relevant fit paramaters that KARP calculates
    # Initalize directory for reduction plots for this science image
    os.makedirs(str(target_dir)+"ImageNumber_"+str(scinum), exist_ok=True) # Initializes a directory for E20 files
    with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "w") as f:
        f.write("KARP Output Log\n")
        f.write("- - - ><(((º>\n")
        f.write("\n")
    
#------------------------------------------
    
    if skip_red == False:
        # Create lists of file paths for the Flat and Bias images
        
        print("KARP is reducing Science Image number:",scinum)
        with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
            f.write("KARP is reducing Science Image number:"+str(scinum)+"\n")
            f.write("KARP has taken the bait! (Input file):\n")
            f.write("------------\n")
            f.write(str(params))
            f.write("------------\n")
        
        #get image filenames
        blist = [format_fits_filename(dloc, b) for b in bim]
        flist = [format_fits_filename(dloc, f) for f in fim]
        sci_location = format_fits_filename(dloc, scinum)
        
        # Make a master bias from the input bias image
        print("Making Master Bias")
        masterbias = ccdproc.combine(blist, method='median', unit='adu',sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,mem_limit=350e6)
        grapher.plot_masterbias(masterbias, target_dir, scinum, objid)
        
        # Make a master flat from the flat images
        print("Making Master Flat")
        masterflat = ccdproc.combine(flist,method='median', unit='adu',sigma_clip=False,mem_limit=350e6)
        grapher.plot_masterflat(masterflat, target_dir, scinum, objid)
        
        # Subtract Master Bias from Master Flat
        masterflatDEbias = ccdproc.subtract_bias(masterflat,masterbias)
        grapher.plot_masterflatDEbias(masterflatDEbias, target_dir, scinum, objid)
        
        # Smooth master flatdebias:
        np_mfdb = np.asarray(masterflatDEbias) # Convert to a numpy array        
        smooth_mf_mb = uniform_filter(np_mfdb, size=5)
        grapher.plot_smooth_mf_mb(smooth_mf_mb, target_dir, scinum, objid)
        
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
        
        xOther2 = np.arange(165,835,1)
        yOther2 = np.arange(2048,2057,1)
        
        for i in range(len(xOther1)):
            for k in range(len(yOther1)):
                final_flat[yOther1[k], xOther1[i]] = 1 # Set the flat value of these small areas to 1
                # Note that this minimally affects the final flat as these areas are relativly small
                # So we still get a representative final flat image
        
        for i in range(len(xOther1)):
            for k in range(len(yOther1)):
                final_flat[yOther1[k], xOther1[i]] = 1
        
        grapher.plot_final_flat(final_flat, target_dir, scinum, objid)
        print("Final Flat Made")
        
        # Subtract off the bias from each of our science images
        # Note that we aren't stacking the sci images just yet,
        # we wait to do that at the end, so we need to do this
        # for each of our sci images
        
        sci_final = (ccdproc.subtract_bias(CCDData.read(sci_location,format = 'fits', unit = "adu"),masterbias).data)/final_flat
        grapher.plot_sci_final(sci_final, target_dir, scinum, objid)
        
        # Read all three of our final sci images and convert to nparrays
        sci_final_1 = np.asarray(CCDData.read(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"sci_final_"+str(scinum)+".fits", unit = "adu").data)
        
        # Convert from ADU to electrons
        sci_final_1 = sci_final_1*0.6
        
        # Shape of sci images is 4096, 1124
        a_width = int(appw) # Pixel aperture width
        buff_width = int(buffw) # Width of the buffer
        bckrnd_width = int(bckw) # Background width
        
#----------------------------
#trace fitting
        
        n_rows = sci_final_1.shape[0]  # This gets the number of rows
        print("n_rows:",n_rows)
        cen_line1 = np.round(np.linspace(int(clmin), int(clmax), n_rows)).astype(int)
        # Note that this repeats each integer 51 times, stretching it to 1124
        
        # Make cen_line a list for ease of use
        cen_line = []
        for i in range(len(cen_line1)):
            cen_line.append(int(cen_line1[i]))
        
        # Fit Gaussian to each row in sci_final_1 using current cen_line estimate
        print("KARP fitting centerline")
        
        # Fit and collect results in a list (each entry is a tuple: a, mu, sig, bck)
        args_list = [(row, cen_line[i], a_width, buff_width, bckrnd_width)
                 for i, row in enumerate(sci_final_1)]
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            cen_fit = list(tqdm(executor.map(run_fit, args_list), total=len(args_list)))
        
        # Convert to NumPy array and extract relevant fit parameters
        cen_fit = np.array(cen_fit)               # Shape: (N_rows, 4)
        a_vals = cen_fit[:, 0]                    # Amplitudes
        mu_vals = cen_fit[:, 1]                   # Gaussian centers (used for center line)
        
        # Create preliminary centerline using fitted mu values
        # If a or mu is NaN, mark the entry as NaN in cen_line for smoothing later
        cen_line = np.round(mu_vals).astype(float)
        cen_line[np.isnan(a_vals) | np.isnan(mu_vals)] = np.nan  # Replace invalid fits with NaN
        
        # Smooth NaN values in cen_line using a local (moving average) window
        # We use a NaN-aware uniform filter that preserves non-NaN values and smooths only missing ones
        print("Smoothing NaNs in centerline")
        smoothed_mu = nan_aware_smooth(cen_line, window=100)
        
        # Replace NaNs with their smoothed estimates
        cen_line[np.isnan(cen_line)] = np.round(smoothed_mu[np.isnan(cen_line)])
        
        # Final pass: Replace any remaining NaNs using local median of ±50 pixels
        print("Replacing remaining NaNs with local median")
        for i in range(len(cen_line)):
            if np.isnan(cen_line[i]):
                low = max(0, i - 50) # Use max and min to account for edges
                high = min(len(cen_line), i + 50)
                window_vals = cen_line[low:high]
                local_valid = window_vals[~np.isnan(window_vals)]
                if len(local_valid) > 0:
                    cen_line[i] = np.median(local_valid)
                else:
                    # As a last resort (edge case), use global median
                    cen_line[i] = np.nanmedian(cen_line)
        
        # Replace remaining outliers that deviate too much from their neighborhood
        # This helps fix spike-like errors or fit artifacts
        print("Replacing outlier values")    
        for i in range(50, len(cen_line) - 200):  # Avoid edges
            # Get 200-pixel local window of mu values
            local = mu_vals[i - 100:i + 100]
            local = local[~np.isnan(local)]       # Ignore any NaNs in local values
        
            if len(local) == 0:
                continue  # Skip if all neighbors are NaN
        
            cmean = np.mean(local)                # Local mean for comparison
        
            # Condition 1: Implausible high value (usually near image end)
            # Condition 2: Large deviation from local average
            if cen_line[i] > cen_line[-1] or abs(cen_line[i] - cmean) > 1.2:
                print(f"Outlier at {i}: Replacing {cen_line[i]} with {np.round(cmean)}")
                cen_line[i] = int(np.round(cmean))     # Replace with smoothed local mean
        
        # For the tails of the image if the fit becomes very poor, bin to the set max values
        for i in range(len(cen_line)):
            if cen_line[i] > clmax:
                cen_line[i] = int(clmax)
            if cen_line[i] < clmin:
                cen_line[i] = int(clmin)
    
        # cen_line is now cleaned, smoothed, and robust to fitting artifacts
        cen_line = np.sort(cen_line) # Doesn't change the number of cen_line pixels, just sorts outliers
        grapher.plot_cen_line(cen_line, target_dir, scinum)
        
#----------------------------------
#flux extraction
        
        # Fit parameters to each row, need new fit after cleaned cen_line
        args_list = [(row, cen_line[i], a_width, buff_width, bckrnd_width)
                 for i, row in enumerate(sci_final_1)]
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            sci1_fit = list(tqdm(executor.map(run_fit, args_list), total=len(args_list)))
        print("KARP finished fitting flux Gaussians")
        
        print("10 Flux Fit Parameters:")
        with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
            f.write("10 Flux Fit Parameters:"+"\n")
        for i in range(0,4000,400):
            a, mu, sigma, bck = sci1_fit[i]
            print("Y:",i," a:",a," mu:",mu," sig:",sigma," bck:",bck)
            with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                f.write("Y:"+str(i)+" a:"+str(a)+" mu:"+str(mu)+" sig:"+str(sigma)+" bck:"+str(bck)+"\n")
        
        # Plot Gaussian center values vs y pix and Gaussian sigma value vs Y pix
        print("Plotting Gauss cen values and Gauss sig values vs Y pix")
        grapher.gauss_cen_plots(sci1_fit, cen_line, target_dir, scinum)
        
        cen_line = np.round(cen_line).astype(int)
        # Make a plot of 20 evenly spaced pix
        # Showing the data, the apertures and buffers as lines,
        # And the over plotted G fit
       
        print("Making 20 aperture plots and aperature fit plots")
        grapher.make_aperature_fit_plots(sci1_fit, sci_final_1, cen_line, a_width, buff_width, bckrnd_width, target_dir, scinum)
        grapher.make_aperature_plots(sci1_fit, sci_final_1, cen_line, a_width, buff_width, bckrnd_width, target_dir, scinum)
        
        # Now set down an aparature and get the flux
        # in the center and another aperture to get the background level
        
        flux_raw1 = [] # Raw, unormalized or lam calibrated flux
        sky_raw1 = [] # Raw, unormalized or lam cal sky value for use later
        
        # Extract spectrum value from each y row
        print("Extracting flux...")
        for i in range(len(sci1_fit)): 
            a, mu, sigma, bck = sci1_fit[i]
            if np.isnan(a):
                continue
            c = cen_line[i]
            g_func = lambda x: functions.G(x,a,mu,sigma,bck) # quad() requires a function; lambda used to wrap G
            c_flux, _ = quad(g_func,c-a_width, c+a_width) # Center app flux
            bkg_right, _ = quad(g_func,(c-(a_width+buff_width+bckrnd_width)),(c-a_width))
            bkg_left, _ = quad(g_func,(c-(a_width+buff_width+bckrnd_width)),(c-a_width))
            # Append flux_raw with our flux for each row
            flux_raw1.append(c_flux-((a_width/(2*bckrnd_width))*(bkg_right+bkg_right)))    
            inrow = sci_final_1[i]
            sky_raw1.append(float(a_width/bckrnd_width)*(np.sum(inrow[(c-a_width-buff_width-bckrnd_width):(c-a_width-buff_width)])+np.sum(inrow[(c+a_width+buff_width+bckrnd_width):(c+a_width+buff_width)]))) # Sum up both sides of the background
            # Recall that we need to scale for the amount of "Sky" that is technically within our aperture size
        
        # Plot output raw flux (We can remove CRs later)
        grapher.plot_raw_flux(flux_raw1, target_dir, scinum)
        
#--------------------------------------
#wavelength calibration

        pixc = [250,295,765,925,953,1275,1330,1396,1480,2870,3355,3421,3612,3780]
        lam = [6416.31,6384.72,6032.13,5912.09,5888.58,5650.70,5606.73,5558.70,5495.87,4510.73,4200.67,4158.59,4044.42,3948.97]
        
        argon = np.asarray(CCDData.read(format_fits_filename(dloc, argnum), unit="adu").data)
        
        # Lambda calibration
        print("Fitting lambda Gaussians...")
        
        lam_width = 7 # Width to fit wavelength gaussians
        lam_fit = [] # Stores fit parameters as [a, mu, sig]
        lam_fit_linenum = [] # Succesfully fitted line wavelength
        lam_fit_pixc = [] # Succesfully fitted line pixc
        
        
        for i in tqdm(range(len(pixc))):
            # We want the flux values around pixc
            a_col = argon[:, cen_line[i]] # Get every column value that is in cen_line
            flux_vals = a_col[(pixc[i]-lam_width) : (pixc[i]+lam_width)] # Get argon values from the same pixels
            # We want locations along the y
            loc_vals = np.arange((pixc[i]-lam_width),(pixc[i]+lam_width)) # fit to y pixel values of 5 of the center of the argon wavelength
           
            p0 = [np.max(flux_vals), pixc[i], 2.0, 0] # Guess that the gussian peak is the max height
            bounds = [(0,0,1,-100),(np.inf,np.inf,np.inf,np.inf)] # a, mu, sig, bck
            try:
                # Curve fit takes, x (locations), y (values)
                popt, _ = curve_fit(functions.G, loc_vals, flux_vals, p0=p0, bounds=bounds, maxfev=1000) # ignore covariance matrix spat out from curve_fit
                lam_fit.append(popt)  # [a, mu, sigma, c] returend from curve_fit
                lam_fit_linenum.append(lam[i]) # Wavelength values that where succesfully fit
                lam_fit_pixc.append(pixc[i]) # Pixc of successfully fit wavelength values
            except RuntimeError:
                print("Curve fit failed at maxfev=1000")
                print("Trying to fit:",lam[i],pixc[i])
                print("p0 is:",p0)
                print("Now trying with maxfev=10000")
                lam_width = 11
                print("And lam_width:",lam_width)
                a_col = argon[:, cen_line[i]] # Get every column value that is in cen_line
                flux_vals = a_col[(pixc[i]-lam_width) : (pixc[i]+lam_width)] # Get argon values from the same pixels
                # We want locations along the y
                loc_vals = np.arange((pixc[i]-lam_width),(pixc[i]+lam_width)) # fit to y pixel values of 5 of the center of the argon wavelength
               
                p0 = [np.max(flux_vals), pixc[i], 2.0, 0] # Guess that the gussian peak is the max height
                bounds = [(0,0,1,-100),(np.inf,np.inf,np.inf,np.inf)] # a, mu, sig, bck
                # Curve fit takes, x (locations), y (values)
                print("NEW p0 is:",p0)
                popt, _ = curve_fit(functions.G, loc_vals, flux_vals, p0=p0, bounds=bounds, maxfev=10000) # ignore covariance matrix spat out from curve_fit
                lam_fit.append(popt)  # [a, mu, sigma, c] returend from curve_fit
                lam_fit_linenum.append(lam[i]) # Wavelength values that where succesfully fit
                lam_fit_pixc.append(pixc[i]) # Pixc of successfully fit wavelength values
        
        print("KARP fitted ", len(lam_fit), " lines")
        with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
            f.write("KARP fitted"+str(len(lam_fit))+"lines\n")
        cfits = [] # Empty array for putting the fitted line centers
        for i in range(len(lam_fit)): # For each line KARP was able to fit, get the center of the Gaussian from the fit    
            cfits.append(lam_fit[i][1])
            
        print("Argon Lines Fit Parameters:")
        with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
            f.write("Argon Lines Fit Parameters:\n")
    
        s_lsp = [] # Sigmas of the lam line fits for the line split function
        for i in range(len(lam_fit)):
            a, mu, sigma, bck = lam_fit[i]
            s_lsp.append(sigma)
            print("Lam Line Number:",i," lam:",lam[i]," a:",a," mu:",mu," sig:",sigma," bck:",bck)
            with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                f.write("Lam Line Number:"+str(i)+" lam:"+str(lam[i])+" a:"+str(a)+" mu:"+str(mu)+" sig:"+str(sigma)+" bck:"+str(bck)+"\n")
        #for i in range(len(lam_fit_linenum)):
        #   print("Lam:",lam_fit_linenum[i],"pixc:",lam_fit_pixc[i],"ycen actual fit to argon lines:",cfits[i])
        
        print("Line Split Function:",np.mean(s_lsp))
        with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
            f.write("Line Split Function:"+str(np.mean(s_lsp))+"\n")
        print("(avg of sig of line fits)")
        with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
            f.write("(avg of sig of line fits)\n")    
        
        # Fit a cubic poly nomial with cfits (actual line centers)
        # Fitting c fits (actual line centers in pixels) to lam_fit_linenum (successfully fit line wavelength values from lam)
        print("Fitting cubic polynomial")
        cfits_red = []
        for i in range(len(cfits)):
            cfits_red.append(cfits[i]/2000)
        a,b,c,d = np.polyfit(cfits_red,lam_fit_linenum,3)    
        #print(a,b,c,d) # As a*x^3+bx^2+c*x+d = lam(y)
        flux_raw1 = np.array(flux_raw1)
        ii = flux_raw1 > 0
        flux_raw_cor1 = flux_raw1[ii]
        y_pix = np.arange(0,len(flux_raw_cor1),1)
        
        v_helio = sci_tools.heliocentric_correction(objRA, objDEC, otime)
    
        print("v_helio:",v_helio,"km/s") # -4 km/s for G093440
        
        grapher.plot_sci_lamcal(y_lam, y_pix, flux_raw_cor1, target_dir, scinum)
        
        print("Calculating RMS A")
    
        # Lambda calibration graph
        del_lam = [] # Lam_fit - Lam_cal
        print("len(cfits)",len(cfits))
        print("len(lam):",len(lam))
        for i in range(len(lam)):
            # cfits is the center fit value in pixels
            print(cfits[i])
            if not np.isnan(cfits[i]):
                del_lam.append(float(lam[i])-float(y_lam(cfits[i]))) # The wavelength value we want to fit the line to MINUS the value our fit says that that line is at
            else:
                print("NAN here")
            #print(offset[i])
            
        # Put print RMS here
        dlS = []
        for i in range(len(del_lam)):
            dlS.append(del_lam[i]**2)
        print("RMS A:",np.sqrt(np.mean(dlS)))
        with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
            f.write("RMS A:"+str(np.sqrt(np.mean(dlS)))+"\n")
        print("Calculating RMS km/s")
        dlols = [] # Delta lambda over lambda
        for i in range(len(del_lam)):
            dlols.append(float(del_lam[i]/lam[i])**2)
        
        print("RMS km/s:",np.sqrt(np.mean(dlols))*3*10**5)
        with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
            f.write("RMS km/s:"+str(np.sqrt(np.mean(dlols))*3*10**5)+"\n")
        # Should be ~10 km/s 0.1 A or even 0.05 A
    
        # Print RMS in A, and in velocity
        # and plot counts vs error
        rnErr = []
        res_vel = [] # Residuals in velocity space
        for i in range(len(lam)):
            print("Angstrom:",lam[i]," sqrt(flux+sky):",(float(flux_raw_cor1[i]+sky_raw1[i])**(1/2))) # Print the RMS error
            print("km/s:",(del_lam[i]/lam[i])*3*10**5," sqrt(flux+sky):",(float(flux_raw_cor1[i]+sky_raw1[i])**(1/2))) # Print the RMS error
            res_vel.append((del_lam[i]/lam[i])*3*10**5)
        
        print("Plotting Residuals...")
        
        for i in range(len(flux_raw_cor1)): # For our corrected fluxes
            rnErr.append(float(flux_raw_cor1[i]+sky_raw1[i])**(1/2))
    
        #make lamcalres plots
        grapher.lam_cal_res(pixc, del_lam, target_dir, scinum)
        grapher.lam_cal_res_vel(pixc, res_vel, target_dir, scinum)
    
        print("Mean Wavelength residual (delLAM):",np.mean(del_lam))
        print("Mean Velocity residual (delVEL):",np.mean(res_vel))
        with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
            f.write("Mean Wavelength residual (delLAM):"+str(np.mean(del_lam))+"\n")
            f.write("Mean Velocity residual (delVEL):"+str(np.mean(res_vel))+"\n")
        
        grapher.plot_sci_lamcal_res(y_lam, y_pix, flux_raw_cor1, rnErr, target_dir, scinum)
        
        # Print dispersion, as (number of angstroms we go over/number of pixels (4096))
        # Recall that our lam is fit inverse of our pixels, ie 246 pix = 6000 Angstroms
        lam_out_max = y_lam(np.min(y_pix)) # Max wavelength as a function of pixel, ie the wavelength fit to the last pixel
        lam_out_min = y_lam(np.max(y_pix)) # Minimum wavelength as a function of pixel, ie the wavelength fit to the first pixel
        print("lam_out_max:",lam_out_max)
        print("lam_out_min:",lam_out_min)
        print("Dispersion: ",(lam_out_max-lam_out_min)/len(y_pix), " Angstroms/pix")
        dispersion = (lam_out_max-lam_out_min)/len(y_pix)
        
        with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
            f.write("lam_out_max:"+str(lam_out_max)+"\n")
            f.write("lam_out_min:"+str(lam_out_min)+"\n")
            f.write("Dispersion:"+str((lam_out_max-lam_out_min)/len(y_pix))+"Angstroms/pix\n")
            
        # Store flux_raw_cor1
        with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"flux_raw_cor1_"+str(scinum)+".txt", "w") as file:
            for val in flux_raw_cor1:
                file.write(f"{val}\n")
        
        # Store y_lam(y_pix)=wavelengths
        # Heliocentric correction here as well (Should be carried over from y_lam)
        wavelengths = y_lam(y_pix) # Wavelength of each pixel
        with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"wavelengths_"+str(scinum)+".txt", "w") as file:
            for wave in wavelengths:
                file.write(f"{wave}\n")
    
        # Store raw sky values
        with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"skyraw_"+str(scinum)+".txt", "w") as file:
            for sky in sky_raw1:
                file.write(f"{sky}\n")
    
    """
    ----------------------
    Begin skip_red section
    ----------------------
    """
    
    if comb_spec == False:
        # Mask out deep absorption features
        # Deep features in order of ascending wavelength, in angstroms
        # We have spectra down to Lam = 3780 and up to 6600 angstroms
        # H_8,H_eta, H_zeta,Ca_K,Ca_H, H_epsilon,H_delta,H_gamma, H_beta, SodiumD2, SodiumD1, H_alpha
        features = [3796.94,3835.40,3889.06,3933.7,3969,3970.075,4101.73,4340.47,4861.35,5889.96,5895.93,6562.81]
        if skip_red == True:
            flux_raw_cor1a = open(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"flux_raw_cor1_"+str(scinum)+".txt") # read in raw flux
            flux_raw_cor1a1 = flux_raw_cor1a.read()
            flux_raw_cor1b = flux_raw_cor1a1.splitlines()
            flux_raw_cor1a.close()
            
            wavelengths1a = open(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"wavelengths_"+str(scinum)+".txt") # read in corresponding wavelengths
            wavelengths1a1 = wavelengths1a.read()
            wavelengthsb = wavelengths1a1.splitlines()
            wavelengths1a.close()
            
            sky1a = open(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"wavelengths_"+str(scinum)+".txt") # read in corresponding wavelengths
            sky1a1 = sky1a.read()
            skyb = sky1a1.splitlines()
            sky1a.close()
            
            # Empty values
            flux_raw_cor1 = []
            wavelengths = []
            # Remake empty values with extracted flux and wavelength
            flux_raw_cor1 = [float(val) for val in flux_raw_cor1b]
            wavelengths = [float(val) for val in wavelengthsb]
            sky_raw1 = [float(val) for val in skyb]
        
        # Remove 4 pixels from either edge to ensure bad chip pixels aren't affecting our fits
        flux_raw_cor1 = flux_raw_cor1[3:-4]
        wavelengths = wavelengths[3:-4]
        
        
        flux_masked = np.array(flux_raw_cor1.copy()) # Create a copy so we can still use flux_raw_cor1
        
        print("Estimating continuum")
        # Make each line have its own feature mask width to get a good mask
        # H_8,H_eta, H_zeta,Ca_K,Ca_H, H_epsilon,H_delta,H_gama, H_beta, SodiumD2, SodiumD1, H_alpha
        
        # Wide features, for bright spectra
    
        feature_mask = [10,15,19,10,10,35,40,60,60,10,15,30]
        # Thin features, for really dim stars
        #feature_mask = [20,20,20,10,10,35,30,40,80,10,15,30]
    
        
        for i in range(len(features)):
            # Find pixel index closest to the feature wavelength
            wavelengths = np.array(wavelengths)
            # use np.argmin to get the closest pixel value to our feature
            center = np.argmin(np.abs(wavelengths - features[i])) # argmin so we round to the closest indexed value, not skipping over wavelengths that we have pixels for
            # Get the maximum and minimum values (up to 40 pix away from center)
            start = max(0, center - feature_mask[i]) # Not 40 pixels, 40*1/*0.7 Ang/pix so like 57
            stop = min(len(flux_masked), center + feature_mask[i])
            flux_masked[start:stop] = np.nan
        
        
        # Fit a fourth degree polynomial to the continuum estimate
        flux_mask_red = []
        lam_red = []
        flux_mask_remove = []
        lam_remove = []
        removed_index = [] # 0 is keep and 1 is remove
        # reduce the flux for a better fit
        for i in range(len(flux_masked)):
            if np.isnan(flux_masked[i]) != True:
                # Pixels we want to keep
                flux_mask_red.append(flux_masked[i])
                lam_red.append(wavelengths[i])
                removed_index.append(0) # The index values of the pixels that we keep for fitting later
            if np.isnan(flux_masked[i]) == True:
                # Pixels that we take out
                flux_mask_remove.append(flux_raw_cor1[i])
                removed_index.append(1) # The index values of the pixels that we remove for fitting later
                lam_remove.append(wavelengths[i])
        
        # Fit a continuum to out spectra:
        fit_params = np.polyfit(lam_red,flux_mask_red,6)    
        #print(aa,bb,cc,dd,ee) # As a*x^4+bx^3+c*x^2+d*x+e = flux_masked
        
        con_lam_test = []
        for i in range(len(wavelengths)):
            con_lam_test.append(functions.con_lam(wavelengths[i], fit_params))
            
        grapher.plot_sci_flux_masked(lam_red, lam_remove, flux_mask_red, flux_mask_remove, wavelengths, con_lam_test, target_dir, scinum)
        
        Fnorm_Es = [] # Estimated normalized flux
        for i in range(len(flux_raw_cor1)):
            Fnorm_Es.append(float(flux_raw_cor1[i]/(functions.con_lam(wavelengths[i], fit_params)))) # Divide raw flux by our continuum that we just fit
        
        # Plot our first estimate on the normalized flux
        grapher.plot_sci_flux_norm_est(wavelengths, Fnorm_Es, target_dir, scinum)
        
        print("Line-fit normilization")
        wavelengths = np.array(wavelengths)
        flux_raw_cor1 = np.array(flux_raw_cor1)
        Fnorm_Es = np.array(Fnorm_Es)
        removed_index = np.array(removed_index)
        
        fitted_values = []
        fitted_fluxes = []
        fitted_wavelengths = []
        fitted_sky = []
        
        flux_masked_global = flux_raw_cor1.copy()
        for i, feat in enumerate(features):
            center = np.argmin(np.abs(wavelengths - feat)) # Use argmin to get the index of the minimum value of all_wavelengths-all_features, (i.e. where the center of each feature is)
            start = max(0, center - feature_mask[i]) # From the center out to our feature mask size, start as far out from the edge as we can
            stop = min(len(flux_masked_global), center + feature_mask[i]) # From the center out to our feature mask size,  start as close to the edge as we can
            flux_masked_global[start:stop] = np.nan # Mask absorption lines with NANs
        
        lam_reda = [] # Kept flux values that do not have Fnorm_es > 1.3
        flux_mask_reda = []
        lam_removea = [] # Removed flux values that do not have Fnorm_es > 1.3
        flux_mask_removea = []
        
        # Precompute NaN masks for spectral features
        for i in range(len(wavelengths)):
            #if Fnorm_Es[i] >= 1.3:
            #    print("Removed one",Fnorm_Es[i])
            rm_param = 1.3
            
            if Fnorm_Es[i] <= rm_param:  # Skip things that are brighter than 1.3 norm flux
                lam_left = max(0, i - norm_line_width)
                lam_right = min(len(wavelengths), i + norm_line_width)
                wave_range = wavelengths[lam_left:lam_right] # Get wavelengths and fluxes around where we're trying to fit
                flux_values = flux_raw_cor1[lam_left:lam_right]
                
                if removed_index[i] == 0: # If this wavelength is not in an absoprtion feature
                    # If our norm_line_width spills over into lines on either side we want to ignore them
                    wave_range_cor = []
                    flux_values_cor = []
                    array = np.arange(lam_left, lam_right, 1)
                    for k in range(len(array)):
                        if removed_index[array[k]] == 0:  # Check if there are any absorption features within the region we want to be fitting
                        # (Recall that removed_inxed=0 corresponds to values we want to keep for fitting)
                        # if there is not an absorption feature at that wavelength then keep the flux value for fitting
                            wave_range_cor.append(wavelengths[array[k]])
                            flux_values_cor.append(flux_values[array[k] - lam_left])  # Align flux_values index, as flux starts at 0
                            
                    if len(wave_range_cor) >= 2: # Check if we have enough points to fit
                        # Use poly fit to fit a line to the corrected flux values as a funtion of wavelength
                        ml, bl = np.polyfit(wave_range_cor, flux_values_cor, 1)
                        # Now get the fit value of the continum at that wavelength
                        fitted_values.append(ml * wavelengths[i] + bl)                        
                    else:
                        continue  # Skip if not enough good values to fit
                        
        
                elif removed_index[i] != 0: # Wavelength is inside of an absorption feature
                    lam_left = max(0, i - norm_line_width)
                    lam_right = min(len(wavelengths), i + norm_line_width)
        
                    wave_range = wavelengths[lam_left:lam_right] # Get wavelengths, fluxes, and the fluxes we want to mask
                    flux_values = flux_raw_cor1[lam_left:lam_right]
                    flux_values_masked = flux_masked_global[lam_left:lam_right] # This is flagged with NANs
        
                    valid = ~np.isnan(flux_values_masked) # Take only values from not within an absorption feature
                    if np.sum(valid) < 2: # Need at least two points to fit a line
                        continue
                    ml, bl = np.polyfit(wave_range[valid], flux_values_masked[valid], 1)
                    fitted_values.append(ml * wavelengths[i] + bl)
                
                # Append fit results for global use
                
                fitted_fluxes.append(flux_raw_cor1[i])
                fitted_wavelengths.append(wavelengths[i])
                fitted_sky.append(sky_raw1[i])
                
                if np.isnan(flux_masked[i]) != True:
                    # Pixels we want to keep
                    flux_mask_reda.append(flux_masked[i])
                    lam_reda.append(wavelengths[i])
                if np.isnan(flux_masked[i]) == True:
                    # Pixels that we take out
                    flux_mask_removea.append(flux_raw_cor1[i])
                    lam_removea.append(wavelengths[i])
    
        # Now smooth this fit with a moving boxcar
        smooth_cont = uniform_filter(np.array(fitted_values), size=norm_line_boxcar)
        
        # Plot fitted flux values from the line fit to see what is going on
        grapher.plot_fit_over_flux(fitted_wavelengths, smooth_cont, fitted_values, fitted_fluxes, lam_removea, flux_mask_removea, lam_red, lam_remove, flux_mask_red, flux_mask_remove, lam_reda, flux_mask_reda, target_dir, scinum)
        
        Fnorm = [] # Normalized flux
        for i in range(len(fitted_fluxes)):
            Fnorm.append(float(fitted_fluxes[i]/smooth_cont[i])) # Divide raw flux by our continuum that we just fit
        
        for i in range(len(Fnorm)):
            if Fnorm[i] >= 1.3:
                print("Found Fnorm >= 1.3:", Fnorm[i])
                Fnorm[i] = 1
        
        print("Plotting Normalized Spectra")
        # Plot our finalized normalized flux
        
        sf_sm = [] # sqrt(sky+flux)/smooth_fit
        for i in range(len(fitted_fluxes)):
            sf_sm.append(float(fitted_fluxes[i]+fitted_sky[i])**(1/2)/smooth_cont[i]) 
        
        grapher.plot_sci_normalized(fitted_wavelengths, Fnorm, sf_sm, target_dir, objid, scinum)
        
        # Write a file for the spectra of this sci image
        # In the form G123456_61_OUT.fits in the OUT folder
        t = Table([fitted_wavelengths,Fnorm,sf_sm], names=('Lam', "Flux", "Sigma"))
        t.write(str(target_dir)+"OUT/"+str(objid)+"_"+str(scinum)+"_OUT.fits", format='fits', overwrite=True)
    
    #---------------------------------
    
    # If comb_spec = True, then take a number of input tables labled as
    # G123456_61_OUT.txt, G123456_62_OUT.txt, G123456_63_OUT.txt
    # for normalized spectra for science images 61, 62,  and 63 for example
    if comb_spec == True:
        print("KARP is combining spectra")
        # Make a KARP_OUT_log file
        
        with open(str(target_dir)+"OUT/KARP_OUT_log.txt", "w") as fo:
            fo.write("KARP is combining spectra\n")
        # Use glob to get everything in the folder
        file_names = glob.glob(str(target_dir)+"OUT/*.fits")
        print("Using these file names:",file_names)
        with open(str(target_dir)+"OUT/KARP_OUT_log.txt", "a") as fo:
            fo.write("Using these file names:"+str(file_names)+"\n")
        
        # Use wavelength grid from first file
        wave_ref = Table.read(file_names[0])["Lam"]
    
        # Interpolate each spectrum onto wave_ref
        gflux_interp = []
        gsig = []
        for fname in file_names:
            t = Table.read(fname)
            f = interp1d(t["Lam"], t["Flux"], bounds_error=False, fill_value=np.nan)
            s = interp1d(t["Lam"], t["Sigma"], bounds_error=False, fill_value=np.nan)
            gflux_interp.append(f(wave_ref))
            gsig.append(s(wave_ref))
            
        # Stack and take median, ignoring NaNs
        gflux_array = np.array(gflux_interp)
        med_comb = np.nanmedian(gflux_array, axis=0)
        gwave = wave_ref
        gsig = np.array(gsig)
    
        # Remove zeroth order line and any remaning outliers from bad pixels
        for i in range(len(gwave)):
            if 5500 <= gwave[i] <= 5700:
                if med_comb[i] > 1.05:
                    med_comb[i] = 1
                    
        for i in range(len(med_comb)):
            if med_comb[i] > 1.25:
                med_comb[i] = 1
                    
        print("Computing RMS")
        # Compute the RMS across 5450-5550 of the median combined spectra
        one_minus_med = []
        for i in range(len(gwave)):
            if 5450 <= gwave[i] <= 5550 and not np.isnan(med_comb[i]):
                one_minus_med.append((1-med_comb[i])**2)
                #print(np.sqrt(1-med_flux[i]))
        # Calculate and print SNR at 5500A
        RMS = np.sqrt(1/(len(one_minus_med))*np.sum(one_minus_med))
        print("SNR at 5500:",1/RMS)
        with open(str(target_dir)+"OUT/KARP_OUT_log.txt", "a") as fo:
            fo.write("SNR at 5500:"+str(1/RMS)+"\n")
        
        # Calculate sig_final(lam), should be 1.2,1.3
        print("Calculating sig_final(lam)")
        
        # Avoid division by zero or NaNs
        inv_var_array = np.where(gsig > 0, 1.0 / gsig**2, 0)
    
        # Sum over spectra (axis=0 is spectra, axis=1 is wavelength, so we sum over 0)
        sum_sigw = np.sum(inv_var_array, axis=0)  # shape: (N_wavelength,)
        
        sig_w = []
        for i in range(len(sum_sigw)):
            sig_w.append(np.sqrt(1/sum_sigw[i]))
        
        
        # Get the closest sig_w to 5500 angstroms
        sig_w55_idx = np.argmin(np.abs(gwave - 5550))
        sig_w55 = sig_w[sig_w55_idx]
        
        # Compute sig_final
        sig_final = RMS * np.array(sig_w)/sig_w55
        
        print("(RMS/sig_w55)",(RMS/sig_w55))
        with open(str(target_dir)+"OUT/KARP_OUT_log.txt", "a") as fo:
            fo.write("(RMS/sig_w55)"+str(RMS/sig_w55)+"\n")
        
        # Over plot Max's spectra for G093440
        # Remember to comment out later
        
        max1 = open("G093440_Max.txt") # read in raw flux
        max1a = max1.read()
        max1aa = max1a.splitlines()
        fout = [] # Output flux
        wout = [] # Output wavelenght
        eout = []
        for i in range(1,(len(max1aa)),1):
            splitm = max1aa[i].split()
            wout.append(float(splitm[0])) # wavelenght
            fout.append(float(splitm[1])) # flux
            eout.append(float(splitm[2])) # error
        max1.close()
        
        grapher.plot_combined_norm(gwave, med_comb, sig_final, wout, fout, eout, RMS, target_dir, objid)
    
#--------------------------------------------------
        if (fit_lines == True):
            # Measure the radial velocity of the star based on these 7 absorption lines
            print("Measuring radial velocity of the star")
            # Line names: H theta, H eta, H zeta, H delta, H gamma, H alpha, Ca K, Don;t use: Hbeta, HeI(sdB) 4471.5, HeI 4173, HeI 4922, HeI 5016
            vel_lines = [3797.91,3835.40,3889.06,4101.73,4340.47,6562.79,3933.66] # In angstroms ,,4861.35,4471.5,4713,4922
            vel_mask = [5,5,5,5,5,5,10,10,10,10,10] # half-width of lines to fit in angstroms
            vel_fit = [] # The fit parameters for the velocity fits
            vel_fit_linenum = [] # Which of the seven lines we just fit
            
            
            wing_offset = []
            # Fit a gaussian to the core of an absorption feature
            for i in tqdm(range(len(vel_lines))):
                # We want the flux values around the line centers
               
                mask = (gwave > vel_lines[i] - vel_mask[i]) & (gwave < vel_lines[i] + vel_mask[i])
                flux_vals = med_comb[mask]
                wave_vals = gwave[mask]
                
                p0 = [-np.min(flux_vals), vel_lines[i], 2.0, 1] # Guess that the minimum flux is the peak of the fit
                # and that the offset is 1 (accounting for normilization), and that the center is the vel_lines value
                bounds = [(-np.inf,0,1,0.5),(0,np.inf,np.inf,1.1)] # a, mu, sig, bck
                try:
                    # Curve fit takes, x (locations), y (values)
                    popt, _ = curve_fit(functions.G, wave_vals, flux_vals, bounds=bounds, p0=p0, maxfev=5000) # ignore covariance matrix spat out from curve_fit
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
            grapher.plot_vel_lines(vel_fit, vel_fit_linenum, vel_lines, gwave, med_comb, wout, fout, vel_mask, target_dir, objid)
            
            # average the velocities of the 7 lines:
            radial_vel = np.mean(line_velocities_karp)
            radial_vel_err = np.std(line_velocities_karp)
            print("Radial velocity:",radial_vel,"+/-", radial_vel_err)
            #print(wing_offset)        
            # Look for HI line, 4471.5
        
#---------------------------------------------------
        
            #find equivalent widths of metal lines
            metal_lines = [3820.425, 3933.66, 4045.812, 4063.594, 4226.728, 4260.474, 4271.76, 4307.902, 4383.545, 4404.75, 4957.596, 5167.321, 5172.684, 5183.604, 5269.537, 5328.038]
            metal_mask = [4, 3, 4, 4, 3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4]
            metal_fit = []
            metal_fit_linenum = []
            metal_results = []
            
            #iterate over metal lines       
            for i, line in enumerate(metal_lines):
                ew_sum = 0
                		
                mask = (gwave > line - metal_mask[i]) & (gwave < line + metal_mask[i])
                flux_vals = med_comb[mask]
                wave_vals = gwave[mask]
                wave_err = sig_final[mask]
                x = np.array(wave_vals) 
                p0 = [-np.min(flux_vals), line, 2.0, 1.0]
                bounds = [(-np.inf,0,1, 0.9),(0,np.inf,np.inf, 1.1)]
                   
                #attempt fit
                try:
                    popt, pcov = curve_fit(functions.G, wave_vals, flux_vals, p0=p0, bounds=bounds, maxfev=5000)
                    metal_fit.append(popt)
                    metal_fit_linenum.append(i)
                except RuntimeError:
                    print("Curve fit failed")
                
                #define function used for fitting    
                line_fit = lambda x: functions.G(x, popt[0], popt[1], popt[2], popt[3])
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
            csv_path = str(target_dir) + f"OUT/{objid}_metal_lines.csv"
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "Line (A)", "EW_g", "EW_riemann", "Error", "Sys", "Adopted_EW", "Adopted_Error, Width of Mask"
                ])
                writer.writerows(metal_results)
            print(f"Metal line results written to {csv_path}")
        
            #make graphs of fits around lines
            grapher.metal_line_plt(metal_fit, gwave, mask, med_comb, metal_lines, target_dir)
            
            #make big plot of all metal lines
            grapher.metal_line_big_plt(metal_fit, metal_lines, metal_mask, gwave, med_comb, target_dir, objid)
            print(f"Big summary plot saved to {str(target_dir)}OUT/{objid}_metal_lines_all.png")
        
    #--------------------------------------------------------
    #finishing tasks
                
    # Duration_run is how long KARP took to run
    duration_run = timedelta(seconds=time.perf_counter()-starttime)
    with open(str(target_dir)+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
        f.write("KARP took:"+str(duration_run)+" to run\n")
        f.write("><(((º>")
    
    #play finishing sound
    sound.play()
    pygame.time.delay(int(duration * 1000))  # wait for playback to finish
    sound1.play()
    pygame.time.delay(int(duration * 1000))  # wait for playback to finish
    
    print("KARP took:",duration_run," to run")
    print("><(((º>")
