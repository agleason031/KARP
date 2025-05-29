#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
KARP

KOSMOS Astronomical Reduction Pipeline

Created by Chase L. Smith, Max Moe

><(((º>
"""

# Importing relevant packages
import numpy as np
import matplotlib.pyplot as plt
import argparse
import ccdproc
from astropy.nddata import CCDData
from astropy.stats import mad_std
from scipy.ndimage import uniform_filter
import warnings
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from scipy.integrate import quad
from datetime import timedelta
from astropy.table import Table
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from astropy import units as u
import glob
import time
import os
from scipy.interpolate import interp1d
import pygame


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

# initialize global variables
dloc = objid = objRA = objDEC = otime = scinum = argnum = fstart = fstop = bstart = bstop = appw = buffw = bckw = clmin = clmax = norm_line_width = norm_line_boxcar = target_dir = skip_red = comb_spec = None
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
def string_to_bool(string):
    # A comparison after converting the string to lower case,
    # ie, if string=False=false is not equal to true so it returns the boolean value of FALSE
    # For example: "False".lower() == "true" gives False
    return string.lower() == "true"
skip_red = string_to_bool(str(params["skip_red"]))
comb_spec = string_to_bool(str(params["comb_spec"]))

print("KARP has taken the bait! (Input file):")
print("------------")
print(params) # Print out parameters
print("------------")

# Target directory should be something like /d/ori1/csmit/metalpoor/RED_G093440
# Syntax of target directory is RED_G123456, with RED=reduced
# Test input: -data_loc /d/ori1/csmit/metalpoor/Chase_G093440/Chase_G093440/ -sci_im_num 61 -arg_im_num 64 -f_start 1 -f_stop 7 -b_start 9 -b_stop 17 -appw 13 -buffw 2 -bckw 7 -cline_min 479 -cline_max 492 -target_dir /d/ori1/csmit/metalpoor/RED_G093440/

print("skip_red:",skip_red)
if skip_red == False:
    print("KARP is doing reduction!")
elif skip_red == True:
    print("KARP is SKIPPING reduction!")

# initialize value so KARP runs empty values if skip_red=True is run by accident
flux_raw_cor1 = []
wavelengths = []
# Global variables
sim = scinum
fim = np.arange(fstart,fstop,1)
bim = np.arange(bstart,bstop,1)
sim = scinum
objID = objid # Object ID 

# Make a global "OUT" folder to keep finalized outputs
os.makedirs(str(target_dir)+"OUT", exist_ok=True) # Initializes a directory OUT files
# Make a output text file detaling relevant fit paramaters that KARP calculates
# Initalize directory for reduction plots for this science image
os.makedirs(str(target_dir)+"ImageNumber_"+str(sim), exist_ok=True) # Initializes a directory for E20 files
with open(str(target_dir)+"ImageNumber_"+str(sim)+"/KARP_log.txt", "w") as f:
    f.write("KARP Output Log\n")
    f.write("- - - ><(((º>\n")
    f.write("\n")

if skip_red == False:
    # Create lists of file paths for the Flat and Bias images
    
    print("KARP is reducing Science Image number:",sim)
    with open(str(target_dir)+"ImageNumber_"+str(sim)+"/KARP_log.txt", "a") as f:
        f.write("KARP is reducing Science Image number:"+str(sim)+"\n")
        f.write("KARP has taken the bait! (Input file):\n")
        f.write("------------\n")
        f.write(str(params))
        f.write("------------\n")
    
    # Make a list of image paths
    blist = [] # A list of file locations of the bias images
    
    # Check for image numbers > 10 and append accordingly
    for i in range(len(bim)):
        if float(bim[i]) < 10:
            blist.append(str(dloc)+"k.000"+str(bim[i])+".fits")
        if float(bim[i]) >= 10:
            blist.append(str(dloc)+"k.00"+str(bim[i])+".fits")
    
    flist = [] # A list of file locations of the flat images
    
    for i in range(len(fim)):
        if float(fim[i]) < 10:
            flist.append(str(dloc)+"k.000"+str(fim[i])+".fits")
        if float(fim[i]) >= 10:
            flist.append(str(dloc)+"k.00"+str(fim[i])+".fits")
    
    slocation = []
    
    
    if float(sim) < 10:
        slocation= str(dloc)+"k.000"+str(sim)+".fits"
    if float(sim) >= 10:
        slocation = str(dloc)+"k.00"+str(sim)+".fits"
    
    
    # Make a master bias from the input bias image
    print("Making Master Bias")
    masterbias = ccdproc.combine(blist, method='median', unit='adu',sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,mem_limit=350e6)
    
    # Display the image using matplotlib in log space
    plt.imshow(masterbias)
    plt.colorbar()
    plt.title('Master Bias Image')
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+objID+"MBias_"+str(sim)+".png")
    print("Master Bias Made")
    
    # Make a master flat from the flat images
    print("Making Master Flat")
    masterflat = ccdproc.combine(flist,method='median', unit='adu',sigma_clip=False,mem_limit=350e6)
        
    # Display the image using matplotlib in log space
    plt.imshow(masterflat)
    plt.colorbar()
    plt.title('Master Flat Image')
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+objID+"_MFlat_"+str(sim)+".png")
    print("Master Flat Made")
    
    # Subtract Master Bias from Master Flat
    masterflatDEbias = ccdproc.subtract_bias(masterflat,masterbias)
    plt.imshow(masterflatDEbias)
    plt.colorbar()
    plt.title('Master Flat sub Bias Image')
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+objID+"_MF_MB_"+str(sim)+".png")
    print("Master Flat-Bias Made")
    
    # Smooth master flatdebias:
    np_mfdb = np.asarray(masterflatDEbias) # Convert to a numpy array
    #print(np_mfdb[0][0]) # This is the lower left corner, 
    
    #lower right is x=1124
    #top y is 4096
    
    smooth_mf_mb = uniform_filter(np_mfdb, size=5)
    plt.imshow(smooth_mf_mb)
    plt.colorbar()
    plt.title('Smoothed MF_MB Image (5x5 Boxcar)')
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+objID+"_smoothed_MF_MB_"+str(sim)+".png")
    print("Smoothed MF_MB Made")
    
    
    # We can write this to a fits file as:
    #masterflatDEbias.write('np_mfdb.fits')
    
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
    
    # Plot final flat
    plt.imshow(final_flat)
    plt.colorbar()
    plt.title('Final Flat')
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+objID+"_final_flat.png")
    final_flat_write = CCDData(final_flat, unit="adu")
    final_flat_write.write(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"final_flat_"+str(sim)+".fits", overwrite = True)
    
    print("Final Flat Made")
    
    # Subtract off the bias from each of our science images
    # Note that we aren't stacking the sci images just yet,
    # we wait to do that at the end, so we need to do this
    # for each of our sci images
    
    sci_final = (ccdproc.subtract_bias(CCDData.read(slocation,format = 'fits', unit = "adu"),masterbias).data)/final_flat
    #plt.imshow(sci_final)
    plt.title('Final Science ' + str(scinum))
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+objID+"_final_sci" + str(scinum)+".png")
    print("Master Science Number: " + str(scinum) + " Made")
    sci_final_write = CCDData(sci_final, unit="adu")
    sci_final_write.write(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"sci_final_"+str(sim)+".fits", overwrite = True)
    
    
    # Read all three of our final sci images and convert to nparrays
    sci_final_1 = np.asarray(CCDData.read(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"sci_final_"+str(sim)+".fits", unit = "adu").data)
    
    # Convert from ADU to electrons
    sci_final_1 = sci_final_1*0.6
    
    # Shape of sci images is 4096, 1124
    a_width = int(appw) # Pixel aperture width
    buff_width = int(buffw) # Width of the buffer
    bckrnd_width = int(bckw) # Background width
    # Make a gaussian function to call later
    
    # Define a Gaussian
    def G(x, a, mu, sigma, bck):
        return (a * np.exp(-(x-mu)**2/(2*sigma**2))) + bck
        # A 4d Gaussian
    
    
    n_rows = sci_final_1.shape[0]  # This gets the number of rows
    print("n_rows:",n_rows)
    #cen_line1 = np.repeat(np.arange(474, 496), n_rows // 22 + 1)[:n_rows] # This gets the closes int center x value for each rows
    cen_line1 = np.round(np.linspace(int(clmin), int(clmax), n_rows)).astype(int)
    # Note that this repeats each integer 51 times, stretching it to 1124
    
    # Make cen_line a list for ease of use
    cen_line = []
    for i in range(len(cen_line1)):
        cen_line.append(int(cen_line1[i]))
    
    
    def fit_cent_gaussian(row, clr):
        clr = int(round(clr))  # ensure integer index
        x_vals = np.arange((clr-a_width),(clr+a_width)) # fit to x pixel values of 9 of the center
        y_vals = row[(clr-a_width) : (clr+a_width)] # Get y values from the same pixels
        y_vals_bck = row[(clr-a_width-buff_width-bckrnd_width) : (clr+a_width+buff_width+bckrnd_width)]
        
        amp_guess = np.max(y_vals)  # Peak height
        mu_guess = clr  # Guess our center line trace as the Gaus center
        sigma_guess = 2.0  # Guess a sigma value of 2
        bck_guess = np.median(y_vals_bck) # Guess background level is just the median of the values out to background app
        
        p0 = [amp_guess, mu_guess, sigma_guess, bck_guess]
        bounds = [(0,0,1,-100),(np.inf,np.inf,np.inf,np.inf)]
        try:
            popt, pcov = curve_fit(G, x_vals, y_vals, p0=p0, bounds=bounds, maxfev=600) # ignore covariance matrix spat out from curve_fit
            return popt  # [a, mu, sigma, bck] returned from curve_fit
        except RuntimeError:
            return [np.nan, np.nan, np.nan, np.nan]
    
    
    """
    print("Plotting YMaxRow")
    fig, axSC = plt.subplots(1, 1, figsize=(8,6))
    for i in range(len(cen_line)):
        axSC.scatter(cen_line[i],np.max(sci_final_1[i])) # Max electron value in the whole row
        axSC.set_xlabel("Center Line Pixel")
        axSC.set_ylabel("Max Y Pix in the row")
    
    plt.savefig("YMaxRow.png")
    
    fig, axSCa = plt.subplots(1, 1, figsize=(8,6))
    for i in range(len(cen_line)):
        rowin = sci_final_1[i]
        axSCa.scatter(cen_line[i],np.max(rowin[(cen_line[i]-a_width) : (cen_line[i]+a_width)])) # Max electron value in our aperture
        axSCa.set_xlabel("Center Line Pixel")
        axSCa.set_ylabel("Max Y Pix in the row")
        
    plt.savefig("YMaxRowInApp.png")
    """
    # Fit Gaussian to each row in sci_final_1 using current cen_line estimate
    print("KARP fitting centerline")
    
    # Fit and collect results in a list (each entry is a tuple: a, mu, sig, bck)
    cen_fit = [fit_cent_gaussian(row, cen_line[i]) for i, row in tqdm(enumerate(sci_final_1), total=len(cen_line))]
    
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
    
    # Apply smoothing to fill gaps in cen_line
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
    cen_line = np.sort(cen_line)
    # Doesn't change the number of cen_line pixels, just sorts outliers
    
    fig, axcs = plt.subplots(1, 1, figsize=(8,6))
    censcat = np.arange(0,len(cen_line),1)
    axcs.scatter(censcat,cen_line, s=3)
    axcs.set_xlabel("Y Pixel")
    axcs.set_ylabel("X Pixel")
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"cen_line_"+str(sim)+".png")
    print("len(cen_line)",len(cen_line))
    
    
    # Fit parameters to each row
    sci1_fit = []
    print("KARP fitting flux Gaussians")
    for i in tqdm(range(len(cen_line))):
        rowin = sci_final_1[i]
        sci1_fit.append(fit_cent_gaussian(rowin, cen_line[i]))
    
    #sci1_fit = np.array([fit_cent_gaussian(sci_final_1[i], cen_line[i]) for i in range(n_rows)])
    print("KARP finished fitting flux Gaussians")
    
    print("10 Flux Fit Parameters:")
    with open(str(target_dir)+"ImageNumber_"+str(sim)+"/KARP_log.txt", "a") as f:
        f.write("10 Flux Fit Parameters:"+"\n")
    for i in range(0,4000,400):
        a, mu, sigma, bck = sci1_fit[i]
        print("Y:",i," a:",a," mu:",mu," sig:",sigma," bck:",bck)
        with open(str(target_dir)+"ImageNumber_"+str(sim)+"/KARP_log.txt", "a") as f:
            f.write("Y:"+str(i)+" a:"+str(a)+" mu:"+str(mu)+" sig:"+str(sigma)+" bck:"+str(bck)+"\n")
    
    # Plot Gaussian center values vs y pix and Gaussian sigma value vs Y pix
    print("Plotting Gauss cen values and Gauss sig values vs Y pix")
    fig, axc = plt.subplots(1, 1, figsize=(8,6))
    for i in range(len(sci1_fit)):
        a, mu, sigma, bck = sci1_fit[i]
        if np.isnan(a):
            continue
        axc.scatter(i,mu, s=1, color = 'blue')
        censcat = np.arange(0,len(cen_line),1)
        axc.scatter(censcat,cen_line, s=3, color = 'black')
        axc.tick_params(labelsize=14)
        axc.set_xlabel("Y Pix", size=14)
        axc.set_ylabel("G fit mu", size=14)
    
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"Sci_Gcen_y_Cenline_"+str(sim)+".png")
    
    # Now for sigma
    fig, axs = plt.subplots(1, 1, figsize=(8,6))
    for i in range(len(sci1_fit)):
        a, mu, sigma, bck = sci1_fit[i]
        if np.isnan(a):
            continue
        axs.scatter(i,sigma, color = 'green')
        axs.tick_params(labelsize=14)
        axs.set_xlabel("Y Pix", size=14)
        axs.set_ylabel("G fit Sig", size=14)
    
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"Sci_Gsig_y_"+str(sim)+".png")

    # Make a zoomed in plot
    plt.cla()
    fig, axs = plt.subplots(1, 1, figsize=(8,6))
    for i in range(len(sci1_fit)):
        a, mu, sigma, bck = sci1_fit[i]
        if np.isnan(a):
            continue
        axs.scatter(i,sigma, color = 'green')
        axs.tick_params(labelsize=14)
        axs.set_xlabel("Y Pix", size=14)
        axs.set_ylabel("G fit Sig", size=14)
        axs.set_ylim(0,2)
    
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"Sci_Gsig_y_ZOOM"+str(sim)+".png")
    
    cen_line = np.round(cen_line).astype(int)
    # Make a plot of 20 evenly spaced pix
    # Showing the data, the apertures and buffers as lines,
    # And the over plotted G fit
    print("Making 20 aperture fit plots")
    for i in range(0,4000,200):
        plt.clf()
        a, mu, sigma, bck = sci1_fit[i]
        rowin = sci_final_1[i]
        x_vals = np.arange((cen_line[i]-a_width-buff_width-bckrnd_width),(cen_line[i]+a_width+buff_width+bckrnd_width+1))
        # Add an extra +1 to the plt.step width to account for the default plt.step parameters
        plt.step(x_vals,rowin[(cen_line[i]-a_width-buff_width-bckrnd_width):(cen_line[i]+a_width+buff_width+bckrnd_width+1)],where='mid',color="black")
        # Plot center line
        plt.axvline(cen_line[i],color="black",linestyle="--")
    
        x_plot_lin = np.linspace((cen_line[i]-a_width),(cen_line[i]+a_width))
        plt.plot(x_plot_lin,G(x_plot_lin,a,mu,sigma,bck), color="purple")
        plt.xlim((cen_line[i]-a_width),(cen_line[i]+a_width))
        plt.xlabel("X pixels")
        plt.ylabel("Counts (e-)")
        os.makedirs(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"E20Fits_"+str(sim), exist_ok=True) # Initializes a directory for E20 files
        plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"E20Fits_"+str(sim)+"/E20Fit"+str(i)+".png")
    
    
    print("Making 20 aperture plots")
    for i in range(0,4000,200):
        plt.clf()
        a, mu, sigma, bck = sci1_fit[i]
        rowin = sci_final_1[i]
        x_vals = np.arange((cen_line[i]-a_width-buff_width-bckrnd_width),(cen_line[i]+a_width+buff_width+bckrnd_width+1))
        # Add an extra +1 to the plt.step width to account for the default plt.step parameters
        plt.step(x_vals,rowin[(cen_line[i]-a_width-buff_width-bckrnd_width):(cen_line[i]+a_width+buff_width+bckrnd_width+1)],where='mid',color="black")
        # Plot center line and aw, bw, buffw
        plt.axvline(cen_line[i],color="black",linestyle="--")
        plt.axvline((cen_line[i]-a_width),color="blue",linestyle="-")
        plt.axvline((cen_line[i]+a_width),color="blue",linestyle="-")
        plt.axvline((cen_line[i]-a_width-buff_width),color="red",linestyle="-")
        plt.axvline((cen_line[i]+a_width+buff_width),color="red",linestyle="-")
        plt.axvline((cen_line[i]-a_width-buff_width-bckrnd_width),color="green",linestyle="-")
        plt.axvline((cen_line[i]+a_width+buff_width+bckrnd_width),color="green",linestyle="-")
        plt.axvline(cen_line[i],color="black",linestyle="--")
        x_plot_lin = np.linspace((cen_line[i]-a_width),(cen_line[i]+a_width))
        plt.plot(x_plot_lin,G(x_plot_lin,a,mu,sigma,bck), color="purple",alpha=0)
        # Nonexistent Gausian helps with centering 
        
        plt.xlim((cen_line[i]-a_width-buff_width-bckrnd_width-1),(cen_line[i]+a_width+buff_width+bckrnd_width+2))
        plt.xlabel("X pixels")
        plt.ylabel("Counts (e-)")
        os.makedirs(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"E20Data_"+str(sim), exist_ok=True) # Initalizes a dircetory for E20 files
        plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"E20Data_"+str(sim)+"/EData"+str(i)+".png")
    
    
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
        g_func = lambda x: G(x,a,mu,sigma,bck) # quad() requires a function; lambda used to wrap G
        c_flux, _ = quad(g_func,c-a_width, c+a_width) # Center app flux
        bkg_right, _ = quad(g_func,(c-(a_width+buff_width+bckrnd_width)),(c-a_width))
        bkg_left, _ = quad(g_func,(c-(a_width+buff_width+bckrnd_width)),(c-a_width))
        # Append flux_raw with our flux for each row
        flux_raw1.append(c_flux-((a_width/(2*bckrnd_width))*(bkg_right+bkg_right)))    
        inrow = sci_final_1[i]
        sky_raw1.append(float(a_width/bckrnd_width)*(np.sum(inrow[(c-a_width-buff_width-bckrnd_width):(c-a_width-buff_width)])+np.sum(inrow[(c+a_width+buff_width+bckrnd_width):(c+a_width+buff_width)]))) # Sum up both sides of the background
        # Recall that we need to scale for the amount of "Sky" that is technically within our aperture size
    
    # Plot output raw flux (We can remove CRs later)
    plt.cla()
    y_pix = np.arange(0,len(flux_raw1),1)
    plt.scatter(y_pix,flux_raw1, s=1)
    plt.xlabel("Y Pixel")
    plt.ylabel("Background subtracted Flux")
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"Sci_Flux_bsub_"+str(sim)+".png", dpi=300)
    #plt.show()
    
    # Now we need to calibrate our flux to wavelength
    pixc = [250,295,765,925,953,1275,1330,1396,1480,2870,3355,3421,3612,3780]
    lam = [6416.31,6384.72,6032.13,5912.09,5888.58,5650.70,5606.73,5558.70,5495.87,4510.73,4200.67,4158.59,4044.42,3948.97]
    
    # Get argon spectra
    arg_num = int(argnum)
    
    if arg_num > 9:
        argon = np.asarray(CCDData.read(str(dloc)+"k.00"+str(arg_num)+".fits", unit = "adu").data)
    else:
        argon = np.asarray(CCDData.read(str(dloc)+"k.0"+str(arg_num)+".fits", unit = "adu").data)
    
    
    # Lambda calibration
    print("Fitting lambda Gaussians...")
    
    lam_width = 7 # Width to fit wavelength gaussians
    lam_fit = [] # Stores fit parameters as [a, mu, sig]
    lam_fit_linenum = [] # Succesfully fitted line wavelength
    lam_fit_pixc = [] # Succesfully fitted line pixc
    
    # We only need len(pixc) argon center line values
    
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
            popt, _ = curve_fit(G, loc_vals, flux_vals, p0=p0, bounds=bounds, maxfev=1000) # ignore covariance matrix spat out from curve_fit
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
            popt, _ = curve_fit(G, loc_vals, flux_vals, p0=p0, bounds=bounds, maxfev=10000) # ignore covariance matrix spat out from curve_fit
            lam_fit.append(popt)  # [a, mu, sigma, c] returend from curve_fit
            lam_fit_linenum.append(lam[i]) # Wavelength values that where succesfully fit
            lam_fit_pixc.append(pixc[i]) # Pixc of successfully fit wavelength values
    
    
    print("KARP fitted ", len(lam_fit), " lines")
    with open(str(target_dir)+"ImageNumber_"+str(sim)+"/KARP_log.txt", "a") as f:
        f.write("KARP fitted"+str(len(lam_fit))+"lines\n")
    cfits = [] # Empty array for putting the fitted line centers
    for i in range(len(lam_fit)): # For each line KARP was able to fit, get the center of the Gaussian from the fit    
        cfits.append(lam_fit[i][1])
        
    print("Argon Lines Fit Parameters:")
    with open(str(target_dir)+"ImageNumber_"+str(sim)+"/KARP_log.txt", "a") as f:
        f.write("Argon Lines Fit Parameters:\n")

    s_lsp = [] # Sigmas of the lam line fits for the line split function
    for i in range(len(lam_fit)):
        a, mu, sigma, bck = lam_fit[i]
        s_lsp.append(sigma)
        print("Lam Line Number:",i," lam:",lam[i]," a:",a," mu:",mu," sig:",sigma," bck:",bck)
        with open(str(target_dir)+"ImageNumber_"+str(sim)+"/KARP_log.txt", "a") as f:
            f.write("Lam Line Number:"+str(i)+" lam:"+str(lam[i])+" a:"+str(a)+" mu:"+str(mu)+" sig:"+str(sigma)+" bck:"+str(bck)+"\n")
    #for i in range(len(lam_fit_linenum)):
    #   print("Lam:",lam_fit_linenum[i],"pixc:",lam_fit_pixc[i],"ycen actual fit to argon lines:",cfits[i])
    
    print("Line Split Function:",np.mean(s_lsp))
    with open(str(target_dir)+"ImageNumber_"+str(sim)+"/KARP_log.txt", "a") as f:
        f.write("Line Split Function:"+str(np.mean(s_lsp))+"\n")
    print("(avg of sig of line fits)")
    with open(str(target_dir)+"ImageNumber_"+str(sim)+"/KARP_log.txt", "a") as f:
        f.write("(avg of sig of line fits)\n")
    
    """
    # Get argon flux if desired
    n_rows_ar = argon.shape[0]  # This gets the number of rows
    
    # Fit parameters to each row
    argon_fit = np.array([fit_cent_gaussian(argon[i], cen_line[i]) for i in range(n_rows_ar)])
    
    x_plot = np.linspace(474-9,495+9,200) # X linspace values to plot with
    
    cmap = cm.get_cmap("coolwarm") # Use cool warm colro map
    norm = Normalize(vmin=0, vmax=len(sci1_fit)-1) # Normalize our row count to 0-1 for color plotting
    
    # Now set down an aparature and get the flux from argon
    # in the center and another aperture to get the background level
    
    a_width = 9 # 9 Pixel aperture width
    buff_width = 2 # Width of the buffer
    bckrnd_width = 7 # Background width
    
    argon_flux_raw1 = [] # Raw, unormalized or lam calibrated flux
    
    # Extract spectrum value from each y row
    for i in range(len(argon_fit)): 
        a, mu, sigma, bck = argon_fit[i]
        if np.isnan(a):
            continue
        c = cen_line[i]
        g_func = lambda x: G(x,a,mu,sigma,c) # Quad wont run G normally
        c_flux, _ = quad(g_func,c-a_width, c+a_width) # Center app flux
        bkg_right, _ = quad(g_func,(c-(a_width+buff_width+bckrnd_width)),(c-a_width))
        bkg_left, _ = quad(g_func,(c-(a_width+buff_width+bckrnd_width)),(c-a_width))
        # Append flux_raw with our flux for each row
        argon_flux_raw1.append(c_flux-((a_width/(2*bckrnd_width))*(bkg_right+bkg_right)))    
    
    
    
    
    # Plot output argon raw flux if desired
    y_pix = np.arange(0,len(argon_flux_raw1),1)
    fig, ax1 = plt.subplots(1, 1, figsize=(8,6))
    ax1.scatter(y_pix,argon_flux_raw1, s=1)
    for i in range(len(cfits)):
        ax1.axvline(x=cfits[i])
    plt.xlabel("Y Pixel")
    plt.ylabel("Background subtracted Flux")
    plt.savefig("Sci1_Flux_ARlines.png", dpi=300)
    #plt.show()
    """
    
    
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
    print("heliocorr",heliocorr)
    # Map to v_helio
    correction = heliocorr.to(u.km/u.s)
    print("correction",correction)
    correction = str(correction)
    correction = correction.replace('km / s', '')
    v_helio = float(correction)
    """
    MFG Origonal Version
    site = EarthLocation.of_site(siteName)
    t = Time(time) #EX: t = Time("2015-06-30 23:59:60.500")
        sc = SkyCoord(coord, unit=(u.hourangle, u.deg))
    heliocorr = sc.radial_velocity_correction('heliocentric', obstime=t, location=site)
    correction = heliocorr.to(u.km/u.s)
    correction = str(correction) #NEEDED TO SPIT OUT USABLE NUMBER
    correction = correction.replace('km / s', '') #NEEDED TO SPIT OUT USABLE NUMBER
    correction = float(correction) #NEEDED TO SPIT OUT USABLE NUMBER
    return correction
    """

    print("v_helio:",v_helio,"km/s") # -4 km/s for G093440
    
    def y_lam(y):
        # Vectorize array
        y = np.array(y)
        # Wavelength as a function of pixel with helocentric velocity correction
        return ((a*(y/2000)**3)+(b*(y/2000)**2)+c*(y/2000)+d)*(1+(v_helio/(3*10**5)))
    
    
    #for i in range(len(y_pix)):
    #   print("High y_pix:",y_pix[i])
    #   print("High y_lam(y_pix[i]):",y_lam(y_pix[i]))
    
    
    
    fig, ax2 = plt.subplots(1, 1, figsize=(8,6))
    ax2.scatter(y_lam(y_pix),flux_raw_cor1, s=1)
    ax2.axvline(x=6562.81,color='red')
    ax2.axvline(x=4861.35,color='cyan')
    ax2.axvline(x=4340.47,color='blueviolet')
    plt.xlabel("Wavelength (A)")
    plt.ylabel("Background subtracted Flux (Electrons)")
    plt.ylim(0,30000)
    plt.title("Sci Image"+str(sim))
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"Sci_Flux_LamCal_"+str(sim)+".png", dpi=300)
    #plt.show()
    
    print("Calculating RMS A")

    # Lambda calibration graph
    fig, axL = plt.subplots(1, 1, figsize=(8,6))
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
    with open(str(target_dir)+"ImageNumber_"+str(sim)+"/KARP_log.txt", "a") as f:
        f.write("RMS A:"+str(np.sqrt(np.mean(dlS)))+"\n")
    print("Calculating RMS km/s")
    dlols = [] # Delta lambda over lambda
    for i in range(len(del_lam)):
        dlols.append(float(del_lam[i]/lam[i])**2)
    
    
    print("RMS km/s:",np.sqrt(np.mean(dlols))*3*10**5)
    with open(str(target_dir)+"ImageNumber_"+str(sim)+"/KARP_log.txt", "a") as f:
        f.write("RMS km/s:"+str(np.sqrt(np.mean(dlols))*3*10**5)+"\n")
    
    # Should be ~10 km/s 0.1 A or even 0.05 A
    
    for i in range(len(sci1_fit)):
        axL.scatter(pixc,del_lam,color="blue")
        axL.axhline(0,linestyle="--",color="black")
        axL.set_xlabel("Y Pixel Value")
        axL.set_ylabel("Angstroms")
    
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"LamCalRes_"+str(sim)+".png")
    
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
    
    plt.cla()
    fig, axV = plt.subplots(1, 1, figsize=(8,6))
    for i in range(len(sci1_fit)):
        axV.scatter(pixc,res_vel,color="green")
        axV.axhline(0,linestyle="--",color="black")
        axV.set_xlabel("Y Pixel Value")
        axV.set_ylabel("km/s")
        
    
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"LamCalRes_Velocity_"+str(sim)+".png")
    
    
    print("Mean Wavelength residual (delLAM):",np.mean(del_lam))
    print("Mean Velocity residual (delVEL):",np.mean(res_vel))
    with open(str(target_dir)+"ImageNumber_"+str(sim)+"/KARP_log.txt", "a") as f:
        f.write("Mean Wavelength residual (delLAM):"+str(np.mean(del_lam))+"\n")
        f.write("Mean Velocity residual (delVEL):"+str(np.mean(res_vel))+"\n")
    
        
    
    
    
    plt.cla() # Clear plt to prevent over plotting
    fig, axE = plt.subplots(1, 1, figsize=(8,6))
    axE.scatter(y_lam(y_pix),flux_raw_cor1, s=1)
    axE.scatter(y_lam(y_pix),rnErr, s=1,color='red')
    axE.set_xlabel("Wavelength (A), Error (10x)")
    axE.set_ylabel("Background subtracted Flux (Electrons)")
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"Sci_Flux_LamCal_Res_"+str(sim)+".png", dpi=300)   
    
    # Print dispersion, as (number of angstroms we go over/number of pixels (4096))
    # Recall that our lam is fit inverse of our pixels, ie 246 pix = 6000 Angstroms
    lam_out_max = y_lam(np.min(y_pix)) # Max wavelength as a function of pixel, ie the wavelength fit to the last pixel
    lam_out_min = y_lam(np.max(y_pix)) # Minimum wavelength as a function of pixel, ie the wavelength fit to the first pixel
    print("lam_out_max:",lam_out_max)
    print("lam_out_min:",lam_out_min)
    print("Dispersion: ",(lam_out_max-lam_out_min)/len(y_pix), " Angstroms/pix")
    dispersion = (lam_out_max-lam_out_min)/len(y_pix)
    
    with open(str(target_dir)+"ImageNumber_"+str(sim)+"/KARP_log.txt", "a") as f:
        f.write("lam_out_max:"+str(lam_out_max)+"\n")
        f.write("lam_out_min:"+str(lam_out_min)+"\n")
        f.write("Dispersion:"+str((lam_out_max-lam_out_min)/len(y_pix))+"Angstroms/pix\n")
        
    # Store flux_raw_cor1
    with open(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"flux_raw_cor1_"+str(sim)+".txt", "w") as file:
        for val in flux_raw_cor1:
            file.write(f"{val}\n")
    
    # Store y_lam(y_pix)=wavelengths
    # Heliocentric correction here as well (Should be carried over from y_lam)
    wavelengths = y_lam(y_pix) # Wavelength of each pixel
    with open(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"wavelengths_"+str(sim)+".txt", "w") as file:
        for wave in wavelengths:
            file.write(f"{wave}\n")

    # Store raw sky values
    with open(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"skyraw_"+str(sim)+".txt", "w") as file:
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
        flux_raw_cor1a = open(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"flux_raw_cor1_"+str(sim)+".txt") # read in raw flux
        flux_raw_cor1a1 = flux_raw_cor1a.read()
        flux_raw_cor1b = flux_raw_cor1a1.splitlines()
        flux_raw_cor1a.close()
        
        wavelengths1a = open(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"wavelengths_"+str(sim)+".txt") # read in corresponding wavelengths
        wavelengths1a1 = wavelengths1a.read()
        wavelengthsb = wavelengths1a1.splitlines()
        wavelengths1a.close()
        
        sky1a = open(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"wavelengths_"+str(sim)+".txt") # read in corresponding wavelengths
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
    
    plt.cla() # Clear plt to prevent over plotting
    fig, axE = plt.subplots(1, 1, figsize=(8,6))
    axE.scatter(lam_red,flux_mask_red, s=1, color="green")
    axE.scatter(lam_remove,flux_mask_remove, s=1, color="red")
    axE.set_xlim(3750,4000)
    axE.set_xlabel("Wavelength (A)")
    axE.set_ylabel("Background subtracted Flux (Electrons)")
    
    
    # Fit a continuum to out spectra:
    aa,bb,cc,dd,ee,ff,gg = np.polyfit(lam_red,flux_mask_red,6)    
    #print(aa,bb,cc,dd,ee) # As a*x^4+bx^3+c*x^2+d*x+e = flux_masked
    
    # Our continuum function:
    def con_lam(wave):
        # Vectorize
        wave = np.array(wave)
        # flux as a function of wavelength
        return (aa*(wave)**6)+(bb*(wave)**5)+(cc*wave**4)+(dd*wave**3)+(ee*wave**2)+(ff*wave)+gg
    
    
    con_lam_test = []
    
    for i in range(len(wavelengths)):
        con_lam_test.append(con_lam(wavelengths[i]))
        
    axE.scatter(wavelengths,con_lam_test, s=1, color="black")
    # 6 Zoomed in sci_flux_masked plots
    for i in range(0,3000,500):
        axE.set_xlim((3700+i),(4200+i))
        os.makedirs(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"Sci_Flux_Masked_All_"+str(sim), exist_ok=True) # Initializes a directory for E20 files
        plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/Sci_Flux_Masked_All_"+str(sim)+"/Sci_Flux_Masked_All_i"+str(sim)+"_"+str(i)+".png", dpi=300)   
    
    
    Fnorm_Es = [] # Estimated normalized flux
    for i in range(len(flux_raw_cor1)):
        Fnorm_Es.append(float(flux_raw_cor1[i]/(con_lam(wavelengths[i])))) # Divide raw flux by our continuum that we just fit
    
    # Plot our first estimate on the normalized flux
    plt.cla()
    fig, axCont = plt.subplots(1, 1, figsize=(8,6))
    axCont.scatter(wavelengths,Fnorm_Es, s=1,color="green")
    axCont.set_xlabel("Wavelength (A)")
    axCont.set_ylabel("Estimate Cont. Normalized Flux")
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"Sci_Flux_NormEst_"+str(sim)+".png", dpi=300)   
    
    
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
    
    plt.cla()
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
                    """
                    for k in range(len(wave_range_cor)): # Since the continum fit is so poor in the noisy areas
                        # Simply remove outlier values from our wave ranges when we do fitting
                        # Good continum fits shouldn't have this problem, and won't activate this section of code
                        if float(wave_range_cor[k]) >= 1.3*np.mean(wave_range_cor):
                            print("Found: wave_range_cor[k] >= 1.3*np.mean(wave_range_cor)")
                            wave_range_cor[k] = np.mean(wave_range_cor)
                            print("Replaced with:",np.mean(wave_range_cor))
                    """
                    # Use poly fit to fit a line to the corrected flux values as a funtion of wavelength
                    ml, bl = np.polyfit(wave_range_cor, flux_values_cor, 1)
                    # Now get the fit value of the continum at that wavelength
                    fitted_values.append(ml * wavelengths[i] + bl)
                    """
                    if 4100 <= wavelengths[i] <= 4200:
                        #print("Fnorm_Es[i]:",Fnorm_Es[i])
                        #plt.scatter(wave_range_cor, flux_values_cor, color = "green")
                        for l in range(len(wave_range_cor)):
                            plt.scatter(wave_range_cor[l], flux_values_cor[l], color = "red", s=2, alpha=0.2)
                            plt.scatter(wave_range_cor[l], ml * wave_range_cor[l] + bl, color = "black", s=2, alpha=0.1)    
                        plt.scatter(wavelengths[i],ml*wavelengths[i]+bl, color = "blue", s=6)
                    """
                        
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

    plt.xlim(4100,4200)    
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"LineFitPlots.png", dpi=300)
    # Now smooth this fit with a moving boxcar
    smooth_cont = uniform_filter(np.array(fitted_values), size=norm_line_boxcar)
    
    # Plot fitted flux values from the line fit to see what is going on
    plt.cla()
    fig, axL = plt.subplots(1, 1, figsize=(8,6))
    axL.scatter(fitted_wavelengths,smooth_cont, s=1,color="blue")
    axL.scatter(fitted_wavelengths,fitted_values, s=0.9,color="green")
    
    axL.scatter(fitted_wavelengths,fitted_fluxes, s=0.1,color="black")
    axL.scatter(lam_removea,flux_mask_removea,s=0.1,color="red")
    axL.set_xlabel("Wavelength (A)")
    axL.set_ylabel("Fitted Values over Flux")
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"FitOverFlux_"+str(sim)+".png", dpi=300)   
    
    # Maybe overplot the before and after continums, with and without the removed points to see how much,
    # if at all the points are affecting the continum fit?
    
    # 4 evenly spaced plots of the fitted values, the raw, origonal fluxes, and the smoothed fitted values
    for i in range(3700,6700,500):
        plt.cla()
        axL.scatter(lam_red,flux_mask_red, s=0.5, color="blue") # Non absorption features flux values Fnorm_es < 1.3
        axL.scatter(lam_remove,flux_mask_remove, s=0.5, color="magenta") # Absoprtion features flux values Fnorm_es < 1.3
        axL.scatter(lam_reda,flux_mask_reda, s=0.5, color="green") # All Non absorption features flux values 
        axL.scatter(lam_removea,flux_mask_removea, s=0.5, color="red") # All Absoprtion features flux values
        axL.scatter(fitted_wavelengths,smooth_cont, s=1,color="black") # Continum flux
        axL.set_xlabel("Wavelength (A)")
        axL.set_ylabel("Fitted Values over Flux")
        axL.set_xlim((i),(i+1000))
        os.makedirs(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"FitOverFluxZoom_"+str(sim), exist_ok=True) # Initializes a directory for E20 files
        plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"FitOverFluxZoom_"+str(sim)+"/FitOverFlux_iter_"+str(sim)+"_"+str(i)+".png", dpi=300)   
    
    
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
    
    plt.cla()
    fig, axNorm = plt.subplots(1, 1, figsize=(8,6))
    axNorm.scatter(fitted_wavelengths,Fnorm, s=1,color="black")
    axNorm.scatter(fitted_wavelengths,sf_sm, s=0.5,color="red")
    axNorm.axhline(1,color="red",linestyle="--")
    axNorm.set_xlabel("Wavelength (A)")
    axNorm.set_ylabel("Normalized Flux")
    axNorm.set_title("Normalized Spectra for "+str(objID)+" Image:"+str(sim))
    axNorm.set_ylim(0,1.5)
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"Sci_Normalized_"+str(sim)+".png", dpi=300)   
    
    # Some other diagnostic plots
    plt.cla()
    fig, axNorm = plt.subplots(1, 1, figsize=(8,6))
    axNorm.scatter(fitted_wavelengths[3000:4096],Fnorm[3000:4096], s=1,color="black")
    axNorm.set_xlim(3760,4400)
    axNorm.axhline(1,color="red",linestyle="--")
    axNorm.set_xlabel("Wavelength (A)")
    axNorm.set_ylabel("Normalized Flux")
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"Sci_Normalized_Left"+str(sim)+".png", dpi=300)   
    
    
    plt.cla()
    fig, axNorm = plt.subplots(1, 1, figsize=(8,6))
    axNorm.scatter(fitted_wavelengths[2700:3300],Fnorm[2700:3300], s=1,color="black")
    axNorm.set_xlabel("Wavelength (A)")
    axNorm.set_ylabel("Normalized Flux")
    plt.savefig(str(target_dir)+"ImageNumber_"+str(sim)+"/"+"Sci_Normalized_hbeta"+str(sim)+".png", dpi=300)   
    
    # Write a file for the spectra of this sci image
    # In the form G123456_61_OUT.fits in the OUT folder
    t = Table([fitted_wavelengths,Fnorm,sf_sm], names=('Lam', "Flux", "Sigma"))
    t.write(str(target_dir)+"OUT/"+str(objid)+"_"+str(sim)+"_OUT.fits", format='fits', overwrite=True)

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
                
    plt.cla()
    fig, axNorm = plt.subplots(1, 1, figsize=(8,6))
    axNorm.scatter(gwave,med_comb, s=0.8,color="black")
    axNorm.axhline(1,color="red",linestyle="--")
    axNorm.set_xlabel("Wavelength (A)")
    axNorm.set_ylabel("Median Normalized Flux")
    axNorm.set_ylim(-0.1,1.5)
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
    
    axNorm.scatter(gwave,sig_final, s=0.3,color="red")
    axNorm.set_title("Median Normalized Spectra for "+str(objID)+", SNR 5500 A:"+str(np.round(1/RMS, decimals=2)))
    
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
    
    axNorm.scatter(wout,fout,s=0.6,color="green")
    axNorm.scatter(wout,eout,s=0.3,color="blue")
    
    plt.savefig(str(target_dir)+"OUT/"+str(objid)+"_OUT.png", dpi=300)
    
    for i in range(0,4000,40):
        axNorm.set_xlim((3200+i),(3700+i))
        plt.savefig(str(target_dir)+"OUT/"+str(objid)+"_OUT_Zoom"+str(i)+".png", dpi=300)


    # Measure the radial velocity of the star based on these 7 absorption lines
    print("Measuring radial velocity of the star")
    # Line names: H theta, H eta, H zeta, H delta, H gamma, H alpha, Ca K, Don;t use: Hbeta, HeI(sdB) 4471.5, HeI 4173, HeI 4922, HeI 5016
    vel_lines = [3797.91,3835.40,3889.06,4101.73,4340.47,6562.79,3933.66] # In angstroms ,,4861.35,4471.5,4713,4922
    vel_mask = [20,35,35,40,50,50,20,35,10,10,10] # half-width of lines to fit in angstroms
    vel_fit = [] # The fit parameters for the velocity fits
    vel_fit_linenum = [] # Which of the seven lines we just fit
    
    # Define a Gaussian
    def G(x, a, mu, sigma, bck):
        return (a * np.exp(-(x-mu)**2/(2*sigma**2))) + bck
        # A 4d Gaussian
    
    
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
            popt, _ = curve_fit(G, wave_vals, flux_vals, bounds=bounds, p0=p0, maxfev=5000) # ignore covariance matrix spat out from curve_fit
            vel_fit.append(popt)  # [a, mu, sigma, c] returend from curve_fit
            vel_fit_linenum.append(i) # Wavelength values that where succesfully fit
        except RuntimeError:
            print("Curve fit failed at maxfev=1000")
            
    # Now convert the measured center of the absorption feature from angstroms to km/s
    # Line names: H theta, H eta, H zeta, H delta, H gamma, H alpha, Ca K
    line_velocities_karp = [] # The velocites of the absorption lines that we get from KARP
    for i in range(len(vel_fit)):
        a, mu, sig, bck = vel_fit[i]
        print("For line:",vel_fit_linenum[i]+1,"KARP fit:",mu,"Angstroms")
        print("a:",a,"sig:",sig,"bck:",bck)
        line_velocities_karp.append(((float(mu)-float(vel_lines[i]))/float(vel_lines[i]))*3*10**5)
        print("(lam-lam_0)/lam_0:",((float(mu)-float(vel_lines[i]))/float(vel_lines[i])),"A")
        print("mu in km/s:",((float(mu)-float(vel_lines[i]))/float(vel_lines[i]))*3*10**5)
        plt.cla()
        fig, axVel = plt.subplots(1, 1, figsize=(8,6))
        axVel.scatter(gwave,med_comb, s=5,color="black")
        axVel.plot(gwave,G(gwave,a,mu,sig,bck))
        axVel.axhline(1,color="red",linestyle="--")
        axVel.axvline(mu,color="black",linestyle="--")
        # Correct Max's points for heliocentric velocity with v=4km/s
        for j in range(len(wout)):
            wout[j] = wout[j]*(1+(4/(3*10**5)))
        
        axVel.scatter(wout,fout,color="green",s=1)
        axVel.axvline(vel_lines[i],color="blue",linestyle="--")
        axVel.set_xlim(int(vel_lines[i]-vel_mask[i]),int(vel_lines[i]+vel_mask[i]))
        axVel.set_xlabel("Wavelength (A)")
        axVel.set_ylabel("Median Normalized Flux")
        del_vel_lam = np.round(float(mu)-float(vel_lines[i]),decimals=4)
        axVel.set_title("lam-lam_0: "+str(del_vel_lam)+" A")
        if i >= 8:
            axVel.set_ylim(0.9,1.1)
            axVel.set_title("HeI "+str(vel_lines[i])+"A line lam-lam_0: "+str(del_vel_lam)+" A radV: "+str(((float(mu)-float(vel_lines[i]))/float(vel_lines[i]))*3*10**5)+"km/s")
        
        plt.savefig(str(target_dir)+"OUT/"+str(objid)+"_Vel_"+str(i+1)+".png", dpi=300)
    
    # average the velocities of the 7 lines:
    radial_vel = np.mean(line_velocities_karp)
    radial_vel_err = np.std(line_velocities_karp)
    print("Radial velocity:",radial_vel,"+/-", radial_vel_err)
    #print(wing_offset)        
    # Look for HI line, 4471.5
    
    
    #find equivalent widths of metal lines
    metal_lines = [3820.425, 3933.66, 4045.812, 4063.594, 4226.728, 4260.474, 4271.76, 4307.902, 4383.545, 4404.75, 4957.596, 5167.321, 5172.684, 5183.604, 5269.537, 5328.038]
    metal_mask = [3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4]
    metal_fit = []
    metal_fit_linenum = []
    
    fix_back = 1
    def G_3d(x, a, mu, sigma):
        return G(x, a, mu, sigma, fix_back)
    
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
            popt, pcov = curve_fit(G, wave_vals, flux_vals, p0=p0, bounds=bounds, maxfev=5000)
            metal_fit.append(popt)
            metal_fit_linenum.append(i)
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
        error = np.sqrt(np.sum(np.square(wave_err)))
        sys = .002 * metal_mask[i] * 2
        adopt_err = np.sqrt(error**2 + sys**2)
        
        
        #print("Model fits", popt)
        print("------------------")
        print(f"Metal line at {line}")
        print(f"EW_g is {equi_width:.4f} and EW is {ew_sum:.4f}")
        print(f"Error is {error:.4f} and sys is {sys}")
        print(f"Adopted is {(equi_width+ew_sum)/2:.4f} with error {adopt_err:.4f}")
         
	
    #make graphs of fits around lines
    for i, popt in enumerate(metal_fit):
        plt.cla()
        fig, axMet = plt.subplots(1, 1, figsize=(8,6))
        axMet.scatter(gwave,med_comb, s=5,color="black")
        axMet.plot(gwave,G(gwave,popt[0],popt[1],popt[2], popt[3]))
        axMet.axhline(1,color="red",linestyle="--")
        axMet.axvline(mu,color="black",linestyle="--")
        axMet.set_xlim(int(metal_lines[i]-5),int(metal_lines[i]+5))
        axMet.set_xlabel("Wavelength (A)")
        axMet.set_ylabel("Median Normalized Flux")
        
        axMet.set_title("Metal line fit at "+str(metal_lines[i])+" A")
        if i >= 8:
            axMet.set_ylim(0.9,1.1)
            
        plt.savefig(str(target_dir)+"OUT/"+str(metal_lines[i]) + ".png", dpi=300)
            
# Duration_run is how long KARP took to run
duration_run = timedelta(seconds=time.perf_counter()-starttime)
with open(str(target_dir)+"ImageNumber_"+str(sim)+"/KARP_log.txt", "a") as f:
    f.write("KARP took:"+str(duration_run)+" to run\n")
    f.write("><(((º>")
    

# Initialize mixer
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

# Create sound object and play
sound = pygame.mixer.Sound(buffer=audio)
sound1 = pygame.mixer.Sound(buffer=audio1)
sound.play()
pygame.time.delay(int(duration * 1000))  # wait for playback to finish
sound1.play()
pygame.time.delay(int(duration * 1000))  # wait for playback to finish


print("KARP took:",duration_run," to run")
print("><(((º>")
