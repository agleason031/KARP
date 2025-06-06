#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
KARP

KOSMOS Astronomical Reduction Pipeline

Created by Chase L. Smith, Max Moe

><(((º>
"""

# Importing relevant packages
from datetime import timedelta
from astropy import log
log.setLevel('ERROR')
from scipy.optimize import dual_annealing
from grapher import grapher
import numpy as np
import argparse
import warnings
import time
import os
import pygame
import functions
import sci_tools
import sci_modules

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
# Suppress warnings
warnings.filterwarnings("ignore")

#----------------------------------
#function definitions

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

def process_images(x):
    #optimize == true overrides control flow
    if optimize == True:
        sci_mods.appw = int(round(x[0]))
        grapher.appw = int(round(x[0]))
        sci_mods.norm_line_width = int(round(x[1]))
        sci_mods.norm_line_boxcar = int(round(x[2]))
        
    if skip_red == False or optimize == True:
        #get image filenames
        blist = [functions.format_fits_filename(dloc, b) for b in bim]
        flist = [functions.format_fits_filename(dloc, f) for f in fim]
        sci_list = [functions.format_fits_filename(dloc, s) for s in sim]
    
        masterbias, final_flat = sci_tools.get_cal_images(blist, flist, verbose, grapher)
        
        #individual image processing
        global_args = sci_list, masterbias, final_flat, verbose
        for scinum in sim:
            sci_mods.reduce_image(scinum, global_args)
        
    if spec_norm == True or optimize == True:
        #do individual image processing here too
        for scinum in sim:
            sci_mods.normalization(scinum, verbose)
            
    # If comb_spec = True, then take a number of input tables labled as
    # G123456_61_OUT.txt, G123456_62_OUT.txt, G123456_63_OUT.txt
    # for normalized spectra for science images 61, 62,  and 63 for example
    if comb_spec == True or optimize == True:
        SNR = sci_mods.combine_spectra(verbose)
        
    return SNR

def negative_snr(x):
    return -process_images(x)
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
    
    # Take the parameter file from the command line
    parser = argparse.ArgumentParser(prog="KARP")
    parser.add_argument('-kparam', '--kp') # Location of our data
    args = parser.parse_args()
    
    #get input parameters
    params = take_bait(str(args.kp))
    
    # assign relevant values
    dloc = str(params["dloc"]) # Location to store output data
    scistart = int(params["scistart"]) #first sci image
    scistop = int(params["scistop"]) + 1 #last sci image, add 1 to include final image
    fstart = int(params["fstart"]) # The first flat image, assuming the flats are ordered sequentially
    fstop = int(params["fstop"]) # The last flat image, assuming the flats are ordered sequentially
    bstart = int(params["bstart"]) # The first bias image assuming the bias images are ordered sequentially
    bstop = int(params["bstop"]) # The last bias image, assuming the bias images are ordered sequentially
    target_dir = str(params["target_dir"])
    skip_red = functions.string_to_bool(params["skip_red"])
    comb_spec = functions.string_to_bool(params["comb_spec"])
    fit_lines = functions.string_to_bool(params["fit_lines"])
    spec_norm = functions.string_to_bool(params["spec_norm"])
    optimize = functions.string_to_bool(params["optimize"])
    verbose = functions.string_to_bool(params["verbose"])
    
    grapher = grapher(params)
    sci_mods = sci_modules.sci_modules(params, grapher)
    
    if (verbose == True):
        print("KARP has taken the bait! (Input file):")
        print("------------")
        print(params) # Print out parameters
        print("------------")
        
        os.makedirs(target_dir+"ImageNumber_"+str(scistart), exist_ok=True)
        with open(target_dir+"ImageNumber_"+str(scistart)+"/KARP_log.txt", "a") as f:
            f.write("KARP has taken the bait! (Input file):\n")
            f.write("------------\n")
            f.write(str(params))
            f.write("------------\n")

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
    sim = np.arange(scistart, scistop, 1)
    
    # Make a global "OUT" folder to keep finalized outputs
    os.makedirs(target_dir+"OUT", exist_ok=True) # Initializes a directory OUT files
    
#------------------------------------------
    
    if (optimize == True):
        bounds = [
            (5, 20),  # a_width
            (150, 400),    # norm_line_width
            (30, 100),    # norm_line_boxcar
            ] 
        
        res = dual_annealing(negative_snr, bounds=bounds, maxiter=30, minimizer_kwargs={'method': 'L-BFGS-B', 'tol': 1.0})
        optimal_width = int(round(res.x[0])) * 2 + 1
        optimal_snr = -res.fun
        optimal_norm_width = res.x[1]
        optimal_boxcar = res.x[2]
        
        print(f"Optimal aperture width: {optimal_width}, SNR: {optimal_snr}")
        print(f"Optimal norm width: {optimal_norm_width} and optimal boxcar: {optimal_boxcar}")
    else:
        SNR = process_images((sci_mods.appw, sci_mods.norm_line_width, sci_mods.norm_line_boxcar))    
        
    if (fit_lines == True):
        sci_tools.fit_vel_lines(sci_mods.gwave, sci_mods.med_comb, grapher)
        sci_tools.fit_metal_lines(sci_mods.gwave, sci_mods.med_comb, sci_mods.sig_final, grapher)
        
#--------------------------------------------------------
#finishing tasks
                
    # Duration_run is how long KARP took to run
    duration_run = timedelta(seconds=time.perf_counter()-starttime)
    with open(target_dir+"OUT/KARP_OUT_log.txt", "a") as f:
        f.write("KARP took:"+str(duration_run)+" to run\n")
        f.write("><(((º>")
    
    #play finishing sound
    sound.play()
    pygame.time.delay(int(duration * 1000))  # wait for playback to finish
    sound1.play()
    pygame.time.delay(int(duration * 1000))  # wait for playback to finish
    
    print("KARP took:",duration_run," to run")
    print("><(((º>")
