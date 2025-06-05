#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:52:33 2025

@author: agleason
"""
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import uniform_filter
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor
from astropy.nddata import CCDData
from astropy.table import Table
from functions import fit_cent_gaussian
from functions import G
from functions import format_fits_filename
from functions import y_lam
from functions import con_lam
from sci_tools import heliocentric_correction
import ccdproc
import os
import glob
import numpy as np

#-------------------------------------
#helper functions

def run_fit(args):
    row, clr, appw, buffw, bckw = args
    return fit_cent_gaussian(row, clr, appw, buffw, bckw)

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

#-------------------------

class sci_modules:
    def __init__ (self, params, grapher):
        # assign relevant values
        self.dloc = str(params["dloc"]) # Location to store output data
        self.objid = str(params["objid"]) # Object ID
        self.objRA = str(params["objRA"]) # Object RA in the form "05 02 58.72"
        self.objDEC = str(params["objDEC"]) # Object DEC in the form "-70 49 44.7"
        self.otime = str(params["obsTime"]) # The time the object was observed, listed as t = Time("2015-06-30 23:59:60.500")
        self.scistart = int(params["scistart"]) #first sci image
        self.argnum = int(params["argnum"]) # The image number of the Argon spectral calibration image
        self.appw = int(np.round(float(params["appw"])-1)/2) # So an appature width inputu of 13 is two 6 pixel sides, and if you put 14 it rounds to 13
        self.buffw = int(params["buffw"]) # Width of the buffer in pixels (on either side of the appature, so 4 is two, 4 pix buffers su,etrically spaced around the center line)
        self.bckw = int(params["bckw"]) # The width of the background in pixels, (on either side of the buffer)
        self.clmin = int(params["clmin"]) # The minimum center line trace value that KARP will fit
        self.clmax = int(params["clmax"]) # The minimum center line trace value that KARP will fit
        self.target_dir = str(params["target_dir"]) 
        self.norm_line_width = int(params["norm_line_width"])  # When normalizing spectra, this is the one-half width of the line KARP will try and fit at each point
        self.norm_line_boxcar = int(params["norm_line_boxcar"]) # When normalizing the spectra, 
        
        self.grapher = grapher
        
    #does all the image reduction through wavelength calibration
    def reduce_image(self, scinum, global_args):
        sci_list, masterbias, final_flat, verbose = global_args   
    
        sci_location = sci_list[scinum - self.scistart]
        
        # Make a output text file detaling relevant fit paramaters that KARP calculates
        # Initalize directory for reduction plots for this science image
        os.makedirs(self.target_dir+"ImageNumber_"+str(scinum), exist_ok=True) # Initializes a directory for E20 files            
        
        print("KARP is reducing Science Image number:",scinum, flush=True)
        with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
            f.write("KARP Output Log\n")
            f.write("- - - ><(((º>\n")
            f.write("\n")
            f.write("KARP is reducing Science Image number:"+str(scinum)+"\n")
        
        sci_final = (ccdproc.subtract_bias(CCDData.read(sci_location,format = 'fits', unit = "adu"),masterbias).data)/final_flat
        if (verbose == True):
            self.grapher.plot_sci_final(sci_final, scinum)
        sci_final_write = CCDData(sci_final, unit="adu")
        sci_final_write.write(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"sci_final_"+str(scinum)+".fits", overwrite = True)
        
        # Read all three of our final sci images and convert to nparrays
        sci_final_1 = np.asarray(CCDData.read(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"sci_final_"+str(scinum)+".fits", unit = "adu").data)
        
        # Convert from ADU to electrons
        sci_final_1 = sci_final_1*0.6
        
    #----------------------------
    #trace fitting
        
        n_rows = sci_final_1.shape[0] #gets row num
        cen_line = np.round(np.linspace(self.clmin, self.clmax, n_rows)).astype(int)
        # Note that this repeats each integer 51 times, stretching it to 1124
        
        # Fit Gaussian to each row in sci_final_1 using current cen_line estimate
        print("KARP fitting centerline")
        
        # Fit and collect results in a list (each entry is a tuple: a, mu, sig, bck)
        args_list = [(row, cen_line[i], self.appw, self.buffw, self.bckw)
                 for i, row in enumerate(sci_final_1)]
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            cen_fit = np.array(list(tqdm(executor.map(run_fit, args_list), total=len(args_list))))
        
        # Convert to NumPy array and extract relevant fit parameters
        a_vals = cen_fit[:, 0]                    # Amplitudes
        mu_vals = cen_fit[:, 1]                   # Gaussian centers (used for center line)
      
        # Create preliminary centerline using fitted mu values
        # If a or mu is NaN, mark the entry as NaN in cen_line for smoothing later
        cen_line = np.round(mu_vals).astype(float)
        cen_line[np.isnan(a_vals) | np.isnan(mu_vals)] = np.nan  # Replace invalid fits with NaN
        
        # Smooth NaN values in cen_line using a local (moving average) window
        # We use a NaN-aware uniform filter that preserves non-NaN values and smooths only missing ones
        print("Smoothing NaNs in centerline", flush=True)
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
                if (verbose == True):
                    print(f"Outlier at {i}: Replacing {cen_line[i]} with {np.round(cmean)}")
                cen_line[i] = int(np.round(cmean))     # Replace with smoothed local mean
        
        # For the tails of the image if the fit becomes very poor, bin to the set max values
        for i in range(len(cen_line)):
            if cen_line[i] > self.clmax:
                cen_line[i] = int(self.clmax)
            if cen_line[i] < self.clmin:
                cen_line[i] = int(self.clmin)
                
        # cen_line is now cleaned, smoothed, and robust to fitting artifacts
        cen_line = np.sort(cen_line) # Doesn't change the number of cen_line pixels, just sorts outliers
        self.grapher.plot_cen_line(cen_line, scinum)
        
    #----------------------------------
    #flux extraction
        
        # Fit parameters to each row, need new fit after cleaned cen_line
        args_list = [(row, cen_line[i], self.appw, self.buffw, self.bckw)
                 for i, row in enumerate(sci_final_1)]
        
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            sci1_fit = list(tqdm(executor.map(run_fit, args_list), total=len(args_list)))
    
        print("KARP finished fitting flux Gaussians")
        
        if (verbose == True):
            print("10 Flux Fit Parameters:")
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                f.write("10 Flux Fit Parameters:"+"\n")
            for i in range(0,4000,400):
                a, mu, sigma, bck = sci1_fit[i]
                print("Y:",i," a:",a," mu:",mu," sig:",sigma," bck:",bck)
                with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                    f.write("Y:"+str(i)+" a:"+str(a)+" mu:"+str(mu)+" sig:"+str(sigma)+" bck:"+str(bck)+"\n")
            # Plot Gaussian center values vs y pix and Gaussian sigma value vs Y pix
            print("Plotting Gauss cen values and Gauss sig values vs Y pix")
            self.grapher.gauss_cen_plots(sci1_fit, cen_line, scinum)
            
        
        cen_line = np.round(cen_line).astype(int)
        # Make a plot of 20 evenly spaced pix
        # Showing the data, the apertures and buffers as lines,
        # And the over plotted G fit
        
        if (verbose == True):
            print("Making 20 aperture plots and aperature fit plots")
            self.grapher.make_aperature_fit_plots(sci1_fit, sci_final_1, cen_line, scinum)
            self.grapher.make_aperature_plots(sci1_fit, sci_final_1, cen_line, scinum)
            
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
            c_flux, _ = quad(g_func,c-self.appw, c+self.appw) # Center app flux
            bkg_right, _ = quad(g_func,(c-(self.appw+self.buffw+self.bckw)),(c-self.appw))
            bkg_left, _ = quad(g_func,(c-(self.appw+self.buffw+self.bckw)),(c-self.appw))
            # Append flux_raw with our flux for each row
            flux_raw1.append(c_flux-((self.appw/(2*self.bckw))*(bkg_right+bkg_right)))    
            inrow = sci_final_1[i]
            sky_raw1.append(float(self.appw/self.bckw)*(np.sum(inrow[(c-self.appw-self.buffw-self.bckw):(c-self.appw-self.buffw)])+np.sum(inrow[(c+self.appw+self.buffw+self.bckw):(c+self.appw+self.buffw)]))) # Sum up both sides of the background
            # Recall that we need to scale for the amount of "Sky" that is technically within our aperture size
        
        if (verbose == True):
            # Plot output raw flux (We can remove CRs later)
            self.grapher.plot_raw_flux(flux_raw1, scinum)
        
    #--------------------------------------
    #wavelength calibration
    
        pixc = [250,295,765,925,953,1275,1330,1396,1480,2870,3355,3421,3612,3780]
        lam = [6416.31,6384.72,6032.13,5912.09,5888.58,5650.70,5606.73,5558.70,5495.87,4510.73,4200.67,4158.59,4044.42,3948.97]
        
        argon = np.asarray(CCDData.read(format_fits_filename(self.dloc, self.argnum), unit="adu").data)
        
        # Lambda calibration
        print("Fitting lambda Gaussians...")
        
        lam_width = 7 # Width to fit wavelength gaussians
        lam_fit = [] # Stores fit parameters as [a, mu, sig]
        lam_fit_linenum = [] # Succesfully fitted line wavelength
        lam_fit_pixc = [] # Succesfully fitted line pixc
        
        
        for i in range(len(pixc)):
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
                if (verbose == True):
                    print("Curve fit failed at maxfev=1000")
                    print("Trying to fit:",lam[i],pixc[i])
                    print("p0 is:",p0)
                    print("Now trying with maxfev=10000")
                lam_width = 11
                #print("And lam_width:",lam_width)
                a_col = argon[:, cen_line[i]] # Get every column value that is in cen_line
                flux_vals = a_col[(pixc[i]-lam_width) : (pixc[i]+lam_width)] # Get argon values from the same pixels
                # We want locations along the y
                loc_vals = np.arange((pixc[i]-lam_width),(pixc[i]+lam_width)) # fit to y pixel values of 5 of the center of the argon wavelength
               
                p0 = [np.max(flux_vals), pixc[i], 2.0, 0] # Guess that the gussian peak is the max height
                bounds = [(0,0,1,-100),(np.inf,np.inf,np.inf,np.inf)] # a, mu, sig, bck
                # Curve fit takes, x (locations), y (values)
                #print("NEW p0 is:",p0)
                popt, _ = curve_fit(G, loc_vals, flux_vals, p0=p0, bounds=bounds, maxfev=10000) # ignore covariance matrix spat out from curve_fit
                lam_fit.append(popt)  # [a, mu, sigma, c] returend from curve_fit
                lam_fit_linenum.append(lam[i]) # Wavelength values that where succesfully fit
                lam_fit_pixc.append(pixc[i]) # Pixc of successfully fit wavelength values
        
        if (verbose == True):
            print("KARP fitted ", len(lam_fit), " lines")
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                f.write("KARP fitted"+str(len(lam_fit))+"lines\n")
        cfits = [] # Empty array for putting the fitted line centers
        for i in range(len(lam_fit)): # For each line KARP was able to fit, get the center of the Gaussian from the fit    
            cfits.append(lam_fit[i][1])
        
        if (verbose == True):
            print("Argon Lines Fit Parameters:")
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                f.write("Argon Lines Fit Parameters:\n")
            
            s_lsp = [] # Sigmas of the lam line fits for the line split function
            for i in range(len(lam_fit)):
                a, mu, sigma, bck = lam_fit[i]
                s_lsp.append(sigma)
                print("Lam Line Number:",i," lam:",lam[i]," a:",a," mu:",mu," sig:",sigma," bck:",bck)
                with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                    f.write("Lam Line Number:"+str(i)+" lam:"+str(lam[i])+" a:"+str(a)+" mu:"+str(mu)+" sig:"+str(sigma)+" bck:"+str(bck)+"\n")
            #for i in range(len(lam_fit_linenum)):
            #   print("Lam:",lam_fit_linenum[i],"pixc:",lam_fit_pixc[i],"ycen actual fit to argon lines:",cfits[i])
            
            print("Line Split Function:",np.mean(s_lsp))
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                f.write("Line Split Function:"+str(np.mean(s_lsp))+"\n")
            print("(avg of sig of line fits)")
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                f.write("(avg of sig of line fits)\n")    
        
        # Fit a cubic poly nomial with cfits (actual line centers)
        # Fitting c fits (actual line centers in pixels) to lam_fit_linenum (successfully fit line wavelength values from lam)
        print("Fitting cubic polynomial")
        cfits_red = []
        for i in range(len(cfits)):
            cfits_red.append(cfits[i]/2000)
        y_lam_fit = np.polyfit(cfits_red,lam_fit_linenum,3)    
        #print(a,b,c,d) # As a*x^3+bx^2+c*x+d = lam(y)
        flux_raw1 = np.array(flux_raw1)
        ii = flux_raw1 > 0
        flux_raw_cor1 = flux_raw1[ii]
        y_pix = np.arange(0,len(flux_raw_cor1),1)
        
        v_helio = heliocentric_correction(self.objRA, self.objDEC, self.otime)
        
        if (verbose == True):
            print("v_helio:",v_helio,"km/s") # -4 km/s for G093440
            self.grapher.plot_sci_lamcal(y_lam, y_pix, flux_raw_cor1, y_lam_fit, v_helio, scinum)
    
        # Lambda calibration graph
        del_lam = [] # Lam_fit - Lam_cal
        if (verbose == True):
            print("Calculating RMS A")
            print("len(cfits)",len(cfits))
            print("len(lam):",len(lam))
        for i in range(len(lam)):
            # cfits is the center fit value in pixels
            if (verbose == True):
                print(cfits[i])
            if not np.isnan(cfits[i]):
                del_lam.append(float(lam[i])-float(y_lam(cfits[i], y_lam_fit, v_helio))) # The wavelength value we want to fit the line to MINUS the value our fit says that that line is at
            else:
                print("NAN here")
            #print(offset[i])
            
        # Put print RMS here
        dlS = []
        for i in range(len(del_lam)):
            dlS.append(del_lam[i]**2)
        if (verbose == True):
            print("RMS A:",np.sqrt(np.mean(dlS)))
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                f.write("RMS A:"+str(np.sqrt(np.mean(dlS)))+"\n")
        print("Calculating RMS km/s")
        dlols = [] # Delta lambda over lambda
        for i in range(len(del_lam)):
            dlols.append(float(del_lam[i]/lam[i])**2)
        
        print("RMS km/s:",np.sqrt(np.mean(dlols))*3*10**5)
        with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
            f.write("RMS km/s:"+str(np.sqrt(np.mean(dlols))*3*10**5)+"\n")
        # Should be ~10 km/s 0.1 A or even 0.05 A
    
        # Print RMS in A, and in velocity
        # and plot counts vs error
        rnErr = []
        res_vel = [] # Residuals in velocity space
        for i in range(len(lam)):
            if (verbose == True):
                print("Angstrom:",lam[i]," sqrt(flux+sky):",(float(flux_raw_cor1[i]+sky_raw1[i])**(1/2))) # Print the RMS error
                print("km/s:",(del_lam[i]/lam[i])*3*10**5," sqrt(flux+sky):",(float(flux_raw_cor1[i]+sky_raw1[i])**(1/2))) # Print the RMS error
            res_vel.append((del_lam[i]/lam[i])*3*10**5)
        
        print("Plotting Residuals...")
        
        for i in range(len(flux_raw_cor1)): # For our corrected fluxes
            rnErr.append(float(flux_raw_cor1[i]+sky_raw1[i])**(1/2))
    
        if (verbose == True):
            #make lamcalres plots
            self.grapher.lam_cal_res(pixc, del_lam, scinum)
            self.grapher.lam_cal_res_vel(pixc, res_vel, scinum)
            
            print("Mean Wavelength residual (delLAM):",np.mean(del_lam))
            print("Mean Velocity residual (delVEL):",np.mean(res_vel))
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                f.write("Mean Wavelength residual (delLAM):"+str(np.mean(del_lam))+"\n")
                f.write("Mean Velocity residual (delVEL):"+str(np.mean(res_vel))+"\n")
        
            self.grapher.plot_sci_lamcal_res(y_lam, y_pix, flux_raw_cor1, rnErr, y_lam_fit, v_helio, scinum)
        
        # Print dispersion, as (number of angstroms we go over/number of pixels (4096))
        # Recall that our lam is fit inverse of our pixels, ie 246 pix = 6000 Angstroms
        lam_out_max = y_lam(np.min(y_pix), y_lam_fit, v_helio) # Max wavelength as a function of pixel, ie the wavelength fit to the last pixel
        lam_out_min = y_lam(np.max(y_pix), y_lam_fit, v_helio) # Minimum wavelength as a function of pixel, ie the wavelength fit to the first pixel
        
        
        if (verbose == True):
            dispersion = (lam_out_max-lam_out_min)/len(y_pix)
            print("lam_out_max:",lam_out_max)
            print("lam_out_min:",lam_out_min)
            print("Dispersion: ",dispersion, " Angstroms/pix")
        
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                f.write("lam_out_max:"+str(lam_out_max)+"\n")
                f.write("lam_out_min:"+str(lam_out_min)+"\n")
                f.write("Dispersion:"+str(dispersion)+"Angstroms/pix\n")
            
        # Store flux_raw_cor1
        with open(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"flux_raw_cor1_"+str(scinum)+".txt", "w") as file:
            for val in flux_raw_cor1:
                file.write(f"{val}\n")
        
        # Store y_lam(y_pix)=wavelengths
        # Heliocentric correction here as well (Should be carried over from y_lam)
        wavelengths = y_lam(y_pix, y_lam_fit, v_helio) # Wavelength of each pixel
        with open(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"wavelengths_"+str(scinum)+".txt", "w") as file:
            for wave in wavelengths:
                file.write(f"{wave}\n")
    
        # Store raw sky values
        with open(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"skyraw_"+str(scinum)+".txt", "w") as file:
            for sky in sky_raw1:
                file.write(f"{sky}\n")

#-----------------------------------------

    def normalization(self, scinum, verbose):
        # Mask out deep absorption features
        # Deep features in order of ascending wavelength, in angstroms
        # We have spectra down to Lam = 3780 and up to 6600 angstroms
        # H_8,H_eta, H_zeta,Ca_K,Ca_H, H_epsilon,H_delta,H_gamma, H_beta, SodiumD2, SodiumD1, H_alpha
        features = [3796.94,3835.40,3889.06,3933.7,3969,3970.075,4101.73,4340.47,4861.35,5889.96,5895.93,6562.81]
        
        flux_raw_cor1a = open(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"flux_raw_cor1_"+str(scinum)+".txt") # read in raw flux
        flux_raw_cor1a1 = flux_raw_cor1a.read()
        flux_raw_cor1b = flux_raw_cor1a1.splitlines()
        flux_raw_cor1a.close()
        
        wavelengths1a = open(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"wavelengths_"+str(scinum)+".txt") # read in corresponding wavelengths
        wavelengths1a1 = wavelengths1a.read()
        wavelengthsb = wavelengths1a1.splitlines()
        wavelengths1a.close()
        
        sky1a = open(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"wavelengths_"+str(scinum)+".txt") # read in corresponding wavelengths
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
        
        print(f"Estimating continuum on {scinum}")
        # Make each line have its own feature mask width to get a good mask
        # H_8,H_eta, H_zeta,Ca_K,Ca_H, H_epsilon,H_delta,H_gama, H_beta, SodiumD2, SodiumD1, H_alpha
        
        # Wide features, for bright spectra
        feature_mask = [10,15,19,10,10,35,40,60,60,10,15,30]
    
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
            con_lam_test.append(con_lam(wavelengths[i], fit_params))            
        Fnorm_Es = [] # Estimated normalized flux
        for i in range(len(flux_raw_cor1)):
            Fnorm_Es.append(float(flux_raw_cor1[i]/(con_lam(wavelengths[i], fit_params)))) # Divide raw flux by our continuum that we just fit
        
        # Plot our first estimate on the normalized flux
        if (verbose == True):
            self.grapher.plot_sci_flux_norm_est(wavelengths, Fnorm_Es, scinum)
            self.grapher.plot_sci_flux_masked(lam_red, lam_remove, flux_mask_red, flux_mask_remove, wavelengths, con_lam_test, scinum)
        
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
        
        rm_param = 1.3
        include = Fnorm_Es <= rm_param
        
        # Precompute NaN masks for spectral features
        for i in range(len(wavelengths)):            
            if Fnorm_Es[i] <= rm_param:  # Skip things that are brighter than 1.3 norm flux
                lam_left = max(0, i - self.norm_line_width)
                lam_right = min(len(wavelengths), i + self.norm_line_width)
                wave_range = wavelengths[lam_left:lam_right] # Get wavelengths and fluxes around where we're trying to fit
                flux_values = flux_raw_cor1[lam_left:lam_right]
                keep = include[lam_left:lam_right]
                
                wave_range = np.array(wave_range)[keep]
                flux_values = np.array(flux_values)[keep]
                
                if removed_index[i] == 0: # If this wavelength is not in an absoprtion feature
                    # If our norm_line_width spills over into lines on either side we want to ignore them
                    wave_range_cor = []
                    flux_values_cor = []
                    array = np.arange(lam_left, lam_right, 1)
                    for k in range(len(wave_range)):
                        if removed_index[array[k]] == 0:  # Check if there are any absorption features within the region we want to be fitting
                        # (Recall that removed_inxed=0 corresponds to values we want to keep for fitting)
                        # if there is not an absorption feature at that wavelength then keep the flux value for fitting
                            wave_range_cor.append(wave_range[k])
                            flux_values_cor.append(flux_values[k])  # Align flux_values index, as flux starts at 0
                            
                    if len(wave_range_cor) >= 2: # Check if we have enough points to fit
                        # Use poly fit to fit a line to the corrected flux values as a funtion of wavelength
                        ml, bl = np.polyfit(wave_range_cor, flux_values_cor, 1)
                        # Now get the fit value of the continum at that wavelength
                        fitted_values.append(ml * wavelengths[i] + bl)                        
                    else:
                        continue  # Skip if not enough good values to fit
                        
        
                elif removed_index[i] != 0: # Wavelength is inside of an absorption feature
                    lam_left = max(0, i - self.norm_line_width)
                    lam_right = min(len(wavelengths), i + self.norm_line_width)
        
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
        smooth_cont = uniform_filter(np.array(fitted_values), size=self.norm_line_boxcar)
        
        # Plot fitted flux values from the line fit to see what is going on
        self.grapher.plot_fit_over_flux(fitted_wavelengths, smooth_cont, fitted_values, fitted_fluxes, lam_removea, flux_mask_removea, lam_red, lam_remove, flux_mask_red, flux_mask_remove, lam_reda, flux_mask_reda, scinum)
        
        Fnorm = [] # Normalized flux
        for i in range(len(fitted_fluxes)):
            Fnorm.append(float(fitted_fluxes[i]/smooth_cont[i])) # Divide raw flux by our continuum that we just fit
        
        for i in range(len(Fnorm)):
            if Fnorm[i] >= 1.3:
                if (verbose == True):
                    print("Found Fnorm >= 1.3:", Fnorm[i])
                Fnorm[i] = 1
        
        sf_sm = [] # sqrt(sky+flux)/smooth_fit
        for i in range(len(fitted_fluxes)):
            sf_sm.append(float(fitted_fluxes[i]+fitted_sky[i])**(1/2)/smooth_cont[i]) 
        
        if (verbose == True):
            print("Plotting Normalized Spectra")
            self.grapher.plot_sci_normalized(fitted_wavelengths, Fnorm, sf_sm, scinum)
        
        # Write a file for the spectra of this sci image
        # In the form G123456_61_OUT.fits in the OUT folder
        t = Table([fitted_wavelengths,Fnorm,sf_sm], names=('Lam', "Flux", "Sigma"))
        t.write(self.target_dir+"OUT/"+str(self.objid)+"_"+str(scinum)+"_OUT.fits", format='fits', overwrite=True)

#---------------------------------------------
    
    def combine_spectra(self, verbose):
        print("KARP is combining spectra")
        # Make a KARP_OUT_log file
        
        with open(self.target_dir+"OUT/KARP_OUT_log.txt", "w") as fo:
            fo.write("KARP is combining spectra\n")
        # Use glob to get everything in the folder
        file_names = glob.glob(self.target_dir+"OUT/*OUT.fits")
        if (verbose == True):
            print("Using these file names:",file_names)
            with open(self.target_dir+"OUT/KARP_OUT_log.txt", "a") as fo:
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
        with open(self.target_dir+"OUT/KARP_OUT_log.txt", "a") as fo:
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
        
        if (verbose == True):
            print("(RMS/sig_w55)",(RMS/sig_w55))
            with open(self.target_dir+"OUT/KARP_OUT_log.txt", "a") as fo:
                fo.write("(RMS/sig_w55)"+str(RMS/sig_w55)+"\n")
        
        
        self.grapher.plot_combined_norm(gwave, med_comb, sig_final, RMS)
        
        return gwave, med_comb, sig_final