#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:52:33 2025

@author: agleason
"""
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d, uniform_filter
from scipy.interpolate import interp1d
from astropy.nddata import CCDData
from astropy.table import Table
from functions import fit_cent_gaussian, G, build_fit_args, gaussian_integral_vec
from functions import format_fits_filename, y_lam, con_lam
from sci_tools import heliocentric_correction, fit_all_fluxes
from joblib import Parallel, delayed
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
        self.gwave = []
        self.med_comb = []
        self.sig_final = []
        
        self.grapher = grapher
        
        self.v_helio = heliocentric_correction(self.objRA, self.objDEC, self.otime)
        self.flux_raw_cor1_dict = {}
        self.wavelengths_dict = {}
        self.sky_raw_dict = {}
        self.sci_final_dict = {}
        self.cen_line_dict = {}
        self.lam_fit_dict = {}
        self.lam_fit_linenum_dict = {}
        self.argon = np.asarray(CCDData.read(format_fits_filename(self.dloc, self.argnum), unit="adu").data)
        
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
        
        #creates sci_final_1 if not already created
        if scinum in self.sci_final_dict:
            sci_final_1 = self.sci_final_dict[scinum]
        else:
            sci_final = np.asarray((ccdproc.subtract_bias(CCDData.read(sci_location,format = 'fits', unit = "adu"),masterbias).data)/final_flat)
            # Convert from ADU to electrons
            sci_final_1 = sci_final*0.6
            self.sci_final_dict[scinum] = sci_final_1
        
        if (verbose == True): #plot and write to disk
            self.grapher.plot_sci_final(sci_final, scinum)
            sci_final_write = CCDData(sci_final, unit="adu")
            sci_final_write.write(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"sci_final_"+str(scinum)+".fits", overwrite = True)
        
    #----------------------------
    #trace fitting
        
        #trace shouldn't change between aperature changes, brightest spot is center and is always in aperature
        #so save and attempt to get from dict to save expensive calculations
        if scinum in self.cen_line_dict:
            cen_line = self.cen_line_dict[scinum]
        else:
            cen_line = self.fit_trace(sci_final_1, verbose) #get center trace
            self.grapher.plot_cen_line(cen_line, scinum)
            self.cen_line_dict[scinum] = cen_line.copy() #save center trace
    #----------------------------------
    #flux extraction
        fit_args = build_fit_args(sci_final_1, cen_line, self.appw, self.buffw, self.bckw)
        sci1_fit = np.array(
            Parallel(n_jobs=-1)(delayed(fit_cent_gaussian)(*args) for args in tqdm(fit_args))
            )
    
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
            print("Plotting Gauss cen values and Gauss sig values vs Y pix")
            self.grapher.gauss_cen_plots(sci1_fit, cen_line, scinum)        
        
        # Make a plot of 20 evenly spaced pix
        # Showing the data, the apertures and buffers as lines,
        # And the over plotted G fit
        if (verbose == True):
            print("Making 20 aperture plots and aperature fit plots")
            self.grapher.make_aperature_fit_plots(sci1_fit, sci_final_1, cen_line, scinum)
            self.grapher.make_aperature_plots(sci1_fit, sci_final_1, cen_line, scinum)
            
        # Extract spectrum value from each y row
        print("Extracting flux...")
        # Now set down an aparature and get the flux in the center and another aperture to get the background level
        
        #mask out rows with nan gaussian fits
        sci1_fit = np.array(sci1_fit)
        valid_mask = ~np.isnan(sci1_fit[:, 0])  # 'a' values not NaN
        indices = np.where(valid_mask)[0]
        valid_fit = sci1_fit[valid_mask]
        valid_cen = cen_line[valid_mask]
        
        a, mu, sigma, bck = valid_fit.T
        c = valid_cen
        
        # Define fixed offset ranges from each center
        x1_ap = c - self.appw
        x2_ap = c + self.appw
        x1_br = c + self.appw + self.buffw
        x2_br = c + self.appw + self.buffw + self.bckw
        
        ap_flux = gaussian_integral_vec(a, mu, sigma, bck+1000, x1_ap, x2_ap) #adding 1000 to prevent negative bck values which cause the integral to sum the wrong part
        bkg_right = gaussian_integral_vec(a, mu, sigma, bck+1000, x1_br, x2_br)
        flux_valid = ap_flux - (((self.appw * 2) / self.bckw) * (bkg_right)) #note appw is not actual width, it is half width
        
        # Sky background is just the flat background level scaled to aperture width
        sky_valid = bck * (2 * self.appw)
        
        flux_raw1 = np.full_like(cen_line, np.nan, dtype=np.float64)
        flux_raw1[indices] = flux_valid
        sky_raw1 = np.full_like(cen_line, np.nan, dtype=np.float64)
        sky_raw1[indices] = sky_valid
        
        if (verbose == True):
            # Plot output raw flux (We can remove CRs later)
            self.grapher.plot_raw_flux(flux_raw1, scinum)
            for i in range(1, len(flux_valid), 10):
                print(f"Flux fitted value: {flux_valid[i]}")
                if (flux_valid[i] < 0):
                    print(f"Parameters: {a[i]} {mu[i]} {sigma[i]} {bck[i]}")
                    print(f"integrals: {ap_flux[i]} {bkg_right[i]}")
                    print(f"widths: {self.appw*2} {self.bckw}")
                    print(f"{c[i]}")
        
    #--------------------------------------
    #wavelength calibration
        
        pixc = [250,295,765,925,953,1275,1330,1396,1480,2870,3355,3421,3612,3780]
        lam = [6416.31,6384.72,6032.13,5912.09,5888.58,5650.70,5606.73,5558.70,5495.87,4510.73,4200.67,4158.59,4044.42,3948.97]
        
        # Lambda calibration
        print("Fitting lambda Gaussians...")
        
        lam_width = 8 # Width to fit wavelength gaussians
        lam_fit = [] # Stores fit parameters as [a, mu, sig]
        lam_fit_linenum = [] # Succesfully fitted line wavelength
        
        flux_raw1 = np.array(flux_raw1) 
        ii = flux_raw1 > 0
        flux_raw_cor1 = flux_raw1[ii]
        sky_raw1 = sky_raw1[ii]
        
        if scinum in self.lam_fit_dict:
            lam_fit = self.lam_fit_dict[scinum]
            lam_fit_linenum = self.lam_fit_linenum_dict[scinum]
        else:
            for i in range(len(pixc)):
                # We want the flux values around pixc
                a_col = self.argon[:, cen_line[i]] # Get every column value that is in cen_line
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
                except RuntimeError:
                    if (verbose == True):
                        print("Curve fit failed at maxfev=1000")
                        print("Trying to fit:",lam[i],pixc[i])
                        print("p0 is:",p0)
                        print("Now trying with maxfev=10000")
                    lam_width = 11
                    flux_vals = a_col[(pixc[i]-lam_width) : (pixc[i]+lam_width)] # Get argon values from the same pixels
                    loc_vals = np.arange((pixc[i]-lam_width),(pixc[i]+lam_width)) # fit to y pixel values of 5 of the center of the argon wavelength
                   
                    p0 = [np.max(flux_vals), pixc[i], 2.0, 0] # Guess that the gussian peak is the max height
                    popt, _ = curve_fit(G, loc_vals, flux_vals, p0=p0, bounds=bounds, maxfev=10000) # ignore covariance matrix spat out from curve_fit
                    lam_fit.append(popt)  # [a, mu, sigma, c] returend from curve_fit
                    lam_fit_linenum.append(lam[i]) # Wavelength values that where succesfully fit
            
            if (verbose == True):
                print("KARP fitted ", len(lam_fit), " lines")
                print("Argon Lines Fit Parameters:")
                with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                    f.write("KARP fitted"+str(len(lam_fit))+"lines\n")
                    f.write("Argon Lines Fit Parameters:\n")                
                
                s_lsp = [] # Sigmas of the lam line fits for the line split function
                for i in range(len(lam_fit)):
                    a, mu, sigma, bck = lam_fit[i]
                    s_lsp.append(sigma)
                    print("Lam Line Number:",i," lam:",lam[i]," a:",a," mu:",mu," sig:",sigma," bck:",bck)
                    with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                        f.write("Lam Line Number:"+str(i)+" lam:"+str(lam[i])+" a:"+str(a)+" mu:"+str(mu)+" sig:"+str(sigma)+" bck:"+str(bck)+"\n")
                
                print("Line Split Function:",np.mean(s_lsp))
                print("(avg of sig of line fits)")
                with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                    f.write("Line Split Function:"+str(np.mean(s_lsp))+"\n")
                    f.write("(avg of sig of line fits)\n") 
                
            self.lam_fit_dict[scinum] = lam_fit
            self.lam_fit_linenum_dict[scinum] = lam_fit_linenum
            
        cfits = [fit[1] for fit in lam_fit] #gets centers of each lambda fit
        
        # Fit a cubic poly nomial with cfits (actual line centers)
        # Fitting c fits (actual line centers in pixels) to lam_fit_linenum (successfully fit line wavelength values from lam)
        print("Fitting cubic polynomial")
        cfits_red = []
        for i in range(len(cfits)):
            cfits_red.append(cfits[i]/2000)
        y_lam_fit = np.polyfit(cfits_red,lam_fit_linenum,3)    
        #print(a,b,c,d) # As a*x^3+bx^2+c*x+d = lam(y)
        y_pix = np.arange(0,len(flux_raw_cor1),1)
        
        # Store y_lam(y_pix)=wavelengths
        # Heliocentric correction here as well (Should be carried over from y_lam)
        wavelengths = y_lam(y_pix, y_lam_fit, self.v_helio) # Wavelength of each pixel
        
        if (verbose == True):
            print("v_helio:",self.v_helio,"km/s") # -4 km/s for G093440
            self.grapher.plot_sci_lamcal(y_lam, y_pix, flux_raw_cor1, y_lam_fit, self.v_helio, scinum)
    
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
                del_lam.append(float(lam[i])-float(y_lam(cfits[i], y_lam_fit, self.v_helio)))  # The wavelength value we want to fit the line to MINUS the value our fit says that that line is at
            else:
                print("NAN here")
            #print(offset[i])
            
        if (verbose == True):
            # Put print RMS here
            dlS = []
            for i in range(len(del_lam)):
                dlS.append(del_lam[i]**2)
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
        
        if (verbose == True):
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
            self.grapher.lam_cal_res(pixc, del_lam, scinum)
            self.grapher.lam_cal_res_vel(pixc, res_vel, scinum)
            
            print("Mean Wavelength residual (delLAM):",np.mean(del_lam))
            print("Mean Velocity residual (delVEL):",np.mean(res_vel))
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                f.write("Mean Wavelength residual (delLAM):"+str(np.mean(del_lam))+"\n")
                f.write("Mean Velocity residual (delVEL):"+str(np.mean(res_vel))+"\n")
        
            self.grapher.plot_sci_lamcal_res(y_lam, y_pix, flux_raw_cor1, rnErr, y_lam_fit, self.v_helio, scinum)
        
            # Print dispersion, as (number of angstroms we go over/number of pixels (4096))
            # Recall that our lam is fit inverse of our pixels, ie 246 pix = 6000 Angstroms
            lam_out_max = y_lam(np.min(y_pix), y_lam_fit, self.v_helio) # Max wavelength as a function of pixel, ie the wavelength fit to the last pixel
            lam_out_min = y_lam(np.max(y_pix), y_lam_fit, self.v_helio) # Minimum wavelength as a function of pixel, ie the wavelength fit to the first pixel
            
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
            
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"wavelengths_"+str(scinum)+".txt", "w") as file:
                for wave in wavelengths:
                    file.write(f"{wave}\n")
        
            # Store raw sky values
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"skyraw_"+str(scinum)+".txt", "w") as file:
                for sky in sky_raw1:
                    file.write(f"{sky}\n")
                    
        #save data to class dictionaries for easy loading
        self.flux_raw_cor1_dict[scinum] = flux_raw_cor1
        self.sky_raw_dict[scinum] = sky_raw1
        self.wavelengths_dict[scinum] = wavelengths
    
#-----------------------------------------

    def normalization(self, scinum, verbose, optimize):        
        # Mask out deep absorption features
        # Deep features in order of ascending wavelength, in angstroms
        # We have spectra down to Lam = 3780 and up to 6600 angstroms
        # H_8,H_eta, H_zeta,Ca_K,Ca_H, H_epsilon,H_delta,H_gamma, H_beta, SodiumD2, SodiumD1, H_alpha
        features = [3796.94,3835.40,3889.06,3933.7,3969,3970.075,4101.73,4340.47,4861.35,5889.96,5895.93,6562.81]
        
        flux_raw_cor1 = list(self.flux_raw_cor1_dict[scinum])
        wavelengths = list(self.wavelengths_dict[scinum])
        sky_raw1 = list(self.sky_raw_dict[scinum])
        
        # Remove 4 pixels from either edge to ensure bad chip pixels aren't affecting our fits
        flux_raw_cor1 = flux_raw_cor1[3:-4]
        wavelengths = wavelengths[3:-4]
        sky_raw1 = sky_raw1[3:-4]
        flux_masked = np.array(flux_raw_cor1.copy()) # Create a copy so we can still use flux_raw_cor1
        flux_raw_cor1_np = np.array(flux_raw_cor1, dtype=float)
        wavelengths_np = np.array(wavelengths, dtype=float)
        sky_raw1 = np.array(sky_raw1)
        
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
            else:
                # Pixels that we take out
                flux_mask_remove.append(flux_raw_cor1[i])
                removed_index.append(1) # The index values of the pixels that we remove for fitting later
                lam_remove.append(wavelengths[i])
        
        # Fit a continuum to out spectra:
        fit_params = np.polyfit(lam_red,flux_mask_red,6)    
        Fnorm_Es = flux_raw_cor1_np / con_lam(wavelengths_np, fit_params) # Divide raw flux by our continuum that we just fit
        
        # Plot our first estimate on the normalized flux
        if (verbose == True):
            con_lam_test = []
            for i in range(len(wavelengths)):
                con_lam_test.append(con_lam(wavelengths_np[i], fit_params)) 
            self.grapher.plot_sci_flux_norm_est(wavelengths, Fnorm_Es, scinum)
            self.grapher.plot_sci_flux_masked(lam_red, lam_remove, flux_mask_red, flux_mask_remove, wavelengths, con_lam_test, scinum)
        
        print("Line-fit normilization")
        removed_index = np.array(removed_index)
        
        fitted_values = []
        fitted_wavelengths = []
        fitted_sky = []
        fitted_fluxes = []
        
        flux_masked_global = flux_raw_cor1_np.copy()
        for i, feat in enumerate(features):
            center = np.argmin(np.abs(wavelengths_np - feat)) # Use argmin to get the index of the minimum value of all_wavelengths-all_features, (i.e. where the center of each feature is)
            start = max(0, center - feature_mask[i]) # From the center out to our feature mask size, start as far out from the edge as we can
            stop = min(len(flux_masked_global), center + feature_mask[i]) # From the center out to our feature mask size,  start as close to the edge as we can
            flux_masked_global[start:stop] = np.nan # Mask absorption lines with NANs
        
        lam_reda = [] # Kept flux values that do not have Fnorm_es > 1.3
        flux_mask_reda = []
        lam_removea = [] # Removed flux values that do not have Fnorm_es > 1.3
        flux_mask_removea = []
        
        rm_param = 1.3
        include = Fnorm_Es <= rm_param
        
        results = fit_all_fluxes(Fnorm_Es, wavelengths_np, flux_masked, include, removed_index,
                           flux_raw_cor1_np, flux_masked_global, sky_raw1, self.norm_line_width)
        for row in results:        
            if np.isnan(row[0]):
                continue
            else:
                val, sky, flux, wave, f_keep, f_remove, l_keep, l_remove = row
                fitted_values.append(val)
                fitted_sky.append(sky)
                fitted_fluxes.append(flux)
                fitted_wavelengths.append(wave)
                if f_keep is not None:
                    flux_mask_reda.append(f_keep)
                    lam_reda.append(l_keep)
                if f_remove is not None:
                    flux_mask_removea.append(f_remove)
                    lam_removea.append(l_remove)
            
        # Now smooth this fit with a moving boxcar
        smooth_cont = uniform_filter(np.array(fitted_values), size=self.norm_line_boxcar)
        
        if optimize == False:
            # Plot fitted flux values from the line fit to see what is going on
            self.grapher.plot_fit_over_flux(fitted_wavelengths, smooth_cont, fitted_values, fitted_fluxes, lam_removea, flux_mask_removea, lam_red, lam_remove, flux_mask_red, flux_mask_remove, lam_reda, flux_mask_reda, scinum)
                
        Fnorm = np.array(fitted_fluxes) / smooth_cont # Divide raw flux by our continuum that we just fit
        if verbose:
            flagged = Fnorm >= 1.3
            print("Found Fnorm >= 1.3:", Fnorm[flagged])
        Fnorm = np.where(Fnorm >= 1.3, 1.0, Fnorm)
        
        sf_sm = np.sqrt(np.array(fitted_fluxes) + np.array(fitted_sky)) / smooth_cont
        
        if (verbose == True):
            print("Plotting Normalized Spectra")
            self.grapher.plot_sci_normalized(fitted_wavelengths, Fnorm, sf_sm, scinum)
        
        # Write a file for the spectra of this sci image
        # In the form G123456_61_OUT.fits in the OUT folder
        t = Table([fitted_wavelengths,Fnorm,sf_sm], names=('Lam', "Flux", "Sigma"))
        t.write(self.target_dir+"OUT/"+str(self.objid)+"_"+str(scinum)+"_OUT.fits", format='fits', overwrite=True)

#---------------------------------------------
    
    def combine_spectra(self, verbose):
        jackknife = False
        
        print("KARP is combining spectra")
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
        #initialize variables
        gsig = []
        med_comb = []
        rms = 0
        gwave = wave_ref
        
        def compute_snr_from_files(leaveout_indices):
            nonlocal gsig
            nonlocal med_comb
            nonlocal rms
            
            # Leave out one file
            selected_files = [f for j, f in enumerate(file_names) if j not in leaveout_indices]
        
            # Interpolate onto reference grid
            gflux_interp = []
            for fname in selected_files:
                t = Table.read(fname)
                f = interp1d(t["Lam"], t["Flux"], bounds_error=False, fill_value=np.nan)
                s = interp1d(t["Lam"], t["Sigma"], bounds_error=False, fill_value=np.nan)
                gflux_interp.append(f(wave_ref))
                gflux_array = np.array(gflux_interp)
                gsig.append(s(wave_ref))
        
            # Stack and take median, ignoring NaNs
            gflux_array = np.array(gflux_interp)
            med_comb = np.nanmedian(gflux_array, axis=0)
            
            # Remove zeroth order line and any remaning outliers from bad pixels
            for i in range(len(gwave)):
                if 5500 <= gwave[i] <= 5700:
                    if med_comb[i] > 1.05:
                        med_comb[i] = 1
                        
            for i in range(len(med_comb)):
                if med_comb[i] > 1.25:
                    med_comb[i] = 1
        
            print("Computing RMS")
            # Compute RMS in 5450–5550
            mask = (gwave >= 5450) & (gwave <= 5550) & (~np.isnan(med_comb))
            residuals_squared = (1 - med_comb[mask])**2
            rms = np.sqrt(np.mean(residuals_squared))
            
            # Calculate and print SNR at 5500A
            print("SNR at 5500:",1/rms)
            with open(self.target_dir+"OUT/KARP_OUT_log.txt", "a") as fo:
                fo.write("SNR at 5500:"+str(1/rms)+"\n")
                if jackknife == True:
                    print(f"with files {selected_files}")
                    fo.write(f"with files {selected_files}\n")
        
        if jackknife == True:
            n_files = len(file_names)
            for i in range(n_files):
                leaveout_indices = [i] #leaves out a single file and calculates SNR
                compute_snr_from_files(leaveout_indices)
        else: #do all files
            leaveout_indices = []
            compute_snr_from_files(leaveout_indices)
        
        gsig = np.array(gsig)
        
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
        sig_final = rms * np.array(sig_w)/sig_w55
        
        if (verbose == True):
            print("(RMS/sig_w55)",(rms/sig_w55))
            with open(self.target_dir+"OUT/KARP_OUT_log.txt", "a") as fo:
                fo.write("(RMS/sig_w55)"+str(rms/sig_w55)+"\n")
        
        self.grapher.plot_combined_norm(gwave, med_comb, sig_final, rms)
        
        self.gwave = gwave
        self.med_comb = med_comb
        self.sig_final = sig_final
        
        return 1/rms
    
#-----------------------------------------
    
    def compare_cen_lines(self, scinum1, scinum2):
        if scinum1 not in self.cen_line_dict or scinum2 not in self.cen_line_dict:
            print("One or both science image centerlines not found.")
            return None
    
        cl1 = self.cen_line_dict[scinum1]
        cl2 = self.cen_line_dict[scinum2]
    
        min_len = min(len(cl1), len(cl2))
        cl1 = cl1[:min_len]
        cl2 = cl2[:min_len]
    
        delta = cl2 - cl1
        abs_diff = np.abs(delta)
        mean_diff = np.mean(abs_diff)
        max_diff = np.max(abs_diff)
    
        print(f"Comparison between image {scinum1} and {scinum2}:")
        print(f"  Mean absolute difference: {mean_diff:.3f} pixels")
        print(f"  Max absolute difference: {max_diff:.3f} pixels")
    
        return delta
    
    def fit_trace(self, sci_final_1, verbose):
        n_rows = sci_final_1.shape[0] #gets row num
        cen_line = np.round(np.linspace(self.clmin, self.clmax, n_rows)).astype(int)
        # Note that this repeats each integer 51 times, stretching it to 1124
        
        # Fit Gaussian to each row in sci_final_1 using current cen_line estimate
        print("KARP fitting centerline")
        
        # Fit and collect results in a list (each entry is a tuple: a, mu, sig, bck)
        fit_args = build_fit_args(sci_final_1, cen_line, self.appw, self.buffw, self.bckw)
        cen_fit = np.array(
            Parallel(n_jobs=-1)(delayed(fit_cent_gaussian)(*args) for args in tqdm(fit_args))
            )
        
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
        cen_line = np.round(cen_line).astype(int)
        
        return cen_line