'''
created by Alex Gleason

aux grapher module for KARP
'''

#import packages
from astropy.nddata import CCDData
import matplotlib.pyplot as plt
import numpy as np
import os

# Define a Gaussian
def G(x, a, mu, sigma, bck):
	return (a * np.exp(-(x-mu)**2/(2*sigma**2))) + bck
	# A 4d Gaussian

#fit individual plots for each metal line
def metal_line_plt(metal_fits, gwave, mask, med_comb, lines, target_dir):
    for i, popt in enumerate(metal_fits):
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        ax.scatter(gwave,med_comb, s=5,color="black")
        ax.plot(gwave,G(gwave,popt[0],popt[1],popt[2], popt[3]))
        ax.axhline(1,color="red",linestyle="--")
        ax.axvline(popt[1],color="black",linestyle="--")
        ax.set_xlim(int(lines[i]-5),int(lines[i]+5))
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Median Normalized Flux")
        ax.set_title("Metal line fit at "+str(lines[i])+" A")
        '''
        #set y scale for each plot
        ydata = med_comb[mask]
        ymin = np.nanmin(ydata)
        ymax = np.nanmax(ydata)
        yrange = ymax - ymin
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        '''
        plt.tight_layout(rect=[0.06, 0.06, 1, 1])  # Leave space for global labels
        plt.savefig(str(target_dir) + f"OUT/{lines[i]}_metal_line.png", dpi=300)

def metal_line_big_plt(metal_fits, metal_lines, metal_mask, gwave, med_comb, target_dir, objid):
    n_lines = len(metal_lines)
    ncols = 4
    nrows = int(np.ceil(n_lines / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3), sharey=False)
    axs = axs.flatten()

    for i, popt in enumerate(metal_fits):
        ax = axs[i]
        mask = (gwave > metal_lines[i] - metal_mask[i]) & (gwave < metal_lines[i] + metal_mask[i])
        ax.scatter(gwave, med_comb, s=5, color="black")
        ax.plot(gwave, G(gwave, *popt), color="orange")
        ax.axhline(1, color="red", linestyle="--", linewidth=0.8)
        ax.axvline(popt[1], color="blue", linestyle="--", linewidth=0.8)
        ax.set_xlim(metal_lines[i] - 5, metal_lines[i] + 5)
        ax.set_title(f"{metal_lines[i]:.1f} Å")
        # Remove individual axis labels for clarity
        ax.set_xlabel("")
        ax.set_ylabel("")
        #set y scale for each plot
        ydata = med_comb[mask]
        ymin = np.nanmin(ydata)
        ymax = np.nanmax(ydata)
        yrange = ymax - ymin
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
    
    # Hide unused subplots
    for j in range(i+1, len(axs)):
        axs[j].axis('off')

    fig.text(0.55, 0.04, "Wavelength (A)", ha='center', fontsize=16)
    fig.text(0.04, 0.5, "Normalized Flux", va='center', rotation='vertical', fontsize=16)

    plt.tight_layout(rect=[0.06, 0.06, 1, 1])  # Leave space for global labels
    plt.savefig(str(target_dir) + f"OUT/{objid}_metal_lines_all.png", dpi=300)
    plt.close(fig)

def plot_vel_lines(vel_fit, vel_fit_linenum, vel_lines, gwave, med_comb, wout, fout, vel_mask, target_dir, objid):
    for i in range(len(vel_fit)):
        a, mu, sig, bck = vel_fit[i]
        print("For line:",vel_fit_linenum[i]+1,"KARP fit:",mu,"Angstroms")
        print("a:",a,"sig:",sig,"bck:",bck)
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
        
def plot_masterbias(masterbias, target_dir, scinum, objid):
    plt.clf()
    plt.imshow(masterbias)
    plt.colorbar()
    plt.title('Master Bias Image')
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+objid+"MBias_"+str(scinum)+".png")
    print("Master Bias Made")
    
def plot_masterflat(masterflat, target_dir, scinum, objid):
    plt.clf()
    plt.imshow(masterflat)
    plt.colorbar()
    plt.title('Master Flat Image')
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+objid+"_MFlat_"+str(scinum)+".png")
    print("Master Flat Made")

def plot_masterflatDEbias(masterflatDEbias, target_dir, scinum, objid):
    plt.clf()
    plt.imshow(masterflatDEbias)
    plt.colorbar()
    plt.title('Master Flat sub Bias Image')
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+objid+"_MF_MB_"+str(scinum)+".png")
    print("Master Flat-Bias Made")
    
def plot_smooth_mf_mb(smooth_mf_mb, target_dir, scinum, objid):
    plt.clf()
    plt.imshow(smooth_mf_mb)
    plt.colorbar()
    plt.title('Smoothed MF_MB Image (5x5 Boxcar)')
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+objid+"_smoothed_MF_MB_"+str(scinum)+".png")
    print("Smoothed MF_MB Made")
    
def plot_final_flat(final_flat, target_dir, scinum, objid):
    plt.clf()
    plt.imshow(final_flat)
    plt.colorbar()
    plt.title('Final Flat')
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+objid+"_final_flat.png")
    final_flat_write = CCDData(final_flat, unit="adu")
    final_flat_write.write(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"final_flat_"+str(scinum)+".fits", overwrite = True)
    
def plot_sci_final(sci_final, target_dir, scinum, objid):
    plt.clf()
    plt.imshow(sci_final)
    plt.colorbar()
    plt.title('Final Science ' + str(scinum))
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+objid+"_final_sci" + str(scinum)+".png")
    print("Master Science Number: " + str(scinum) + " Made")
    sci_final_write = CCDData(sci_final, unit="adu")
    sci_final_write.write(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"sci_final_"+str(scinum)+".fits", overwrite = True)
    
def gauss_cen_plots(sci1_fit, cen_line, target_dir, scinum):

    # ---- Ensure target directory exists ----
    save_dir = os.path.join(str(target_dir), f"ImageNumber_{scinum}")
    os.makedirs(save_dir, exist_ok=True)

    # ---- Extract valid data in one pass ----
    valid_data = [(i, mu, sigma) for i, (a, mu, sigma, bck) in enumerate(sci1_fit) if not np.isnan(a)]

    if valid_data:
        indices, mus, sigmas = zip(*valid_data)
        
        # ---- Plot 1: mu vs index with cen_line ----
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(indices, mus, s=1, color='blue', label='mu')
        ax1.scatter(np.arange(len(cen_line)), cen_line, s=3, color='black', label='cen_line')
        ax1.tick_params(labelsize=14)
        ax1.set_xlabel("Y Pix", size=14)
        ax1.set_ylabel("G fit mu", size=14)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig(os.path.join(save_dir, f"Sci_Gcen_y_Cenline_{scinum}.png"))

        # ---- Plot 2: sigma vs index ----
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(indices, sigmas, color='green')
        ax2.tick_params(labelsize=14)
        ax2.set_xlabel("Y Pix", size=14)
        ax2.set_ylabel("G fit Sig", size=14)
        fig2.tight_layout()
        fig2.savefig(os.path.join(save_dir, f"Sci_Gsig_y_{scinum}.png"))

        # ---- Plot 3: zoomed-in sigma plot ----
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.scatter(indices, sigmas, color='green')
        ax3.tick_params(labelsize=14)
        ax3.set_xlabel("Y Pix", size=14)
        ax3.set_ylabel("G fit Sig", size=14)
        ax3.set_ylim(0, 2)
        fig3.tight_layout()
        fig3.savefig(os.path.join(save_dir, f"Sci_Gsig_y_ZOOM{scinum}.png"))
        
def lam_cal_res(pixc, del_lam, target_dir, scinum):
    fig, axL = plt.subplots(1, 1, figsize=(8,6))
    
    axL.scatter(pixc,del_lam,color="blue")
    axL.axhline(0,linestyle="--",color="black")
    axL.set_xlabel("Y Pixel Value")
    axL.set_ylabel("Angstroms")
    
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"LamCalRes_"+str(scinum)+".png")
    
def lam_cal_res_vel(pixc, res_vel, target_dir, scinum):
    plt.cla()
    fig, axV = plt.subplots(1, 1, figsize=(8,6))
    axV.scatter(pixc,res_vel,color="green")
    axV.axhline(0,linestyle="--",color="black")
    axV.set_xlabel("Y Pixel Value")
    axV.set_ylabel("km/s")
    
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"LamCalRes_Velocity_"+str(scinum)+".png")

def plot_cen_line(cen_line, target_dir, scinum):
    fig, axcs = plt.subplots(1, 1, figsize=(8,6))
    censcat = np.arange(0,len(cen_line),1)
    axcs.scatter(censcat,cen_line, s=3)
    axcs.set_xlabel("Y Pixel")
    axcs.set_ylabel("X Pixel")
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"cen_line_"+str(scinum)+".png")
    print("len(cen_line)",len(cen_line))
    
def make_aperature_fit_plots(sci1_fit, sci_final_1, cen_line, a_width, buff_width, bckrnd_width, target_dir, scinum):
    # Prepare save directory once
    save_dir = os.path.join(str(target_dir), f"ImageNumber_{scinum}", f"E20Fits_{scinum}")
    os.makedirs(save_dir, exist_ok=True)

    # Loop through every 200th index up to 4000
    for i in range(0, 4000, 200):
        a, mu, sigma, bck = sci1_fit[i]
        rowin = sci_final_1[i]
        cen = cen_line[i]

        # Compute useful slice indices once
        left = cen - a_width - buff_width - bckrnd_width
        right = cen + a_width + buff_width + bckrnd_width + 1
        fit_left = cen - a_width
        fit_right = cen + a_width

        x_vals = np.arange(left, right)
        y_vals = rowin[left:right]

        # Create figure and axes explicitly
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.step(x_vals, y_vals, where='mid', color="black")
        ax.axvline(cen, color="black", linestyle="--")
        
        # Fit curve
        x_fit = np.linspace(fit_left, fit_right, 300)
        ax.plot(x_fit, G(x_fit, a, mu, sigma, bck), color="purple")
        ax.set_xlim(fit_left, fit_right)
        ax.set_xlabel("X pixels")
        ax.set_ylabel("Counts (e⁻)")

        # Save and close
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"E20Fit{i}.png"))
        plt.close(fig)  # Prevent accumulation of open figures

def make_aperature_plots(sci1_fit, sci_final_1, cen_line, a_width, buff_width, bckrnd_width, target_dir, scinum):
    save_dir = os.path.join(str(target_dir), f"ImageNumber_{scinum}", f"E20Data_{scinum}")
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(0, 4000, 200):
        a, mu, sigma, bck = sci1_fit[i]
        rowin = sci_final_1[i]
        center = cen_line[i]
    
        # Compute all x-range boundaries
        left_edge = center - a_width - buff_width - bckrnd_width
        right_edge = center + a_width + buff_width + bckrnd_width + 1
    
        x_vals = np.arange(left_edge, right_edge)
        y_vals = rowin[left_edge:right_edge]
    
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.step(x_vals, y_vals, where='mid', color="black")
    
        # Draw boundary lines
        ax.axvline(center, color="black", linestyle="--")  # Center line
        ax.axvline(center - a_width, color="blue")         # Signal region start
        ax.axvline(center + a_width, color="blue")         # Signal region end
        ax.axvline(center - a_width - buff_width, color="red")        # Buffer start
        ax.axvline(center + a_width + buff_width, color="red")        # Buffer end
        ax.axvline(left_edge, color="green")                           # Background start
        ax.axvline(center + a_width + buff_width + bckrnd_width, color="green")  # Background end
    
        # Plot invisible Gaussian for centering (alpha=0)
        x_fit = np.linspace(center - a_width, center + a_width, 300)
        ax.plot(x_fit, G(x_fit, a, mu, sigma, bck), color="purple", alpha=0)
    
        # Set axis limits and labels
        ax.set_xlim(left_edge - 1, right_edge + 1)
        ax.set_xlabel("X pixels")
        ax.set_ylabel("Counts (e⁻)")
        ax.tick_params(labelsize=14)
    
        # Save and clean up
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"EData{i}.png"))
        plt.close(fig)
        
def plot_raw_flux(flux_raw1, target_dir, scinum):
    plt.cla()
    y_pix = np.arange(0,len(flux_raw1),1)
    plt.scatter(y_pix,flux_raw1, s=1)
    plt.xlabel("Y Pixel")
    plt.ylabel("Background subtracted Flux")
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_bsub_"+str(scinum)+".png", dpi=300)
    
def plot_sci_lamcal(y_lam, y_pix, flux_raw_cor1, target_dir, scinum):
    fig, ax2 = plt.subplots(1, 1, figsize=(8,6))
    ax2.scatter(y_lam(y_pix),flux_raw_cor1, s=1)
    ax2.axvline(x=6562.81,color='red')
    ax2.axvline(x=4861.35,color='cyan')
    ax2.axvline(x=4340.47,color='blueviolet')
    plt.xlabel("Wavelength (A)")
    plt.ylabel("Background subtracted Flux (Electrons)")
    plt.ylim(0,30000)
    plt.title("Sci Image"+str(scinum))
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_LamCal_"+str(scinum)+".png", dpi=300)

def plot_sci_lamcal_res(y_lam, y_pix, flux_raw_cor1, rnErr, target_dir, scinum):
    plt.cla() # Clear plt to prevent over plotting
    fig, axE = plt.subplots(1, 1, figsize=(8,6))
    axE.scatter(y_lam(y_pix),flux_raw_cor1, s=1)
    axE.scatter(y_lam(y_pix),rnErr, s=1,color='red')
    axE.set_xlabel("Wavelength (A), Error (10x)")
    axE.set_ylabel("Background subtracted Flux (Electrons)")
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_LamCal_Res_"+str(scinum)+".png", dpi=300)   
    
def plot_sci_flux_masked(lam_red, lam_remove, flux_mask_red, flux_mask_remove, wavelengths, con_lam_test, target_dir, scinum):
    plt.cla() # Clear plt to prevent over plotting
    fig, axE = plt.subplots(1, 1, figsize=(8,6))
    axE.scatter(lam_red,flux_mask_red, s=1, color="green")
    axE.scatter(lam_remove,flux_mask_remove, s=1, color="red")
    axE.set_xlim(3750,4000)
    axE.set_xlabel("Wavelength (A)")
    axE.set_ylabel("Background subtracted Flux (Electrons)")
    
    axE.scatter(wavelengths,con_lam_test, s=1, color="black")
    # 6 Zoomed in sci_flux_masked plots
    for i in range(0,3000,500):
        axE.set_xlim((3700+i),(4200+i))
        os.makedirs(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_Masked_All_"+str(scinum), exist_ok=True) # Initializes a directory for E20 files
        plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/Sci_Flux_Masked_All_"+str(scinum)+"/Sci_Flux_Masked_All_i"+str(scinum)+"_"+str(i)+".png", dpi=300)   

def plot_sci_flux_norm_est(wavelengths, Fnorm_Es, target_dir, scinum):
    plt.cla()
    fig, axCont = plt.subplots(1, 1, figsize=(8,6))
    axCont.scatter(wavelengths,Fnorm_Es, s=1,color="green")
    axCont.set_xlabel("Wavelength (A)")
    axCont.set_ylabel("Estimate Cont. Normalized Flux")
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_NormEst_"+str(scinum)+".png", dpi=300)   

def plot_fit_over_flux(fitted_wavelengths, smooth_cont, fitted_values, fitted_fluxes, lam_removea, flux_mask_removea, lam_red, lam_remove, flux_mask_red, flux_mask_remove, lam_reda, flux_mask_reda, target_dir, scinum):
    plt.cla()
    fig, axL = plt.subplots(1, 1, figsize=(8,6))
    axL.scatter(fitted_wavelengths,smooth_cont, s=1,color="blue")
    axL.scatter(fitted_wavelengths,fitted_values, s=0.9,color="green")
    
    axL.scatter(fitted_wavelengths,fitted_fluxes, s=0.1,color="black")
    axL.scatter(lam_removea,flux_mask_removea,s=0.1,color="red")
    axL.set_xlabel("Wavelength (A)")
    axL.set_ylabel("Fitted Values over Flux")
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"FitOverFlux_"+str(scinum)+".png", dpi=300)   
    
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
        os.makedirs(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"FitOverFluxZoom_"+str(scinum), exist_ok=True) # Initializes a directory for E20 files
        plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"FitOverFluxZoom_"+str(scinum)+"/FitOverFlux_iter_"+str(scinum)+"_"+str(i)+".png", dpi=300)   
    
def plot_sci_normalized(fitted_wavelengths, Fnorm, sf_sm, target_dir, objid, scinum):
    plt.cla()
    fig, axNorm = plt.subplots(1, 1, figsize=(8,6))
    axNorm.scatter(fitted_wavelengths,Fnorm, s=1,color="black")
    axNorm.scatter(fitted_wavelengths,sf_sm, s=0.5,color="red")
    axNorm.axhline(1,color="red",linestyle="--")
    axNorm.set_xlabel("Wavelength (A)")
    axNorm.set_ylabel("Normalized Flux")
    axNorm.set_title("Normalized Spectra for "+str(objid)+" Image:"+str(scinum))
    axNorm.set_ylim(0,1.5)
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"Sci_Normalized_"+str(scinum)+".png", dpi=300)   
    
    # Some other diagnostic plots
    plt.cla()
    fig, axNorm = plt.subplots(1, 1, figsize=(8,6))
    axNorm.scatter(fitted_wavelengths[3000:4096],Fnorm[3000:4096], s=1,color="black")
    axNorm.set_xlim(3760,4400)
    axNorm.axhline(1,color="red",linestyle="--")
    axNorm.set_xlabel("Wavelength (A)")
    axNorm.set_ylabel("Normalized Flux")
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"Sci_Normalized_Left"+str(scinum)+".png", dpi=300)   
    
    
    plt.cla()
    fig, axNorm = plt.subplots(1, 1, figsize=(8,6))
    axNorm.scatter(fitted_wavelengths[2700:3300],Fnorm[2700:3300], s=1,color="black")
    axNorm.set_xlabel("Wavelength (A)")
    axNorm.set_ylabel("Normalized Flux")
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+"Sci_Normalized_hbeta"+str(scinum)+".png", dpi=300)   

def plot_combined_norm(gwave, med_comb, sig_final, wout, fout, eout, RMS, target_dir, objid):
    plt.cla()
    fig, axNorm = plt.subplots(1, 1, figsize=(8,6))
    axNorm.scatter(gwave,med_comb, s=0.8,color="black")
    axNorm.axhline(1,color="red",linestyle="--")
    axNorm.set_xlabel("Wavelength (A)")
    axNorm.set_ylabel("Median Normalized Flux")
    axNorm.set_ylim(-0.1,1.5)
    
    axNorm.scatter(gwave,sig_final, s=0.3,color="red")
    axNorm.set_title("Median Normalized Spectra for "+str(objid)+", SNR 5500 A:"+str(np.round(1/RMS, decimals=2)))
    
    axNorm.scatter(wout,fout,s=0.6,color="green")
    axNorm.scatter(wout,eout,s=0.3,color="blue")
    
    plt.savefig(str(target_dir)+"OUT/"+str(objid)+"_OUT.png", dpi=300)
    
    for i in range(0,2600,40):
        axNorm.set_xlim((3700+i),(4200+i))
        plt.savefig(str(target_dir)+"OUT/"+str(objid)+"_OUT_Zoom"+str(i)+".png", dpi=300)