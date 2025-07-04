'''
created by Alex Gleason

aux grapher module for KARP
'''

#import packages
from functions import string_to_bool
import matplotlib.pyplot as plt
import numpy as np
import os

# Define a Gaussian
#different from functions.G bc no njit
def G(x, a, mu, sigma, bck):
	return (a * np.exp(-(x-mu)**2/(2*sigma**2))) + bck
	# A 4d Gaussian
    
class grapher:
    def __init__ (self, params):
        self.objid = str(params["objid"]) # Object ID
        self.target_dir = str(params["target_dir"])
        self.verbose = string_to_bool(params["verbose"])
        self.appw = int(np.round(float(params["appw"])-1)/2) # So an appature width inputu of 13 is two 6 pixel sides, and if you put 14 it rounds to 13
        self.buffw = int(params["buffw"]) # Width of the buffer in pixels (on either side of the appature, so 4 is two, 4 pix buffers su,etrically spaced around the center line)
        self.bckw = int(params["bckw"]) # The width of the background in pixels, (on either side of the buffer)

    #fit individual plots for each metal line
    def metal_line_plt(self, metal_fits, gwave, masks, med_comb, lines, center_shifts):
        for i, popt in enumerate(metal_fits):
            plt.clf()
            fig, ax = plt.subplots(1, 1, figsize=(8,6))
            ax.scatter(gwave,med_comb, s=5,color="black")
            ax.plot(gwave,G(gwave,popt[0],popt[1],popt[2], popt[3]))
            ax.axhline(1,color="red",linestyle="--")
            ax.axvline(popt[1],color="black",linestyle="--")
            ax.axvline(lines[i],color="blue",linestyle="--")
            ax.axvline(lines[i]+center_shifts[i], color="green", linestyle="--")
            ax.set_xlim(int(lines[i]-5+center_shifts[i]),int(lines[i]+5)+center_shifts[i])
            ax.set_xlabel("Wavelength (A)")
            ax.set_ylabel("Median Normalized Flux")
            ax.set_title("Metal line fit at "+str(lines[i])+" A")
            
            #set y scale for each plot
            ydata = med_comb[masks[i]]
            try: #safeguard for data out of range
                ymin = np.nanmin(ydata)
                ymax = np.nanmax(ydata)
            except ValueError:
                ymin = 1
                ymax = 1
            ax.set_ylim(min(.9, ymin-0.05), max(ymax+0.05, 1.1))
            
            plt.tight_layout(rect=[0.06, 0.06, 1, 1])  # Leave space for global labels
            plt.savefig(self.target_dir + f"OUT/{lines[i]}_metal_line.png", dpi=300)
            ax.clear()

    def metal_line_big_plt(self, metal_fits, metal_lines, masks, gwave, med_comb, center_shifts):
        n_lines = len(metal_lines)
        ncols = 4
        nrows = int(np.ceil(n_lines / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3), sharey=False)
        axs = axs.flatten()
    
        for i, popt in enumerate(metal_fits):
            ax = axs[i]
            ax.scatter(gwave, med_comb, s=5, color="black")
            ax.plot(gwave, G(gwave, *popt), color="orange")
            ax.axhline(1, color="red", linestyle="--", linewidth=0.8)
            ax.axvline(popt[1], color="blue", linestyle="--", linewidth=0.8)
            ax.axvline(metal_lines[i],color="blue",linestyle="--")
            ax.set_xlim(int(metal_lines[i]-5+center_shifts[i]),int(metal_lines[i]+5)+center_shifts[i])
            ax.set_title(f"{metal_lines[i]:.1f} Å")
            # Remove individual axis labels for clarity
            ax.set_xlabel("")
            ax.set_ylabel("")
            #set y scale for each plot
            ydata = med_comb[masks[i]]
            try: #safeguard for data out of range
                ymin = np.nanmin(ydata)
                ymax = np.nanmax(ydata)
            except ValueError:
                ymin = 1
                ymax = 1
            ax.set_ylim(min(.9, ymin-0.05), max(ymax+0.05, 1.1))
        
        # Hide unused subplots
        for j in range(i+1, len(axs)):
            axs[j].axis('off')
    
        fig.text(0.55, 0.04, "Wavelength (A)", ha='center', fontsize=16)
        fig.text(0.04, 0.5, "Normalized Flux", va='center', rotation='vertical', fontsize=16)
    
        plt.tight_layout(rect=[0.06, 0.06, 1, 1])  # Leave space for global labels
        plt.savefig(self.target_dir + f"OUT/{self.objid}_metal_lines_all.png", dpi=300)
        plt.close(fig)
    
    def plot_vel_lines(self, vel_fit, vel_fit_linenum, vel_lines, gwave, med_comb, vel_mask):
        for i in range(len(vel_fit)):
            a, mu, sig, bck = vel_fit[i]
            print("For line:",vel_fit_linenum[i]+1,"KARP fit:",mu,"Angstroms")
            print("a:",a,"sig:",sig,"bck:",bck)
            print("(lam-lam_0)/lam_0:",((float(mu)-float(vel_lines[vel_fit_linenum[i]]))/float(vel_lines[vel_fit_linenum[i]])),"A")
            print("mu in km/s:",((float(mu)-float(vel_lines[vel_fit_linenum[i]]))/float(vel_lines[vel_fit_linenum[i]]))*3*10**5)
            
            plt.cla()
            fig, axVel = plt.subplots(1, 1, figsize=(8,6))
            axVel.scatter(gwave,med_comb, s=5,color="black")
            axVel.plot(gwave,G(gwave,a,mu,sig,bck))
            axVel.axhline(1,color="red",linestyle="--")
            axVel.axvline(mu,color="black",linestyle="--")
            axVel.axvline(vel_lines[vel_fit_linenum[i]],color="blue",linestyle="--")
            axVel.set_xlim(int(vel_lines[vel_fit_linenum[i]]-vel_mask[i]),int(vel_lines[vel_fit_linenum[i]]+vel_mask[i]))
            axVel.set_xlabel("Wavelength (A)")
            axVel.set_ylabel("Median Normalized Flux")
            del_vel_lam = np.round(float(mu)-float(vel_lines[vel_fit_linenum[i]]),decimals=4)
            axVel.set_title("lam-lam_0: "+str(del_vel_lam)+" A")
            if i >= 8:
                axVel.set_ylim(0.9,1.1)
                axVel.set_title("HeI "+str(vel_lines[vel_fit_linenum[i]])+"A line lam-lam_0: "+str(del_vel_lam)+" A radV: "+str(((float(mu)-float(vel_lines[vel_fit_linenum[i]]))/float(vel_lines[vel_fit_linenum[i]]))*3*10**5)+"km/s")
            
            plt.savefig(self.target_dir+"OUT/"+self.objid+"_Vel_"+str(i+1)+".png", dpi=300)
        
    def plot_image(self, image, name):        
        plt.clf()
        plt.imshow(image)
        plt.colorbar()
        plt.title(name + ' Image')
        plt.savefig(self.target_dir+"OUT/"+self.objid+"_"+name+".png")
        
    def plot_sci_final(self, sci_final, scinum):
        plt.clf()
        plt.imshow(sci_final)
        plt.colorbar()
        plt.title('Final Science ' + str(scinum))
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+self.objid+"_final_sci" + str(scinum)+".png")
        print("Master Science Number: " + str(scinum) + " Made")
        
    def gauss_cen_plots(self, sci1_fit, cen_line, scinum):
    
        # ---- Ensure target directory exists ----
        save_dir = os.path.join(self.target_dir, f"ImageNumber_{scinum}")
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
            
    def lam_cal_res(self, pixc, del_lam, scinum):
        fig, axL = plt.subplots(1, 1, figsize=(8,6))
        
        axL.scatter(pixc,del_lam,color="blue")
        axL.axhline(0,linestyle="--",color="black")
        axL.set_xlabel("Y Pixel Value")
        axL.set_ylabel("Angstroms")
        
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"LamCalRes_"+str(scinum)+".png")
        
    def lam_cal_res_vel(self, pixc, res_vel, scinum):
        plt.cla()
        fig, axV = plt.subplots(1, 1, figsize=(8,6))
        axV.scatter(pixc,res_vel,color="green")
        axV.axhline(0,linestyle="--",color="black")
        axV.set_xlabel("Y Pixel Value")
        axV.set_ylabel("km/s")
        
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"LamCalRes_Velocity_"+str(scinum)+".png")
    
    def plot_cen_line(self, cen_line, scinum):
        fig, axcs = plt.subplots(1, 1, figsize=(8,6))
        censcat = np.arange(0,len(cen_line),1)
        axcs.scatter(censcat,cen_line, s=3)
        axcs.set_xlabel("Y Pixel")
        axcs.set_ylabel("X Pixel")
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"cen_line_"+str(scinum)+".png")
        #print("len(cen_line)",len(cen_line))
        
    def make_aperature_fit_plots(self, sci1_fit, sci_final_1, cen_line, scinum):
        # Prepare save directory once
        save_dir = os.path.join(self.target_dir, f"ImageNumber_{scinum}", f"E20Fits_{scinum}")
        os.makedirs(save_dir, exist_ok=True)
    
        # Loop through every 200th index up to 4000
        for i in range(0, 4000, 200):
            a, mu, sigma, bck = sci1_fit[i]
            rowin = sci_final_1[i]
            cen = cen_line[i]
    
            # Compute useful slice indices once
            left = cen - self.appw - self.buffw - self.bckw
            right = cen + self.appw + self.buffw + self.bckw + 1
            fit_left = cen - self.appw
            fit_right = cen + self.appw
    
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
    
    def make_aperature_plots(self, sci1_fit, sci_final_1, cen_line, scinum):
        save_dir = os.path.join(self.target_dir, f"ImageNumber_{scinum}", f"E20Data_{scinum}")
        os.makedirs(save_dir, exist_ok=True)
        
        for i in range(0, 4000, 200):
            a, mu, sigma, bck = sci1_fit[i]
            rowin = sci_final_1[i]
            center = cen_line[i]
        
            # Compute all x-range boundaries
            left_edge = center - self.appw - self.buffw - self.bckw
            right_edge = center + self.appw + self.buffw + self.bckw + 1
        
            x_vals = np.arange(left_edge, right_edge)
            y_vals = rowin[left_edge:right_edge]
        
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.step(x_vals, y_vals, where='mid', color="black")
        
            # Draw boundary lines
            ax.axvline(center, color="black", linestyle="--")  # Center line
            ax.axvline(center - self.appw, color="blue")         # Signal region start
            ax.axvline(center + self.appw, color="blue")         # Signal region end
            ax.axvline(center - self.appw - self.buffw, color="red")        # Buffer start
            ax.axvline(center + self.appw + self.buffw, color="red")        # Buffer end
            ax.axvline(left_edge, color="green")                           # Background start
            ax.axvline(center + self.appw + self.buffw + self.bckw, color="green")  # Background end
        
            # Plot invisible Gaussian for centering (alpha=0)
            x_fit = np.linspace(center - self.appw, center + self.appw, 300)
            ax.plot(x_fit, G(x_fit, a, mu, sigma, bck), color="purple", alpha=0)
        
            # Set axis limits and labels
            ax.set_xlim(left_edge - 1, right_edge + 1)
            ax.set_xlabel("X pixels")
            ax.set_ylabel("Counts (e⁻)")
            ax.tick_params(labelsize=14)
        
            # Save and clean up
            #fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f"EData{i}.png"))
            plt.close(fig)
            
    def plot_raw_flux(self, flux_raw1, scinum):
        plt.cla()
        y_pix = np.arange(0,len(flux_raw1),1)
        plt.scatter(y_pix,flux_raw1, s=1)
        plt.xlabel("Y Pixel")
        plt.ylabel("Background subtracted Flux")
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_bsub_"+str(scinum)+".png", dpi=300)
        
    def plot_sci_lamcal(self, y_lam, y_pix, flux_raw_cor1, fit, v_helio, scinum):
        fig, ax2 = plt.subplots(1, 1, figsize=(8,6))
        ax2.scatter(y_lam(y_pix, fit, v_helio),flux_raw_cor1, s=1)
        ax2.axvline(x=6562.81,color='red')
        ax2.axvline(x=4861.35,color='cyan')
        ax2.axvline(x=4340.47,color='blueviolet')
        plt.xlabel("Wavelength (A)")
        plt.ylabel("Background subtracted Flux (Electrons)")
        plt.ylim(0,30000)
        plt.title("Sci Image"+str(scinum))
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_LamCal_"+str(scinum)+".png", dpi=300)
    
    def plot_sci_lamcal_res(self, y_lam, y_pix, flux_raw_cor1, rnErr, fit, v_helio, scinum):
        plt.cla() # Clear plt to prevent over plotting
        fig, axE = plt.subplots(1, 1, figsize=(8,6))
        axE.scatter(y_lam(y_pix, fit, v_helio), flux_raw_cor1, s=1)
        axE.scatter(y_lam(y_pix, fit, v_helio), rnErr, s=1,color='red')
        axE.set_xlabel("Wavelength (A), Error (10x)")
        axE.set_ylabel("Background subtracted Flux (Electrons)")
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_LamCal_Res_"+str(scinum)+".png", dpi=300)   
        
    def plot_sci_flux_masked(self, lam_red, lam_remove, flux_mask_red, flux_mask_remove, wavelengths, con_lam_test, scinum):
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
            os.makedirs(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_Masked_All_"+str(scinum), exist_ok=True) # Initializes a directory for E20 files
            plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/Sci_Flux_Masked_All_"+str(scinum)+"/Sci_Flux_Masked_All_i"+str(scinum)+"_"+str(i)+".png", dpi=300)   
    
    def plot_sci_flux_norm_est(self, wavelengths, Fnorm_Es, scinum):
        plt.cla()
        fig, axCont = plt.subplots(1, 1, figsize=(8,6))
        axCont.scatter(wavelengths,Fnorm_Es, s=1,color="green")
        axCont.set_xlabel("Wavelength (A)")
        axCont.set_ylabel("Estimate Cont. Normalized Flux")
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_NormEst_"+str(scinum)+".png", dpi=300)   
    
    def plot_fit_over_flux(self, fitted_wavelengths, smooth_cont, fitted_values, fitted_fluxes, lam_removea, flux_mask_removea, lam_red, lam_remove, flux_mask_red, flux_mask_remove, lam_reda, flux_mask_reda, scinum):
        plt.cla()
        fig, axL = plt.subplots(1, 1, figsize=(8,6))
        axL.scatter(fitted_wavelengths,smooth_cont, s=1,color="blue")
        axL.scatter(fitted_wavelengths,fitted_values, s=0.9,color="green")
        
        axL.scatter(fitted_wavelengths,fitted_fluxes, s=0.1,color="black")
        axL.scatter(lam_removea,flux_mask_removea,s=0.1,color="red")
        axL.set_xlabel("Wavelength (A)")
        axL.set_ylabel("Fitted Values over Flux")
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"FitOverFlux_"+str(scinum)+".png", dpi=300)   
        
        # 4 evenly spaced plots of the fitted values, the raw, origonal fluxes, and the smoothed fitted values
        if (self.verbose == True):
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
                os.makedirs(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"FitOverFluxZoom_"+str(scinum), exist_ok=True) # Initializes a directory for E20 files
                plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"FitOverFluxZoom_"+str(scinum)+"/FitOverFlux_iter_"+str(scinum)+"_"+str(i)+".png", dpi=300)   
        
    def plot_sci_normalized(self, fitted_wavelengths, Fnorm, sf_sm, scinum):
        plt.cla()
        fig, axNorm = plt.subplots(1, 1, figsize=(8,6))
        axNorm.scatter(fitted_wavelengths,Fnorm, s=1,color="black")
        axNorm.scatter(fitted_wavelengths,sf_sm, s=0.5,color="red")
        axNorm.axhline(1,color="red",linestyle="--")
        axNorm.set_xlabel("Wavelength (A)")
        axNorm.set_ylabel("Normalized Flux")
        axNorm.set_title("Normalized Spectra for "+self.objid+" Image:"+str(scinum))
        axNorm.set_ylim(0,1.5)
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"Sci_Normalized_"+str(scinum)+".png", dpi=300)   
        
        # Some other diagnostic plots
        plt.cla()
        fig, axNorm = plt.subplots(1, 1, figsize=(8,6))
        axNorm.scatter(fitted_wavelengths[3000:4096],Fnorm[3000:4096], s=1,color="black")
        axNorm.set_xlim(3760,4400)
        axNorm.axhline(1,color="red",linestyle="--")
        axNorm.set_xlabel("Wavelength (A)")
        axNorm.set_ylabel("Normalized Flux")
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"Sci_Normalized_Left"+str(scinum)+".png", dpi=300)   
        
        
        plt.cla()
        fig, axNorm = plt.subplots(1, 1, figsize=(8,6))
        axNorm.scatter(fitted_wavelengths[2700:3300],Fnorm[2700:3300], s=1,color="black")
        axNorm.set_xlabel("Wavelength (A)")
        axNorm.set_ylabel("Normalized Flux")
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"Sci_Normalized_hbeta"+str(scinum)+".png", dpi=300)   
    
    def plot_combined_norm(self, gwave, med_comb, sig_final, RMS):
        plt.cla()
        fig, axNorm = plt.subplots(1, 1, figsize=(8,6))
        axNorm.scatter(gwave,med_comb, s=0.8,color="black")
        axNorm.axhline(1,color="red",linestyle="--")
        axNorm.set_xlabel("Wavelength (A)")
        axNorm.set_ylabel("Median Normalized Flux")
        axNorm.set_ylim(-0.1,1.5)
        
        axNorm.scatter(gwave,sig_final, s=0.3,color="red")
        axNorm.set_title("Median Normalized Spectra for "+self.objid+", SNR 5500 A:"+str(np.round(1/RMS, decimals=2)))
        
        plt.savefig(self.target_dir+"OUT/"+self.objid+"_OUT.png", dpi=300)
        
        if (self.verbose == True):
            for i in range(0,2600,40):
                axNorm.set_xlim((3700+i),(4200+i))
                plt.savefig(self.target_dir+"OUT/"+self.objid+"_OUT_Zoom"+str(i)+".png", dpi=300)