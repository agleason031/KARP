'''
created by Alex Gleason

aux grapher module for KARP
'''

#import packages
import matplotlib.pyplot as plt
import numpy as np
import os

# Define a Gaussian
def G(x, a, mu, sigma, bck):
	return (a * np.exp(-(x-mu)**2/(2*sigma**2))) + bck
	# A 4d Gaussian


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
    plt.imshow(masterbias)
    plt.colorbar()
    plt.title('Master Bias Image')
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+objid+"MBias_"+str(scinum)+".png")
    print("Master Bias Made")
    
def plot_masterflat(masterflat, target_dir, scinum, objid):
    plt.imshow(masterflat)
    plt.colorbar()
    plt.title('Master Flat Image')
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+objid+"_MFlat_"+str(scinum)+".png")
    print("Master Flat Made")

def plot_masterflatDEbias(masterflatDEbias, target_dir, scinum, objid):
    plt.imshow(masterflatDEbias)
    plt.colorbar()
    plt.title('Master Flat sub Bias Image')
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+objid+"_MF_MB_"+str(scinum)+".png")
    print("Master Flat-Bias Made")
    
def plot_smooth_mf_mb(smooth_mf_mb, target_dir, scinum, objid):
    plt.imshow(smooth_mf_mb)
    plt.colorbar()
    plt.title('Smoothed MF_MB Image (5x5 Boxcar)')
    plt.savefig(str(target_dir)+"ImageNumber_"+str(scinum)+"/"+objid+"_smoothed_MF_MB_"+str(scinum)+".png")
    print("Smoothed MF_MB Made")
    
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
    