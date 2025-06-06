#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:13:39 2025

@author: agleason
"""
from scipy.optimize import least_squares
from numba import njit
import numpy as np

# Define a Gaussian
@njit
def G(x, a, mu, sigma, bck):
	return (a * np.exp(-(x-mu)**2/(2*sigma**2))) + bck
	# A 4d Gaussian
 
@njit
def residuals(p, x, y):
    res = np.empty_like(y)
    a, mu, sigma, bck = p
    for i in range(len(x)):
        res[i] = G(x[i], a, mu, sigma, bck) - y[i]
    return res

# Our continuum function:
@njit
def con_lam(wave, params):
    aa,bb,cc,dd,ee,ff,gg = params
    # flux as a function of wavelength
    return (aa*(wave)**6)+(bb*(wave)**5)+(cc*wave**4)+(dd*wave**3)+(ee*wave**2)+(ff*wave)+gg

def y_lam(y, fit, v_helio):
    a, b, c, d = fit
    # Vectorize array
    y = np.array(y)
    # Wavelength as a function of pixel with helocentric velocity correction
    return ((a*(y/2000)**3)+(b*(y/2000)**2)+c*(y/2000)+d)*(1+(v_helio/(3*10**5)))

def fit_cent_gaussian(row, clr, a_width, buff_width, bckrnd_width):
    clr = int(clr + 0.5)

    fit_start = clr - a_width
    fit_end   = clr + a_width
    buff_start = fit_start - buff_width
    buff_end   = fit_end + buff_width
    bck_start = buff_start - bckrnd_width
    bck_end   = buff_end + bckrnd_width

    x_vals = np.arange(fit_start, fit_end)
    y_vals = row[fit_start:fit_end]
    # Exclude the fit + buffer region from background
    y_bck_left  = row[bck_start:buff_start]
    y_bck_right = row[buff_end:bck_end]
    y_bck = np.concatenate([y_bck_left, y_bck_right])

    bck = np.median(y_bck)
    amp = np.max(y_vals) - bck
    p0 = [amp, clr, 2.0, bck]

    bounds = ([0, 0, 1, -100], [np.inf, np.inf, np.inf, np.inf])

    result = least_squares(residuals, p0, args=(x_vals, y_vals), bounds=bounds, max_nfev=30)

    if result.success:
        return result.x
    else:
        return [np.nan, np.nan, np.nan, np.nan]
    
# Helper function to format image filenames
def format_fits_filename(dloc, num):
    return f"{dloc}k.{int(num):04d}.fits"

#input string true or false and returns bool
def string_to_bool(s):
    return {"true": True, "false": False}.get(s.lower())