#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:13:39 2025

@author: agleason
"""
from scipy.optimize import least_squares
from numba import njit
import numpy as np
import math

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

@njit
def gaussian_integral_vec(a, mu, sigma, bck, x1, x2):
    result = np.empty(a.shape)
    sqrt2 = math.sqrt(2.0)
    sqrt_2pi = math.sqrt(2.0 * math.pi)
    for i in range(a.shape[0]):
        term1 = (x2[i] - mu[i]) / (sqrt2 * sigma[i])
        term2 = (x1[i] - mu[i]) / (sqrt2 * sigma[i])
        erf_term = math.erf(term1) - math.erf(term2)
        gauss_area = a[i] * sigma[i] * sqrt_2pi * 0.5 * erf_term
        bck_area = bck[i] * (x2[i] - x1[i])
        result[i] = gauss_area + bck_area
    return result

def y_lam(y, fit, v_helio):
    a, b, c, d = fit
    # Vectorize array
    y = np.array(y)
    # Wavelength as a function of pixel with helocentric velocity correction
    return ((a*(y/2000)**3)+(b*(y/2000)**2)+c*(y/2000)+d)*(1+(v_helio/(3*10**5)))

def y_lam_4(y, fit, v_helio):
    e, a, b, c, d = fit
    # Vectorize array
    y = np.array(y)
    # Wavelength as a function of pixel with helocentric velocity correction
    return (e*(y/2000)**4+(a*(y/2000)**3)+(b*(y/2000)**2)+c*(y/2000)+d)*(1+(v_helio/(3*10**5)))

def fit_cent_gaussian(x_vals, y_vals, y_bck, clr):
    if x_vals is None:
        return [np.nan, np.nan, np.nan, np.nan]

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

def build_fit_args(sci_final_1, cen_line, appw, buffw, bckw):
    n_rows, row_len = sci_final_1.shape
    
    clr = np.round(cen_line).astype(int)

    fit_start = clr - appw
    fit_end   = clr + appw
    buff_start = fit_start - buffw
    buff_end   = fit_end + buffw
    bck_start  = buff_start - bckw
    bck_end    = buff_end + bckw
    
    # Identify valid rows where all slices are within bounds
    valid = (bck_start >= 0) & (bck_end < row_len)

    # Preallocate result list
    fit_args = []

    for i in range(n_rows):
        if not valid[i]:
            fit_args.append((None, None, None, clr[i]))
            continue

        row = sci_final_1[i]
        x_vals = np.arange(fit_start[i], fit_end[i])
        y_vals = row[fit_start[i]:fit_end[i]]
        y_bck = np.concatenate([
            row[bck_start[i]:buff_start[i]],
            row[buff_end[i]:bck_end[i]]
        ])
        fit_args.append((x_vals, y_vals, y_bck, clr[i]))

    return fit_args
