#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 17:01:27 2025

@author: agleason
"""
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from astropy import units as u

def heliocentric_correction(objRA, objDEC, otime):
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
    #print("heliocorr",heliocorr)
    # Map to v_helio
    correction = heliocorr.to(u.km/u.s)
    #print("correction",correction)
    correction = str(correction)
    correction = correction.replace('km / s', '')
    return(float(correction))