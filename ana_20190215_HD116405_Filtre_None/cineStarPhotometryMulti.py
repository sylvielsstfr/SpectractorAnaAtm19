#########################################################################################################
# Process grey attenuation
#########################################################################################################


import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.dates as mdates
from matplotlib import gridspec

import numpy as np

from scipy.interpolate import interp1d
from astropy.time import Time
from astropy.table import Table,QTable
from astropy.io import fits,ascii
from astropy.visualization.mpl_normalize import (ImageNormalize,MinMaxInterval, SqrtStretch)

import os,sys
from os import listdir
from os.path import isfile, join
import pandas as pd
import re

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


if not 'workbookDir' in globals():
    workbookDir = os.getcwd()
print('workbookDir: ' + workbookDir)

spectractordir=workbookDir+"/../../Spectractor"
print('spectractordir: ' + spectractordir)
toolsdir=workbookDir+"/../common_tools"
print("toolsdir:",toolsdir)


import sys
sys.path.append(workbookDir)
sys.path.append(spectractordir)
sys.path.append(os.path.dirname(workbookDir))
sys.path.append(toolsdir)



from spectractor import parameters
from spectractor.extractor.extractor import Spectractor
from spectractor.logbook import LogBook
from spectractor.extractor.dispersers import *
from spectractor.extractor.spectrum import *
from spectractor.tools import ensure_dir





# PhotUtil
from astropy.stats import sigma_clipped_stats
from photutils import aperture_photometry
from photutils import CircularAperture,CircularAnnulus
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import astropy_mpl_style
#plt.style.use(astropy_mpl_style)

from photutils import DAOStarFinder


from photutils import make_source_mask
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground

import astropy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy.table import Table
from astropy.coordinates import Angle
from astropy.time import Time, TimezoneInfo
from pytz import timezone
import pytz
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import Longitude, Latitude
from astropy.coordinates import get_sun,get_moon



plt.rcParams["axes.labelsize"]="large"
plt.rcParams["axes.linewidth"]=2.0
plt.rcParams["xtick.major.size"]=8
plt.rcParams["ytick.major.size"]=8
plt.rcParams["ytick.minor.size"]=5
plt.rcParams["xtick.labelsize"]="large"
plt.rcParams["ytick.labelsize"]="large"

plt.rcParams["figure.figsize"]=(8,8)
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
#plt.rcParams['axes.facecolor'] = 'blue'
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['lines.markeredgewidth'] = 0.3 # the line width around the marker symbol
plt.rcParams['lines.markersize'] = 5  # markersize, in points
plt.rcParams['grid.alpha'] = 0.75 # transparency, between 0.0 and 1.0
plt.rcParams['grid.linestyle'] = '-' # simple line
plt.rcParams['grid.linewidth'] = 0.4 # in points




PDM_Longitude=Longitude(u'0°08′34″')
PDM_Latitude=Latitude(u'42°56′11″')
PDM_Height=2.877*u.m



# where are the image
#----------------------
thedate = "20190215"
rawinput_directory="/Users/dagoret/DATA/PicDuMidiFev2019/prod_"+thedate+"_v4"

#-----------------------------------------------------------------------------------------------
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.

    For example for the PSF

    x=pixel number
    y=Intensity in pixel

    values-x
    weights=y=f(x)

    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)  # Fast and numerically precise
    return average, np.sqrt(variance)


#---------------------------------------------------------------
def PlotStarmagvsUTC(mydatetime, mystarmag,mystarmag_err,ax,TMIN,TMAX,Ntot):
    """

    :param ifig:
    :param all_airmass:
    :param all_datetime:
    :param all_flag:
    :return:
    """


    N=len(mydatetime)



    # wavelength bin colors
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=Ntot)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(Ntot), alpha=1)


    myFmt = mdates.DateFormatter('%d-%H:%M')
    ax.xaxis.set_major_formatter(myFmt)


    #plt.errorbar(all_datetime, all_starmag, yerr=all_starmag_err, fmt='.', color="red", ecolor='grey')
    ax.scatter(mydatetime, mystarmag, marker="o", c=all_colors[:N])


    myFmt = mdates.DateFormatter('%d-%H:%M')
    #ax.gca().xaxis.set_major_formatter(myFmt)
    ax.xaxis.set_major_formatter(myFmt)

    #ax.gcf().autofmt_xdate()
    #ax.autofmt_xdate()

    ax.set_xlim(TMIN,TMAX)
    ax.set_ylim(-17.,-15.)

    ax.grid(True, color="k")
    ax.set_xlabel("date (UTC)")
    ax.set_ylabel("Star magnitude (mag)")
    ax.set_title("Star magnitude vs date")

#---------------------------------------------------------------
def PlotBkgmagvsUTC(mydatetime, mybkgmag,mybkgmag_err,ax,TMIN,TMAX,Ntot):
    """

    :param ifig:
    :param all_airmass:
    :param all_datetime:
    :param all_flag:
    :return:
    """


    N=len(mydatetime)



    # wavelength bin colors
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=Ntot)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(Ntot), alpha=1)


    myFmt = mdates.DateFormatter('%d-%H:%M')
    ax.xaxis.set_major_formatter(myFmt)


    #plt.errorbar(all_datetime, all_starmag, yerr=all_starmag_err, fmt='.', color="red", ecolor='grey')
    ax.scatter(mydatetime, mybkgmag, marker="o", c=all_colors[:N])


    myFmt = mdates.DateFormatter('%d-%H:%M')
    #ax.gca().xaxis.set_major_formatter(myFmt)
    ax.xaxis.set_major_formatter(myFmt)

    #ax.gcf().autofmt_xdate()
    #ax.autofmt_xdate()

    ax.set_xlim(TMIN,TMAX)

    ax.grid(True, color="k")
    ax.set_xlabel("date (UTC)")
    ax.set_ylabel("Bkb magnitude (mag)")
    ax.set_title("Background magnitude vs date")


#------------------
def PlotElevation(delta_midnight,sunaltazs,moonaltazs,staraltazs,DTim,ax):
    """

    :return:
    """
    #plt.figure(figsize=(16., 10.))
    ax.plot(delta_midnight, sunaltazs.alt, color='r', label='Sun')
    ax.plot(delta_midnight, moonaltazs.alt, color='orange', label='Moon')
    ax.scatter(delta_midnight, staraltazs.alt,
                c=staraltazs.az, label='hd_116405', lw=0, s=8,
                cmap='viridis')

    ax.plot([DTim,DTim],[0,90],"-b")

    # plot astronomical crepuscule
    ax.fill_between(delta_midnight.to('hr').value, 0, 90,
                     sunaltazs.alt < -0 * u.deg, color='0.5', zorder=0)
    # plot astronomical night
    ax.fill_between(delta_midnight.to('hr').value, 0, 90,
                     sunaltazs.alt < -18 * u.deg, color='k', zorder=0)
    #plt.colorbar().set_label('Azimuth [deg]')
    ax.legend(loc='upper left',fontsize=8)
    ax.set_xlim(-12, 12)
    ax.set_xticks(np.arange(13) * 2 - 12)
    ax.set_ylim(0, 90)
    #ax.set_title('hd_116405 and Moon during night 15 to 16 February 2019')
    ax.set_xlabel('Hours from Midnight (France:Pic du Midi)')
    ax.set_ylabel('Altitude [deg]')
    #plt.show()




#-------------------------------------------------------------------------
#
# MAIN()
#
#---------------------------------------------------

if __name__ == "__main__":

    starloc = astropy.coordinates.SkyCoord.from_name('HD116405')
    # definition of the location to astropy
    site_location = astropy.coordinates.EarthLocation(lat=PDM_Latitude, lon=PDM_Longitude, height=PDM_Height)

    # UTC offset
    utcoffset = 1 * u.hour  # France
    # local midnight in UTC
    midnight = Time('2019-2-16 00:00:00') - utcoffset
    delta_midnight = np.linspace(-12, 12, 1000) * u.hour

    # array of times in UTC
    times_seq = midnight + delta_midnight

    # sequence in altitude Azimuth object at current site
    frame_seq = AltAz(obstime=times_seq, location=site_location)

    # Altitude and Elevation at current site at selected time
    sunaltazs_seq = get_sun(times_seq).transform_to(frame_seq)
    moonaltazs_seq = get_moon(times_seq).transform_to(frame_seq)
    staraltazs_seq = starloc.transform_to(frame_seq)




    t = Table.read('processStarPhotometry.ecsv', format='ascii.ecsv')

    t["airmass"].info.format = '%1.2f'  # for consistent table output
    t["starmag"].info.format = '%2.3f'
    t["bkgmag"].info.format = '%2.3f'
    t["starmagerr"].info.format = '%1.3g'
    t["bkgmagerr"].info.format = '%1.3g'
    t["x0"].info.format = '%3.0f'
    t["y0"].info.format = '%3.0f'


    print(t)
    Nobs=len(t)
    Ntot=Nobs

    all_files=t["file"]
    all_datetime = [Time(d, format='isot', scale='utc').to_datetime() for d in t["date"]]
    TMIN=all_datetime[0]
    TMAX=all_datetime[-1]


    cutmax=10





    current_date=[]
    current_starmag=[]
    current_starmagerr=[]
    current_bkgmag = []
    current_bkgmagerr = []

    for idx in np.arange(Nobs):

        thefile = all_files[idx]
        therawfile=thefile.replace("_spectrum.fits",".fit")

        #print("{}) : {}, {} ".format(idx, thefile,therawfile))

        fullfilename = os.path.join(rawinput_directory, therawfile)

        fig, ax = plt.subplots(2, 2, figsize=(20,8), gridspec_kw={'height_ratios': [1, 1],'width_ratios': [1,3]})
        ax1 = ax[0, 0]
        ax2 = ax[0, 1]
        ax3 = ax[1, 0]
        ax4 = ax[1, 1]


        #try:
        if 1:
            hdu = fits.open(fullfilename)

            data=hdu[0].data
            vmin = data.min()
            vmax = data.max() / cutmax

            norm = ImageNormalize(data, interval=MinMaxInterval(), stretch=SqrtStretch())

            ax1.imshow(hdu[0].data,origin="lower",cmap="jet",vmin=vmin,vmax=vmax,norm=norm,aspect="equal")
            title="{}) :  {}".format(idx,therawfile,fontsize="10")
            title=title.split("Filter")[0]
            label0=t["date"][idx]
            label1=label0.split("T")[1]
            label2='airmass = {:1.2f}'.format(t["airmass"][idx])
            ax1.text(100, 1900,title, fontsize=12,color='yellow',fontweight='bold')
            ax1.text(100, 1700,label1, fontsize=12,color='yellow',fontweight='bold')
            ax1.text(100, 1500, label2, fontsize=12,color='yellow',fontweight='bold')
            #ax1.set_title(title,fontsize=10)
            ax1.grid(color="white")
            x0=t["x0"][idx]
            y0=t["y0"][idx]
            aperture = CircularAperture((x0,y0), r=2*30)
            aperture.plot(color='red', lw=2,ax=ax1)

            current_date.append(all_datetime[idx])
            current_starmag.append(t["starmag"][idx])
            current_starmagerr.append(t["starmagerr"][idx])
            current_bkgmag.append(t["bkgmag"][idx])
            current_bkgmagerr.append(t["bkgmagerr"][idx])

            PlotStarmagvsUTC(current_date, current_starmag, current_starmagerr, ax2, TMIN, TMAX, Ntot)
            PlotBkgmagvsUTC(current_date, current_bkgmag, current_bkgmagerr, ax4, TMIN, TMAX, Ntot)

            #Detla time relative to midnight : current
            DTim=(Time(t["date"][idx], format='isot', scale='utc')-(Time('2019-2-16 00:00:00') - utcoffset)).sec/3600.

            PlotElevation(delta_midnight, sunaltazs_seq, moonaltazs_seq, staraltazs_seq,DTim ,ax3)


            fig.tight_layout()
            fig.subplots_adjust(wspace=0, hspace=0)

            plt.draw()


            plt.pause(1e-8)
            #plt.clf()
            plt.close()





        #except:
        if 0:
            print("Unexpected error:", sys.exc_info()[0])
            pass

    #----------------------------------------------------------------------------------------------------------------------