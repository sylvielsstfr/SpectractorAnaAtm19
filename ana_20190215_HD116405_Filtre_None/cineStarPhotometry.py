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
from photutils import DAOStarFinder


from photutils import make_source_mask
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground




plt.rcParams["axes.labelsize"]="large"
plt.rcParams["axes.linewidth"]=2.0
plt.rcParams["xtick.major.size"]=8
plt.rcParams["ytick.major.size"]=8
plt.rcParams["ytick.minor.size"]=5
plt.rcParams["xtick.labelsize"]="large"
plt.rcParams["ytick.labelsize"]="large"

plt.rcParams["figure.figsize"]=(8,8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
#plt.rcParams['axes.facecolor'] = 'blue'
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['lines.markeredgewidth'] = 0.3 # the line width around the marker symbol
plt.rcParams['lines.markersize'] = 5  # markersize, in points
plt.rcParams['grid.alpha'] = 0.75 # transparency, between 0.0 and 1.0
plt.rcParams['grid.linestyle'] = '-' # simple line
plt.rcParams['grid.linewidth'] = 0.4 # in points





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





#-------------------------------------------------------------------------
#
# MAIN()
#
#---------------------------------------------------

if __name__ == "__main__":

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

    all_files=t["file"]

    cutmax=1

    for idx in np.arange(Nobs):

        thefile = all_files[idx]
        therawfile=thefile.replace("_spectrum.fits",".fit")

        print("{}) : {}, {} ".format(idx, thefile,therawfile))

        fullfilename = os.path.join(rawinput_directory, therawfile)
        #try:
        if 1:
            hdu = fits.open(fullfilename)

            data=hdu[0].data
            vmin = data.min()
            vmax = data.max() / cutmax

            norm = ImageNormalize(data, interval=MinMaxInterval(), stretch=SqrtStretch())

            plt.imshow(hdu[0].data,origin="lower",cmap="jet",vmin=vmin,vmax=vmax,norm=norm)
            title="{}) :  {}".format(idx,therawfile,fontsize="12")
            plt.title(title)
            plt.grid()
            plt.draw()
            plt.pause(0.00001)
            plt.clf()
        #except:
        if 0:
            print("Unexpected error:", sys.exc_info()[0])
            pass

    #----------------------------------------------------------------------------------------------------------------------