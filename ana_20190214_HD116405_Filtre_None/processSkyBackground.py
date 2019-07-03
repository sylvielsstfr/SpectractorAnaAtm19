#########################################################################################################
# Process grey attenuation
#########################################################################################################

import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import re

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

from libatmscattering import *


plt.rcParams["axes.labelsize"]="large"
plt.rcParams["axes.linewidth"]=2.0
plt.rcParams["xtick.major.size"]=8
plt.rcParams["ytick.major.size"]=8
plt.rcParams["ytick.minor.size"]=5
plt.rcParams["xtick.labelsize"]="large"
plt.rcParams["ytick.labelsize"]="large"

plt.rcParams["figure.figsize"]=(20,20)
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

######### CONFIGURATION

## wavelength
WLMIN = 380.0
WLMAX = 1000.0
#NBWLBIN = 62
NBWLBIN = 124
WLBINWIDTH = (WLMAX - WLMIN) / float(NBWLBIN)

WLMINBIN = np.arange(WLMIN, WLMAX, WLBINWIDTH)
WLMAXBIN = np.arange(WLMIN + WLBINWIDTH, WLMAX + WLBINWIDTH, WLBINWIDTH)
WLMEANBIN=(WLMINBIN + WLMAXBIN)/2.



print('WLMINBIN..................................=', WLMINBIN.shape, WLMINBIN)
print('WLMAXBIN..................................=', WLMAXBIN.shape, WLMAXBIN)
print('NBWLBIN...................................=', NBWLBIN)
print('WLBINWIDTH................................=', WLBINWIDTH)

## colors

# wavelength bin colors
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=NBWLBIN)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
all_colors = scalarMap.to_rgba(np.arange(NBWLBIN), alpha=1)

## output directory for tables
ouputtabledir="outputtabledir"

## create output directory
ensure_dir(ouputtabledir)
#---------------------------------------------------------------------
def GetWLBin(wl):
    """

    :param wl: wavelength scalar
    :return: index
    """

    set_ibin = np.where(np.logical_and(WLMINBIN <= wl, WLMAXBIN > wl))[0]

    if len(set_ibin)==1:
        return set_ibin[0]
    else:
        return -1
#---------------------------------------------------------------------
def GETWLLabels():

    all_labels=[]
    for idx in np.arange(NBWLBIN):
        label="$\lambda$ : {:3.0f}-{:3.0f} nm".format(WLMINBIN[idx], WLMAXBIN[idx])
        all_labels.append(label)
    all_labels=np.array(all_labels)
    return all_labels
#------------------------------------------------------------------------

WLLABELS=GETWLLabels()

# reference number
#--------------------
#IDXMINREF=222
#IDXMAXREF=222

IDXMINREF=0
IDXMAXREF=0


#IDXMINREF=0
#IDXMAXREF=0

# where are the spectra
#----------------------
thedate = "20190214"
#input_directory = "/Users/dagoret/DATA/PicDuMidiFev2019/spectractor_output_prod3/" + thedate
#input_directory = "/Users/dagoret/DATA/PicDuMidiFev2019/spectractor_output_prod4/" + thedate
input_directory = "/Users/dagoret/DATA/PicDuMidiFev2019/spectractor_output_test_marcsel_july19/" + thedate


#-------------------------------------------------------------------------------------
def GetAllFiles(dir):
    """
    GetAllFiles(dir): provides the list of relevant files inside the directory dir


    :param dir: input directory pathname
    :return: list of files
    """



    # get all files
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    onlyfiles = np.array(onlyfiles)

    # sort files
    sortedindexes = np.argsort(onlyfiles)
    onlyfiles = onlyfiles[sortedindexes]

    # get only _spectrum.fits file
    onlyfilesspectrum = []
    for file_name in onlyfiles:
        if re.search("^T.*_spectrum.fits$", file_name):
            # check if other files exits

            filetype = file_name.split('.')[-1]

            output_filename = os.path.basename(file_name)
            output_filename = os.path.join(dir, output_filename)
            output_filename_spectrogram = output_filename.replace('spectrum', 'spectrogram')
            output_filename_psf = output_filename.replace('spectrum.fits', 'table.csv')

            # go to next simulation if output files already exists
            if os.path.isfile(output_filename) and os.path.isfile(output_filename_spectrogram) and os.path.isfile(
                    output_filename_psf):
                filesize = os.stat(output_filename).st_size
                print(">>>>> output filename : {} already exists with size {} ".format(output_filename, filesize))

                filesize = os.stat(output_filename_spectrogram).st_size
                print(">>>>> output filename : {} already exists with size {} ".format(output_filename_spectrogram,
                                                                                       filesize))

                filesize = os.stat(output_filename_psf).st_size
                print(">>>>> output filename : {} already exists with size {} ".format(output_filename_psf, filesize))

                onlyfilesspectrum.append(re.findall("(^T.*_spectrum.fits$)", file_name)[0])

    # sort again all the files
    onlyfilesspectrum = np.array(onlyfilesspectrum)
    sortedindexes = np.argsort(onlyfilesspectrum)
    onlyfilesspectrum = onlyfilesspectrum[sortedindexes]

    # get basnemae of files for later use (to check if _table.csv and _spectrogram.fits exists
    onlyfilesbasename = []
    for f in onlyfilesspectrum:
        onlyfilesbasename.append(re.findall("(^T.*)_spectrum.fits$", f)[0])


    return onlyfilesspectrum,onlyfilesbasename
#------------------------------------------------------------------------------------------------------------------------------




def ReadAllFiles(dir, filelist):
    """

    ReadAllFiles(dir, filelist): Read all files

    :param dir: input directory
    :param filelist: list fo files
    :return: various containers
    """


    # init all container
    all_indexes=[]   # continuous index
    all_eventnum=[]  # event number on filename

    all_airmass=[]   #airmass

    all_lambdas = [] # wavelength

    #fluxes
    all_flux=[]
    all_errflux=[]


    # magnitudes
    all_mag=[]
    all_errmag=[]

    # absorption (Magnitude corrected from Rayleigh
    all_abs=[]
    all_errabs=[]


    all_dt       = []   # time since beginning in hours
    all_datetime = []   # datetime
    all_T = []  # astropy time

    # preselection flag
    all_flag = []


    #bad indexes and filename
    all_badidx=[]
    all_badfn=[]


    all_BGimg=[]

    NBSPEC = len(filelist)


    #----------------------------------
    # Extract spectra information from files
    # compute magnitudes
    #-----------------------------

    count=-1  #counter of good files before filtering

    # loop on all files
    for idx in np.arange(0, NBSPEC):
        if idx==322:
            print("SKIP bad file",filelist[idx] )
            continue

        count+=1

        theeventnum = int(filelist[idx].split(".")[1].split("_")[0])


        print(" read {}) : event {} , file {}".format(idx,theeventnum,filelist[idx]))


        fullfilename = os.path.join(dir, filelist[idx])

        fullfilename_spectrogram = fullfilename.replace('spectrum', 'spectrogram')



        #try:
        if 1:
            # read fits file
            hdu = fits.open(fullfilename)
            hdu2=fits.open(fullfilename_spectrogram)





            # decode header
            header=hdu[0].header

            am=header["AIRMASS"]
            date=header["DATE-OBS"]



            if idx==0:
                T0=t = Time(date, format='isot', scale='utc')
            T=Time(date, format='isot', scale='utc')
            thedatetime=T.to_datetime()

            DT=(T-T0).sec/(3600.0)  # convert in hours


            # decode data
            data=hdu[0].data

            # extract wavelength, spectrum and errors
            wavelength=data[0,:]
            spec = data[1,:]
            err=data[2,: ]

            if(len(wavelength)==0 or len(spec)==0 or len(err)==0):
                print(">>>>>>>>>>>>>>  Empty file ",idx,")::",onlyfilesspectrum[idx] )
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>",len(wavelength)," , ",len(spec0), " , ",len(err))



            # sort the wavelengths
            wl_sorted_idx=np.argsort(wavelength)

            wl=wavelength[wl_sorted_idx]
            fl=spec[wl_sorted_idx]
            errfl=err[wl_sorted_idx]
            wlbins = [GetWLBin(w) for w in wl]
            wlbins = np.array(wlbins)


            # defines good measurmements as flux >0 and wavelength in selected bons
            goodpoints=np.where(np.logical_and(fl != 0, wlbins != -1))

            if(len(goodpoints)==0):
                print(">>>>>>>>>>>>>>  No Good points  ", idx, ")::", onlyfilesspectrum[idx])
                print(">>>>>>>>>>>>>>  No Good points  ", "wl = ",wl)
                print(">>>>>>>>>>>>>>  No Good points  ", "fl = ", fl)
                print(">>>>>>>>>>>>>>  No Good points  ", "errfl = ", errfl)



            # keep good points (wl,flux)
            wl=wl[goodpoints]
            fl=fl[goodpoints]
            errfl=errfl[goodpoints]


            # convert flux into magnitudes for each wavelength
            mag = -2.5 * np.log10(fl)
            errmag = errfl/fl

            #compute effective slant Rayleigh optical depth (not the vertical optical depth)
            od_adiab = RayOptDepth_adiabatic(wl,altitude=2890.5, costh=1./am)  # Rayleigh optical depth

            # absorption magnitude corrected from Rayleigh attenuation
            abs=mag-2.5/np.log(10.)*od_adiab
            errabs=errmag





            # save for each observation  { event-id, airmass, set of points (wl,flux,errflux, mag,abs,errabs) }
            if(len(mag>0)):
                #all_indexes.append(idx)
                all_indexes.append(count)
                all_eventnum.append(theeventnum)
                all_airmass.append(am)
                all_lambdas.append(wl)
                all_flux.append(fl)
                all_errflux.append(errfl)
                all_mag.append(mag)
                all_errmag.append(errmag)
                all_abs.append(abs)
                all_errabs.append(errabs)
                all_dt.append(DT)
                all_datetime.append(thedatetime)
                all_T.append(T)
                all_BGimg.append(hdu2[2].data)

                absmin = abs.min()
                absmax = abs.max()


                # Set a quality flag
                if absmin < 25 or absmax > 32:
                    print("file index idx = ", count, "==>  filename = ", onlyfilesspectrum[idx], " absmin= ", absmin,
                          " absmax = ", absmax)
                    all_flag.append(False)
                    all_badidx.append(count)
                    all_badfn.append(onlyfilesspectrum[idx])
                else:
                    all_flag.append(True)



        #except:
        if 0:
            print("Unexpected error:", sys.exc_info()[0])
            pass

    # transform container in numpy arrays
    all_indexes=np.array(all_indexes)
    all_eventnum = np.array(all_eventnum)

    all_airmass = np.array(all_airmass)

    all_lambdas = np.array(all_lambdas)

    all_flux = np.array(all_flux)
    all_errflux = np.array(all_errflux)


    all_mag=np.array(all_mag)
    all_errmag=np.array(all_errmag)

    all_abs = np.array(all_abs)
    all_errabs = np.array(all_errabs)

    all_dt = np.array(all_dt)
    all_datetime=np.array(all_datetime)

    all_flag = np.array(all_flag)

    all_badidx=np.array(all_badidx)
    all_badfn=np.array(all_badfn)


    return all_indexes,all_eventnum,all_airmass,all_lambdas,all_flux,all_errflux,all_mag,all_errmag,all_abs,all_errabs,all_dt,all_datetime,all_T,all_flag,all_badidx,all_badfn, all_BGimg




#---------------------------------------------------------------
def PlotBGvsUTC(ifig,all_datetime, all_images,all_flag):
    """

    :param ifig:
    :param all_airmass:
    :param all_datetime:
    :param all_flag:
    :return:
    """

    fig = plt.figure(num=ifig, figsize=(16, 8))

    Nobs = len(all_airmass)

    # wavelength bin colors
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=Nobs)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(Nobs), alpha=1)

    BGstd = []
    BGmean=[]
    for img in all_images:
        flat=img.flatten()
        flat2=-2.5*np.log10(flat)
        BGmean.append(flat2.mean())
        BGstd.append(flat2.std())


    BGmean=np.array(BGmean)
    BGstd = np.array(BGstd)


    myFmt = mdates.DateFormatter('%d-%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

    #plt.scatter(all_datetime, all_airmass, marker="o", c=all_colors)
    #plt.scatter(all_datetime, BGmean, marker="o", c="blue")
    plt.errorbar(all_datetime,BGmean,yerr=BGstd,fmt='o', color="blue",ecolor='grey')

    plt.plot([all_datetime[IDXMINREF], all_datetime[IDXMINREF]], [BGmean.min(), BGmean.max()], "g-")
    plt.plot([all_datetime[IDXMAXREF], all_datetime[IDXMAXREF]], [BGmean.min(), BGmean.max()], "g-")


    myFmt = mdates.DateFormatter('%d-%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

    plt.gcf().autofmt_xdate()

    plt.xlim(all_datetime[0], all_datetime[-1])

    plt.grid(True, color="r")
    plt.xlabel("date (UTC)")
    plt.ylabel("Sky Background (mag)")
    plt.title("Sky Background vs date")

    plt.show()

#---------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------
#
# MAIN()
#
#---------------------------------------------------

if __name__ == "__main__":






    #############################################
    # 1) Get list of files
    ##########################################
    onlyfilesspectrum,onlyfilesbasename=GetAllFiles(input_directory)



    basenamecut=[]
    for f in onlyfilesspectrum:
        basenamecut.append(f.split("_HD")[0])


    #############################################
    # 2) Read fits file
    ##########################################


    NBSPEC = len(onlyfilesspectrum)

    print('NBSPEC....................................= ', NBSPEC)


    all_indexes, all_eventnum, all_airmass, all_lambdas, all_flux, all_errflux, all_mag, all_errmag, all_abs, all_errabs, all_dt, all_datetime,all_T ,all_flag, all_badidx, all_badfn, all_BGimg=\
        ReadAllFiles(input_directory, onlyfilesspectrum)


    ifig=900

    PlotBGvsUTC(ifig, all_datetime, all_BGimg, all_flag)



#----------------------------------------------------------------------------------------------------------------------