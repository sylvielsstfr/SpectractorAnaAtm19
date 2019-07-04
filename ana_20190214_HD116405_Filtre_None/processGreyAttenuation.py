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
outputtabledir="out_processgreyattenuation"

## create output directory
ensure_dir(outputtabledir)
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

    # preselection flag
    all_flag = []


    #bad indexes and filename
    all_badidx=[]
    all_badfn=[]

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
        #try:
        if 1:
            # read fits file
            hdu = fits.open(fullfilename)

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


    return all_indexes,all_eventnum,all_airmass,all_lambdas,all_flux,all_errflux,all_mag,all_errmag,all_abs,all_errabs,all_dt,all_datetime,all_flag,all_badidx,all_badfn




#---------------------------------------------------------------------------------------------------


def PlotAbsvsIndex(ifig,all_indexes,all_lambdas,all_abs,all_errabs,all_flag):

    # ---------------------------------------
    #  Figure 3 : Magnitude corrigée de Rayleigh pour star falling vs event number
    # ------------------------------------

    # wavelength bin colors
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=NBWLBIN)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(NBWLBIN), alpha=1)



    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    # loop on wavelength bins

    # loop on events
    # for idx in goup_idx:
    for idx in all_indexes:
        thewl = all_lambdas[idx]
        theabs = all_abs[idx]
        theerrabs = all_errabs[idx]
        wlcolors = []
        flag = all_flag[idx]

        if flag:

            for w0 in thewl:
                ibin = GetWLBin(w0)

                if ibin >= 0:
                    colorVal = scalarMap.to_rgba(ibin, alpha=1)
                else:
                    colorVal = scalarMap.to_rgba(0, alpha=0.25)

                wlcolors.append(colorVal)

            plt.scatter(np.ones(len(theabs)) * idx, theabs, marker="o", c=wlcolors)
            # plt.errorbar(np.ones(len(theabs))*am,theabs,yerr=theerrabs,ecolor="k",fmt=".")

        #else:
        #    for w0 in thewl:
        #        ibin = GetWLBin(w0)

        #    if ibin >= 0:
        #        colorVal = scalarMap.to_rgba(ibin, alpha=0.25)
        #    else:
        #        colorVal = scalarMap.to_rgba(0, alpha=0.25)

        #   wlcolors.append(colorVal)

        # plt.scatter(np.ones(len(theabs)) * idx, theabs, marker="o", c=wlcolors)
        # plt.errorbar(np.ones(len(theabs))*am,theabs,yerr=theerrabs,ecolor="k",fmt=".")

    plt.plot([IDXMINREF, IDXMINREF], [0, 35], "k-")
    plt.plot([IDXMAXREF, IDXMAXREF], [0, 35], "k-")

    plt.ylim(20, 35.)
    plt.grid(True, color="r")

    plt.title("Grey Abs = $M(\lambda)-K(\lambda).Z$ vs event number")
    plt.xlabel("Event number")
    plt.ylabel("absorption $m-k(\lambda).z$ (mag)")

    plt.show()

#---------------------------------------------------------------
def PlotAMvsUTC(ifig, all_airmass, all_datetime, all_flag):
    """

    :param ifig:
    :param all_airmass:
    :param all_datetime:
    :param all_flag:
    :return:
    """

    fig = plt.figure(num=ifig, figsize=(16, 3))

    Nobs = len(all_airmass)

    # wavelength bin colors
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=Nobs)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(Nobs), alpha=1)



    myFmt = mdates.DateFormatter('%d-%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

    #plt.scatter(all_datetime, all_airmass, marker="o", c=all_colors)
    plt.scatter(all_datetime, all_airmass, marker="o", c="blue")

    plt.plot([all_datetime[IDXMINREF], all_datetime[IDXMINREF]], [all_airmass.min(), all_airmass.max()], "g-")
    plt.plot([all_datetime[IDXMAXREF], all_datetime[IDXMAXREF]], [all_airmass.min(), all_airmass.max()], "g-")


    myFmt = mdates.DateFormatter('%d-%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

    plt.gcf().autofmt_xdate()

    plt.xlim(all_datetime[0], all_datetime[-1])

    plt.grid(True, color="r")
    plt.xlabel("date (UTC)")
    plt.ylabel("airmass")
    plt.title("airmass vs date")

    plt.show()

#---------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------


def PlotMagvsUTCAM(ifig, all_airmass,all_datetime, all_lambdas, all_mag, all_errmag, all_flag):
    """

    :param ifig:
    :param all_airmass:
    :param all_datetime:
    :param all_lambdas:
    :param all_abs:
    :param all_errabs:
    :param all_flag:
    :return:
    """

    fig = plt.figure(num=ifig, figsize=(16, 8))


    #---------------------

    # wavelength bin colors
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=NBWLBIN)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(NBWLBIN), alpha=1)

    myFmt = mdates.DateFormatter('%d-%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)


    # loop on events
    # for idx in goup_idx:
    idx=0
    for time in all_datetime:
        thewl = all_lambdas[idx]
        theabs = all_mag[idx]
        theerrabs = all_errmag[idx]
        wlcolors = []
        flag = all_flag[idx]

        if flag:

            for w0 in thewl:
                ibin = GetWLBin(w0)

                if ibin >= 0:
                    colorVal = scalarMap.to_rgba(ibin, alpha=1)
                else:
                    colorVal = scalarMap.to_rgba(0, alpha=0.25)

                wlcolors.append(colorVal)

            listOfdates=[ all_datetime[idx] for k in np.arange(len(theabs))]
            #print(listOfdates)
            plt.scatter(listOfdates, theabs, marker="o", c=wlcolors)

        idx+=1


    plt.plot([all_datetime[IDXMINREF], all_datetime[IDXMINREF]], [24, 32], "g-")
    plt.plot([all_datetime[IDXMAXREF], all_datetime[IDXMAXREF]], [24, 32], "g-")


    myFmt = mdates.DateFormatter('%d-%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

    plt.ylim(24, 32.)
    plt.grid(True, color="r")

    plt.title("Instrumental magnitudes $M(\lambda)$ vs date")
    plt.xlabel("date (UTC)")
    plt.ylabel("instr magnitudes (mag)")
    plt.gcf().autofmt_xdate()

    plt.xlim(all_datetime[0], all_datetime[-1])


    plt.show()




#-------------------------------------------------------------------------------------


def PlotAbsvsUTCAM(ifig, all_airmass,all_datetime, all_lambdas, all_abs, all_errabs, all_flag):
    """

    :param ifig:
    :param all_airmass:
    :param all_datetime:
    :param all_lambdas:
    :param all_abs:
    :param all_errabs:
    :param all_flag:
    :return:
    """

    fig = plt.figure(num=ifig, figsize=(16, 8))


    #---------------------

    # wavelength bin colors
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=NBWLBIN)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(NBWLBIN), alpha=1)

    myFmt = mdates.DateFormatter('%d-%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)


    # loop on events
    # for idx in goup_idx:
    idx=0
    for time in all_datetime:
        thewl = all_lambdas[idx]
        theabs = all_abs[idx]
        theerrabs = all_errabs[idx]
        wlcolors = []
        flag = all_flag[idx]

        if flag:

            for w0 in thewl:
                ibin = GetWLBin(w0)

                if ibin >= 0:
                    colorVal = scalarMap.to_rgba(ibin, alpha=1)
                else:
                    colorVal = scalarMap.to_rgba(0, alpha=0.25)

                wlcolors.append(colorVal)

            listOfdates=[ all_datetime[idx] for k in np.arange(len(theabs))]
            #print(listOfdates)
            plt.scatter(listOfdates, theabs, marker="o", c=wlcolors)

        idx+=1


    plt.plot([all_datetime[IDXMINREF], all_datetime[IDXMINREF]], [24, 32], "g-")
    plt.plot([all_datetime[IDXMAXREF], all_datetime[IDXMAXREF]], [24, 32], "g-")


    myFmt = mdates.DateFormatter('%d-%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

    plt.ylim(24, 32.)
    plt.grid(True, color="r")

    plt.title("Grey Abs = $M(\lambda)-K(\lambda).Z$ vs date")
    plt.xlabel("date (UTC)")
    plt.ylabel("absorption $m-k(\lambda).z$ (mag)")
    plt.gcf().autofmt_xdate()

    plt.xlim(all_datetime[0], all_datetime[-1])


    plt.show()




#--------------------------------------------------------------------------------------------

def ComputeRefPointAbs(idxminref,idxmaxref,all_lambdas,all_abs,all_errabs):
    """
    ComputeRefPointAbs(idxmaxref,idxminref,all_lambas,all_abs,all_errabs) : Compute the absorption of reference points in wavelength bins

    between indexes

    :param idxmaxref: max index
    :param idxminref: min index
    :param all_lambas: wavelengths
    :param all_abs:    absorption
    :param all_errabs: error on absorption
    :return:
               Attenuation_Ref_mean,
               Attenuation_Ref_std,
               Attenuation_Ref_err

    """


    assert idxminref <=  idxmaxref

    NBIDXREF=idxmaxref-idxminref+1


    # the attenuation will be average inside a wavelength bin
    Attenuation_Ref=np.zeros((NBWLBIN,NBIDXREF))  # sum of attenuation for each point (wl,abs) in the bins
    NAttenuation_Ref = np.zeros((NBWLBIN, NBIDXREF)) # counters on the number of entries inside these bins
    Attenuation_Ref_Err = np.zeros((NBWLBIN, NBIDXREF)) # sum of error squared


    # double loop on reference idx, wlbin to compute attenuation at reference point
    for idx in np.arange(idxminref,idxmaxref+1):
        print("---------------------------------------------------------------------------------------")
        print(idx)


        thewl = all_lambdas[idx]
        theabs = all_abs[idx]
        theerrabs = all_errabs[idx]

        # loop on wavelength
        iw0=0   # index inside thewl array
        for w0 in thewl:
            iwlbin = GetWLBin(w0)
            if iwlbin>=0 and theabs[iw0]!=0:
                if iwlbin==0:
                    print("iwlbin={}, .... theabs={}".format(iwlbin,theabs[iw0]))
                Attenuation_Ref[iwlbin,idx-idxminref]+=theabs[iw0]
                NAttenuation_Ref[iwlbin, idx-idxminref] +=1
                Attenuation_Ref_Err[iwlbin, idx - idxminref] += theerrabs[iw0]**2
            iw0+=1 # increase counter in thewlarray

    # normalize the attenuation and error sum squared to the number of entries
    Attenuation_Ref=np.where(NAttenuation_Ref>=1, Attenuation_Ref/NAttenuation_Ref,0)
    Attenuation_Ref_Err = np.where(NAttenuation_Ref >= 1, Attenuation_Ref_Err / NAttenuation_Ref, 0)

    #print("Attenuation_Ref[0,:]=",Attenuation_Ref[0,:])

    #because some observation are empty in a given bin, the empty bins are masked
    mask=Attenuation_Ref==0
    #print("mask=",mask)

    #masked attenuation
    MAttenuation_Ref = np.ma.masked_array(Attenuation_Ref, mask)

    #compute average attenuation over good obs
    Attenuation_Ref_mean=np.average(MAttenuation_Ref,axis=1)

    #print("Attenuation_Ref_mean[0]=", Attenuation_Ref_mean[0])

    #compute attenuation standard error over non empty bins
    Attenuation_Ref_std = np.std(MAttenuation_Ref, axis=1)

    # masked attenuation error
    MAttenuation_Ref_err = np.ma.masked_array(Attenuation_Ref_Err, mask)

    #compute average error over good bins
    Attenuation_Ref_err = np.sqrt(np.average(MAttenuation_Ref_err, axis=1))

    Lambdas_ref=WLMEANBIN

    print("Attenuation_Ref_mean",Attenuation_Ref_mean)
    print("Attenuation_Ref_std", Attenuation_Ref_std)
    print("Attenuation_Ref_err", Attenuation_Ref_err)


    return Lambdas_ref,Attenuation_Ref_mean, Attenuation_Ref_std, Attenuation_Ref_err
#-------------------------------------------------------------------------------------------

def ComputeRelativeAbs(all_indexes, all_lambdas, all_abs, all_errabs,all_flag,Attenuation_Ref_mean):
    """

    ComputeRelativeAbs(all_indexes, all_lambdas, all_abs, all_errabs,Attenuation_Ref_mean): Compute absorption
    relative to reference absorption

    :param all_indexes:
    :param all_lambdas:
    :param all_abs:
    :param all_errabs:
    :param Attenuation_Ref_mean:
    :return: MAttenuation_mean_ALL,  MAttenuation_Err_ALL
    """


    IDXMIN = all_indexes.min()
    IDXMAX = all_indexes.max()
    NBIDX = IDXMAX - IDXMIN + 1

    Attenuation_all = np.zeros((NBWLBIN, NBIDX))
    NAttenuation_all = np.zeros((NBWLBIN, NBIDX))
    Attenuation_Err_all = np.zeros((NBWLBIN, NBIDX))
    Attenuation_mean_all = np.zeros((NBWLBIN, NBIDX))


    # loop on all observations
    for idx in np.arange(IDXMIN, IDXMAX + 1):

        thewl = all_lambdas[idx]
        theabs = all_abs[idx]
        theerrabs = all_errabs[idx]
        flag=all_flag[idx]

        if flag:

            # loop on wavelength
            iw0 = 0 # index in thewl array
            for w0 in thewl:
                iwlbin = GetWLBin(w0)
                if iwlbin >= 0 and theabs[iw0] != 0:
                    Attenuation_all[iwlbin, idx - IDXMIN] += theabs[iw0]
                    NAttenuation_all[iwlbin, idx - IDXMIN] += 1
                    Attenuation_Err_all[iwlbin, idx - IDXMIN] += theerrabs[iw0] ** 2
                iw0 += 1

    # normalize the Attenuation sum
    Attenuation_all = np.where(NAttenuation_all >= 1, Attenuation_all / NAttenuation_all, 0)
    # root squared of average squared errors
    Attenuation_Err_all = np.sqrt(np.where(NAttenuation_all >= 1, Attenuation_Err_all / NAttenuation_all, 0))


    # Mask for exactly zero attenuation
    mask = Attenuation_all == 0



    # Express attenuation wrt reference point
    Attenuation_mean_ALL = Attenuation_all - Attenuation_Ref_mean[:, np.newaxis]
    Attenuation_Err_ALL= Attenuation_Err_all

    print("Attenuation_mean_ALL.shape:", Attenuation_mean_ALL.shape)
    print("Attenuation_Err_all.shape:", Attenuation_Err_all.shape)

    # masked attenuations
    MAttenuation_mean_ALL = np.ma.masked_array(Attenuation_mean_ALL, mask)
    MAttenuation_Err_ALL  = np.ma.masked_array(Attenuation_Err_ALL,mask)

    return  MAttenuation_mean_ALL,  MAttenuation_Err_ALL,mask


#-----------------------------------------------------------------------------------------------------------------
def ComputeRelativeAbsMedian(WL0,DWL0, MAttenuation_mean_ALL):
    """

    :param WL0:
    :param DWL0:
    :param MAttenuation_mean_ALL:
    :return:
    """

    # get the series of wavelengths

    SelWLBinIndexes= np.where(np.logical_and(WLMINBIN >= WL0-DWL0, WLMAXBIN < WL0+DWL0))[0]

    # get all relevant
    SelAttenuations=MAttenuation_mean_ALL[SelWLBinIndexes,:]

    all_medians=np.median(SelAttenuations,axis=0)

    return all_medians



#-----------------------------------------------------------------------------------------------------------------


def PlotReferencePointAbs(ifig,Lambdas_ref,Attenuation_Ref_mean,Attenuation_Ref_std,Attenuation_Ref_err):
    """

    :param ifig:
    :param Lambdas_ref:
    :param Attenuation_Ref_mean:
    :param Attenuation_Ref_std:
    :param ttenuation_Ref_err:
    :return:
    """

    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    plt.errorbar(Lambdas_ref+1.0,Attenuation_Ref_mean,yerr=Attenuation_Ref_std,ecolor="k",fmt=".")
    plt.errorbar(Lambdas_ref-1.0, Attenuation_Ref_mean, yerr=Attenuation_Ref_err, ecolor="r", fmt=".")
    plt.plot(Lambdas_ref, Attenuation_Ref_mean, "o-b")
    plt.xlabel("$\lambda$ (nm)")
    plt.ylabel("Absorption at z=1")
    plt.title("Absorption reference Point wrt wavelength")
    #plt.ylim(10, 40.)
    plt.ylim(24, 32.)
    plt.grid(True, color="r")
    plt.show()

#-------------------------------------------------------------------------------------------


def PlotRelativeAbsvsIndexes(ifig,all_indexes,MAttenuation_mean_ALL):
    """

    :param ifig:
    :param all_indexes:
    :param MAttenuation_mean_ALL:
    :return:
    """


    plt.figure(num=ifig, figsize=(16, 10))

    # Loop on wavelength bins
    for iwlbin in np.arange(NBWLBIN):
        colorVal = scalarMap.to_rgba(iwlbin, alpha=1)


        #plt.plot(all_indexes, Attenuation_mean_ALL[iwlbin, :], "o", color=colorVal)
        plt.plot(all_indexes, MAttenuation_mean_ALL[iwlbin, :], "o", color=colorVal)

    plt.plot([IDXMINREF, IDXMINREF], [1., 1], "k-")
    plt.plot([IDXMAXREF, IDXMAXREF], [-1., 1.], "k-")


    plt.grid(True, color="r")
    plt.xlabel("Event Number")
    plt.ylabel("Attenuation (mag)")
    plt.title("Attenuation relative to reference point")
    plt.ylim(-1., 1.)
    plt.legend()
    plt.show()
#-------------------------------------------------------------------------------------------

def PlotRelativeAbsvsIndexeswthErr(ifig, all_indexes, MAttenuation_mean_ALL):
    """

    :param ifig:
    :param all_indexes:
    :param MAttenuation_mean_ALL:
    :return:
    """

    plt.figure(num=ifig, figsize=(16, 10))
    # Loop on wavelength bins
    for iwlbin in np.arange(NBWLBIN):
        colorVal = scalarMap.to_rgba(iwlbin, alpha=1)


        plt.errorbar(all_indexes, MAttenuation_mean_ALL[iwlbin, :], yerr=MAttenuation_Err_ALL[iwlbin, :], ecolor="grey",
                     color=colorVal, fmt=".")

    plt.plot([IDXMINREF, IDXMINREF], [1., 1], "k-")
    plt.plot([IDXMAXREF, IDXMAXREF], [-1., 1.], "k-")


    plt.grid(True, color="r")
    plt.xlabel("Event Number")
    plt.ylabel("Attenuation (mag)")
    plt.title("Attenuation relative to reference point")
    plt.ylim(-1., 1.)
    plt.legend()
    plt.show()

#---------------------------------------------------------------------------------------------

def PlotRelativeAbsvsUTC(ifig,all_datetime,MAttenuation_mean_ALL, MAttenuation_Err_ALL):
    """

    :param ifig:
    :param all_datetime:
    :param MAttenuation_mean_ALL:
    :param MAttenuation_Err_ALL:
    :return:
    """




    plt.figure(num=ifig, figsize=(16, 10))

    # Loop on wavelength bins
    for iwlbin in np.arange(NBWLBIN):
        colorVal = scalarMap.to_rgba(iwlbin, alpha=1)

        plt.errorbar(all_datetime, MAttenuation_mean_ALL[iwlbin, :], yerr=MAttenuation_Err_ALL[iwlbin, :], ecolor="grey",
                     color=colorVal, fmt=".")

    plt.plot([all_datetime[IDXMINREF], all_datetime[IDXMINREF]], [1., 1], "g-")
    plt.plot([all_datetime[IDXMAXREF], all_datetime[IDXMAXREF]], [-1., 1.], "g-")

    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%d-%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

    plt.grid(True, color="r")
    plt.xlabel("Observation time (UTC)")
    plt.ylabel("Attenuation (mag)")
    plt.title("Attenuation relative to reference point")
    plt.ylim(-1., 1.)

    plt.xlim(all_datetime[0], all_datetime[-1])

    plt.legend()
    plt.show()
#-----------------------------------------------------------------------------------------------------------------

def SmoothGreyCorrAbsvsUTC(dt, referencebasedattenuation):
    """

    :param ifig:
    :param dt:
    :param referencebasedattenuation:
    :return:
    """

    NbObs=len(dt)

    # from scipy.interpolate import interp1d
    smoothed = interp1d(dt, referencebasedattenuation, kind='cubic')

    # numpy
    #referenceout=np.interp(dt,dt,referencebasedattenuation)
    # scipy
    referenceout = smoothed(dt)



    return referenceout

#----------------------------------------------------------------------------------------------------------------------
def PlotGreyCorrAbsvsUTC(ifig, all_datetime, referencebasedattenuation):
    """

    :param ifig:
    :param all_datetime:
    :param MAttenuation_mean_ALL:
    :param MAttenuation_Err_ALL:
    :param referencebasedattenuation:
    :return:
    """

    plt.figure(num=ifig, figsize=(16, 10))

    refmin=np.median(referencebasedattenuation)

    print("refmin=",refmin)

    refref=referencebasedattenuation - refmin

    plt.plot(all_datetime, refref,"ro")

    plt.plot([all_datetime[IDXMINREF], all_datetime[IDXMINREF]], [refref.min(), refref.max()], "g-")
    plt.plot([all_datetime[IDXMAXREF], all_datetime[IDXMAXREF]], [refref.min(), refref.max()], "g-")

    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%d-%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

    plt.grid(True, color="r")
    plt.xlabel("Observation time (UTC)")
    plt.ylabel("Attenuation (mag)")
    plt.title("Grey attenuation (median@700 nm)")

    plt.xlim(all_datetime[0], all_datetime[-1])

    #plt.ylim(-1., 1.)
    plt.legend()
    plt.show()




#----------------------------------------------------------------------------------------------------------------------
def PlotGreyCorrRelativeAbsvsUTC(ifig, all_datetime, MAttenuation_mean_ALL, MAttenuation_Err_ALL, referencebasedattenuation):
    """

    :param ifig:
    :param all_datetime:
    :param MAttenuation_mean_ALL:
    :param MAttenuation_Err_ALL:
    :param referencebasedattenuation:
    :return:
    """

    plt.figure(num=ifig, figsize=(16, 10))

    # Loop on wavelength bins
    for iwlbin in np.arange(NBWLBIN):
        colorVal = scalarMap.to_rgba(iwlbin, alpha=1)

        plt.errorbar(all_datetime, MAttenuation_mean_ALL[iwlbin, :]-referencebasedattenuation, yerr=MAttenuation_Err_ALL[iwlbin, :],
                     ecolor="grey",
                     color=colorVal, fmt=".")

    plt.plot([all_datetime[IDXMINREF], all_datetime[IDXMINREF]], [1., 1], "g-")
    plt.plot([all_datetime[IDXMAXREF], all_datetime[IDXMAXREF]], [-1., 1.], "g-")

    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%d-%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

    plt.grid(True, color="r")
    plt.xlabel("Observation time (UTC)")
    plt.ylabel("Attenuation (mag)")
    plt.title("Attenuation relative to reference point corrected from grey att")

    plt.xlim(all_datetime[0], all_datetime[-1])

    plt.ylim(-1., 1.)
    plt.xlim(all_datetime[0], all_datetime[-1])
    plt.legend()
    plt.show()



#----------------------------------------------------------------------------------------------------------------------
def PlotGreyCorrRelativeAbsvsIndexes(ifig,  all_indexes ,MAttenuation_mean_ALL, MAttenuation_Err_ALL, referencebasedattenuation):
    """

    :param ifig:
    :param all_datetime:
    :param MAttenuation_mean_ALL:
    :param MAttenuation_Err_ALL:
    :param referencebasedattenuation:
    :return:
    """

    plt.figure(num=ifig, figsize=(16, 10))

    # Loop on wavelength bins
    for iwlbin in np.arange(NBWLBIN):
        colorVal = scalarMap.to_rgba(iwlbin, alpha=1)

        plt.errorbar(all_indexes, MAttenuation_mean_ALL[iwlbin, :]-referencebasedattenuation, yerr=MAttenuation_Err_ALL[iwlbin, :],
                     ecolor="grey",
                     color=colorVal, fmt=".")

    plt.plot([IDXMINREF, IDXMINREF], [1., 1], "g-")
    plt.plot([IDXMAXREF, IDXMAXREF], [-1., 1.], "g-")



    plt.grid(True, color="r")
    plt.xlabel("Event number")
    plt.ylabel("Attenuation (mag)")
    plt.title("Attenuation relative to reference point corrected from grey att")
    plt.ylim(-1., 1.)
    plt.legend()
    plt.show()

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


    all_indexes, all_eventnum, all_airmass, all_lambdas, all_flux, all_errflux, all_mag, all_errmag, all_abs, all_errabs, all_dt, all_datetime, all_flag, all_badidx, all_badfn=\
        ReadAllFiles(input_directory, onlyfilesspectrum)


    ################################################
    # 3) Compute airmasss min
    #####################################################

    print("len(all_mairmass)=", len(all_airmass))
    #print("all_airmass=",all_airmass)

    # find where is zmin
    zmin_idx=np.where(all_airmass==all_airmass.min())[0][0]
    zmin=all_airmass[zmin_idx]


    print("zmin_idx...............=",zmin_idx)
    print("zmin...................=", zmin)

    #series of indexes for which z decrease (star rise)
    godown_idx = np.where(np.arange(len(all_airmass)) <= zmin_idx)[0]
    # series of indexes for which z increase (star fall)
    goup_idx = np.where(np.arange(len(all_airmass)) >= zmin_idx)[0]

    print('len(all_indexes).......=', len(all_indexes))
    print('all_indexes............=', all_indexes)
    print('godown_idx.............=', godown_idx)
    print('goup_idx...............=', goup_idx)

    airmass_godown = all_airmass[godown_idx]
    airmass_goup = all_airmass[goup_idx]

    event_number_godown=all_indexes[godown_idx]
    event_number_goup = all_indexes[goup_idx]

    print('event num godown_idx.............=', event_number_godown)
    print('event num goup_idx...............=', event_number_goup)


    print(">>>>> Bad indexes=",all_badidx)
    #print(">>>>> Bad files = ", all_badfn)


    # Initialize Figure numbers
    ifig=800

    ################################################
    # 4) Plot absorption vs Index number
    #####################################################
    # TOO LONG
    #PlotAbsvsIndex(ifig, all_indexes, all_lambdas, all_abs, all_errabs, all_flag)
    PlotAMvsUTC(ifig, all_airmass, all_datetime, all_flag)
    ifig+=1
    #PlotAbsvsUTCAM(ifig, all_airmass,all_datetime, all_lambdas, all_abs, all_errabs, all_flag)
    PlotMagvsUTCAM(ifig, all_airmass,all_datetime, all_lambdas, all_mag, all_errmag, all_flag)
    ifig+=1
    ################################################
    #  5) Compute REFERENCE POINT :::: Wavelength dependence of reference magnitude point
    #################################################

    Lambdas_ref,Attenuation_Ref_mean, Attenuation_Ref_std, Attenuation_Ref_err=ComputeRefPointAbs(IDXMINREF, IDXMAXREF, all_lambdas, all_abs, all_errabs)

    ################################################
    # 6) Plot REFRENCE POINT correction
    #####################################################
    ifig+=1
    PlotReferencePointAbs(ifig, Lambdas_ref, Attenuation_Ref_mean, Attenuation_Ref_std, Attenuation_Ref_err)


    #################################################
    # 7) Compute Relative Attenuation for the whole observation range
    #################################################
    MAttenuation_mean_ALL, MAttenuation_Err_ALL,mask=ComputeRelativeAbs(all_indexes, all_lambdas, all_abs, all_errabs,all_flag,Attenuation_Ref_mean)


    #################################################################
    #  8)  : Masked attenuation without error bars
    ######################################################################
    ifig+=1
    PlotRelativeAbsvsIndexes(ifig, all_indexes, MAttenuation_mean_ALL)

    #################################################################
    #  9)  : Masked attenuation with error bars
    ######################################################################
    ifig += 1
    PlotRelativeAbsvsIndexeswthErr(ifig, all_indexes, MAttenuation_mean_ALL)

    #################################################################
    #  10)  : Masked attenuation  wrt UTC
    ######################################################################
    ifig += 1
    PlotRelativeAbsvsUTC(ifig, all_datetime, MAttenuation_mean_ALL, MAttenuation_Err_ALL)

    #################################################################
    #  11)  : Grey attenuation
    ######################################################################
    referencebasedattenuation=ComputeRelativeAbsMedian(700, 40, MAttenuation_mean_ALL)

    #################################################################
    #  12)  : Plot Grey attenuation
    ######################################################################
    ifig += 1
    referenceout=SmoothGreyCorrAbsvsUTC(all_dt, referencebasedattenuation)
    PlotGreyCorrAbsvsUTC(ifig, all_datetime, referenceout)


    #################################################################
    #  13)  : Masked attenuation  wrt UTC
    ######################################################################
    ifig += 1
    PlotGreyCorrRelativeAbsvsUTC(ifig, all_datetime, MAttenuation_mean_ALL, MAttenuation_Err_ALL, referencebasedattenuation)

    #################################################################
    #  14)  : Masked attenuation  wrt Event number
    ######################################################################
    ifig += 1
    PlotGreyCorrRelativeAbsvsIndexes(ifig, all_indexes, MAttenuation_mean_ALL, MAttenuation_Err_ALL,referencebasedattenuation)




    # Save  tables



    general_info=np.c_[all_indexes,all_eventnum,all_airmass,all_dt,all_datetime,referencebasedattenuation,all_flag]

    np.save(os.path.join(outputtabledir,"Info.npy"),general_info)
    np.save(os.path.join(outputtabledir,"Lambdas_ref.npy"),Lambdas_ref)
    np.save(os.path.join(outputtabledir,"Mask.npy"), mask)
    np.save(os.path.join(outputtabledir,"MAttenuation_mean_ALL.npy"), MAttenuation_mean_ALL.compressed())
    np.save(os.path.join(outputtabledir,"MAttenuation_Err_ALL.npy"), MAttenuation_Err_ALL.compressed())


    print(general_info)


#----------------------------------------------------------------------------------------------------------------------