#########################################################################################################
# Study grey attenuation
#########################################################################################################

import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import re

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import numpy as np

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

WLMIN = 380.0
WLMAX = 1000.0
NBWLBIN = 62
WLBINWIDTH = (WLMAX - WLMIN) / float(NBWLBIN)

WLMINBIN = np.arange(WLMIN, WLMAX, WLBINWIDTH)
WLMAXBIN = np.arange(WLMIN + WLBINWIDTH, WLMAX + WLBINWIDTH, WLBINWIDTH)
WLMEANBIN=(WLMINBIN + WLMAXBIN)/2.



print('WLMINBIN..................................=', WLMINBIN.shape, WLMINBIN)
print('WLMAXBIN..................................=', WLMAXBIN.shape, WLMAXBIN)
print('NBWLBIN...................................=', NBWLBIN)
print('WLBINWIDTH................................=', WLBINWIDTH)

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


#-------------------------------------------------------------------------
#
# MAIN()
#
#---------------------------------------------------

if __name__ == "__main__":

    #####################
    # 1) Configuration
    ######################

    parameters.VERBOSE = True
    parameters.DEBUG = True

    #thedate="20190214"
    thedate = "20190215"

    #output_directory = "output/" + thedate
    #output_directory = "/Users/dagoret/DATA/PicDuMidiFev2019/spectractor_spectra/" + thedate
    output_directory = "/Users/dagoret/DATA/PicDuMidiFev2019/spectractor_output_deco/"+ thedate
    #output_directory = "/Users/dagoret/DATA/PicDuMidiFev2019/spectractor_output_deco2/" + thedate

    parameters.VERBOSE = True
    parameters.DISPLAY = True




    ############################
    # 2) Get the config
    #########################

    config = spectractordir+"/config/picdumidi.ini"

    my_logger = set_logger(__name__)
    my_logger.info('\n\tStart RunViewSpectra')
    # Load config file
    load_config(config)



    ############################
    # 3) Get Spectra filelist
    #########################


    # get all files
    onlyfiles = [f for f in listdir(output_directory) if isfile(join(output_directory, f))]
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
            output_filename = os.path.join(output_directory, output_filename)
            output_filename_spectrogram = output_filename.replace('spectrum', 'spectrogram')
            output_filename_psf = output_filename.replace('spectrum.fits', 'table.csv')



            # go to next simulation if output files already exists
            if os.path.isfile(output_filename) and os.path.isfile(output_filename_spectrogram) and os.path.isfile(
                        output_filename_psf):
                filesize = os.stat(output_filename).st_size
                print(">>>>> output filename : {} already exists with size {} ".format(output_filename,filesize))

                filesize = os.stat(output_filename_spectrogram).st_size
                print(">>>>> output filename : {} already exists with size {} ".format(output_filename_spectrogram,filesize))

                filesize = os.stat(output_filename_psf).st_size
                print(">>>>> output filename : {} already exists with size {} ".format(output_filename_psf,filesize))

                onlyfilesspectrum.append(re.findall("(^T.*_spectrum.fits$)", file_name)[0])




    # sort again all the files
    onlyfilesspectrum = np.array(onlyfilesspectrum)
    sortedindexes = np.argsort(onlyfilesspectrum)
    onlyfilesspectrum = onlyfilesspectrum[sortedindexes]


    #get basnemae of files for later use (to check if _table.csv and _spectrogram.fits exists
    onlyfilesbasename=[]
    for f in onlyfilesspectrum:
        onlyfilesbasename.append(re.findall("(^T.*)_spectrum.fits$",f)[0])


    basenamecut=[]
    for f in onlyfilesspectrum:
        basenamecut.append(f.split("_HD")[0])


    #############################################
    # 3) Read fits file
    ##########################################

    # parameters
    NBSPEC = len(sortedindexes)

    print('NBSPEC....................................= ', NBSPEC)


    #assert False


    all_indexes=[]
    all_airmass=[]
    all_flag=[]
    all_flux=[]
    all_flux_err=[]
    all_lambdas=[]
    all_mag=[]
    all_errmag=[]
    all_abs=[]
    all_errabs=[]
    all_dt=[]
    all_badidx=[]
    all_badfn=[]

    #----------------------------------
    # Extract spectra information from files
    # compute magnitudes
    #-----------------------------
    for idx in np.arange(0, NBSPEC):
        # if idx in [0,1,4]:
        #    continue

        #print("{}) : {}".format(idx,onlyfilesspectrum[idx]))

        fullfilename = os.path.join(output_directory, onlyfilesspectrum[idx])
        #try:
        if 1:
            hdu = fits.open(fullfilename)
            header=hdu[0].header

            am=header["AIRMASS"]
            date=header["DATE-OBS"]
            if idx==0:
                T0=t = Time(date, format='isot', scale='utc')
            T=Time(date, format='isot', scale='utc')
            DT=(T-T0).sec/(3600.0)

            data=hdu[0].data

            # extract wavelength, spectrum and errors
            wavelength=data[0,:]
            spec = data[1,:]
            err=data[2,: ]


            # sort the wavelengths
            wl_sorted_idx=np.argsort(wavelength)

            wl=wavelength[wl_sorted_idx]
            fl=spec[wl_sorted_idx]
            errfl=err[wl_sorted_idx]
            wlbins = [GetWLBin(w) for w in wl]
            wlbins = np.array(wlbins)


            # defines good measurmements as flux >0 and wavelength in selected bons
            goodpoints=np.where(np.logical_and(fl != 0, wlbins != -1))

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

            absmin = abs.min()
            absmax = abs.max()

            # Set a quality flag
            if absmin<25 or absmax> 32:
                print("file index idx = ",idx,"==>  filename = ",onlyfilesspectrum[idx]," absmin= ",absmin," absmax = ",absmax)
                all_flag.append(False)
                all_badidx.append(idx)
                all_badfn.append(onlyfilesspectrum[idx])
            else:
                all_flag.append(True)


            # save for each observation  { event-id, airmass, set of points (wl,flux,errflux, mag,abs,errabs) }
            if(len(mag>0)):
                all_indexes.append(idx)
                all_airmass.append(am)
                all_lambdas.append(wl)
                all_flux.append(fl)
                all_flux_err.append(errfl)
                all_mag.append(mag)
                all_errmag.append(errmag)
                all_abs.append(abs)
                all_errabs.append(errabs)
                all_dt.append(DT)

        #except:
        if 0:
            print("Unexpected error:", sys.exc_info()[0])
            pass

    # transform container in numpy arrays
    all_indexes=np.array(all_indexes)
    all_airmass = np.array(all_airmass)
    all_flux = np.array(all_flux)
    all_flux_err = np.array(all_flux_err)
    all_lambdas = np.array(all_lambdas)
    all_mag=np.array(all_mag)
    all_errmag=np.array(all_errmag)
    all_dt=np.array(all_dt)


    #assert False

    print(all_airmass)

    # find where is zmin
    zmin_idx=np.where(all_airmass==all_airmass.min())[0][0]
    zmin=all_airmass[zmin_idx]


    print("zmin_idx...............=",zmin_idx)
    print("zmin...................=", zmin)

    #series of indexes for which z decrease (star rise)
    godown_idx = np.where(np.arange(len(all_airmass)) <= zmin_idx)[0]
    # series of indexes for which z increase (star fall)
    goup_idx = np.where(np.arange(len(all_airmass)) >= zmin_idx)[0]


    print('godown_idx.............=', godown_idx)
    print('goup_idx...............=', goup_idx)

    airmass_godown = all_airmass[godown_idx]
    airmass_goup = all_airmass[goup_idx]

    event_number_godown=all_indexes[godown_idx]
    event_number_goup = all_indexes[goup_idx]

    print('event num godown_idx.............=', event_number_godown)
    print('event num goup_idx...............=', event_number_goup)


    print(">>>>> Bad indexes=",all_badidx)
    print(">>>>> Bad files = ", all_badfn)


    # Figure numbers
    ifig=600


    #---------------------------------------------------------------------------------------------------
    # Search for the reference point
    #--------------------------------------------------------------------------------------------------


    # bins in wevelength
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=NBWLBIN)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(NBWLBIN), alpha=1)

    # ---------------------------------------
    #  Figure 1: attenuation vs airmass goup
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    # loop on wavelength bins

    # loop on all observation time ordered
    for idx in goup_idx:
        thewl = all_lambdas[idx]
        themag = all_mag[idx]
        theerrmag = all_errmag[idx]
        wlcolors = []
        am = all_airmass[idx]
        flag=all_flag[idx]

        #print(idx)
        #print(themag)
        #print(theerrmag)

        if flag:


            for w0 in thewl:
                ibin = GetWLBin(w0)

                if ibin >= 0:
                    colorVal = scalarMap.to_rgba(ibin, alpha=1)
                else:
                    colorVal = scalarMap.to_rgba(0, alpha=1)

                wlcolors.append(colorVal)

            #plt.errorbar(np.ones(len(themag)) * am, themag, yerr=theerrmag,color="grey" ,ecolor="grey", fmt=".")
            plt.scatter(np.ones(len(themag)) * am, themag, marker="o", c=wlcolors)
        else:
            for w0 in thewl:
                ibin = GetWLBin(w0)


                if ibin >= 0:
                    colorVal = scalarMap.to_rgba(ibin, alpha=1)
                else:
                    colorVal = scalarMap.to_rgba(0, alpha=1)

                wlcolors.append(colorVal)


            #DO NOT PLOT BADLY RECONSTRUCTED SPECTRA
            # plt.errorbar(np.ones(len(themag)) * am, themag, yerr=theerrmag,color="grey" ,ecolor="grey", fmt=".")
            #plt.scatter(np.ones(len(themag)) * am, themag, marker="x", c=wlcolors)




    plt.ylim(20, 35.)
    plt.grid(True, color="r")
    plt.title("Instrumental Magnitude vs airmass (z decrease)")
    plt.xlabel("airmass")
    plt.ylabel("magnitude (mag)")
    plt.show()

    # ---------------------------------------
    #  Figure 2 : Magnitude corrigée de Rayleigh pour star falling vs airmass
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    # loop on wavelength bins

    # loop on events
    for idx in goup_idx:
        thewl=all_lambdas[idx]
        theabs=all_abs[idx]
        theerrabs=all_errabs[idx]
        wlcolors=[]
        am=all_airmass[idx]


        for w0 in thewl:
            ibin=GetWLBin(w0)

            if ibin>=0:
                colorVal = scalarMap.to_rgba(ibin, alpha=1)
            else:
                colorVal = scalarMap.to_rgba(0, alpha=1)

            wlcolors.append(colorVal)

        plt.scatter(np.ones(len(theabs))*am,theabs,marker="o",c=wlcolors)
        #plt.errorbar(np.ones(len(theabs))*am,theabs,yerr=theerrabs,ecolor="k",fmt=".")


    plt.ylim(20,35.)
    plt.grid(True,color="r")

    plt.title("Grey Abs = $M(\lambda)-K(\lambda).Z$ vs airmass (star falling)")
    plt.xlabel("airmass")
    plt.ylabel("absorption $m-k(\lambda).z$ (mag)")

    plt.show()

    # ---------------------------------------
    #  Figure 3 : Magnitude corrigée de Rayleigh pour star falling vs event number
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    # loop on wavelength bins

    # loop on events
    for idx in goup_idx:
        thewl=all_lambdas[idx]
        theabs=all_abs[idx]
        theerrabs=all_errabs[idx]
        wlcolors=[]
        am=all_airmass[idx]


        for w0 in thewl:
            ibin=GetWLBin(w0)

            if ibin>=0:
                colorVal = scalarMap.to_rgba(ibin, alpha=1)
            else:
                colorVal = scalarMap.to_rgba(0, alpha=1)

            wlcolors.append(colorVal)

        plt.scatter(np.ones(len(theabs))*idx,theabs,marker="o",c=wlcolors)
        #plt.errorbar(np.ones(len(theabs))*am,theabs,yerr=theerrabs,ecolor="k",fmt=".")

    plt.plot([292,292],[0,35],"k-")
    plt.plot([303, 303], [0, 35], "k-")

    plt.ylim(20,35.)
    plt.grid(True,color="r")

    plt.title("Grey Abs = $M(\lambda)-K(\lambda).Z$ vs event number (star falling)")
    plt.xlabel("Event number")
    plt.ylabel("absorption $m-k(\lambda).z$ (mag)")

    plt.show()

    # ---------------------------------------
    #  Figure 4 : Absorption relative for star  vs event number
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    # loop on wavelength bins

    # loop on events
    for idx in all_indexes:
        thewl = all_lambdas[idx]
        theabs = all_abs[idx]
        theerrabs = all_errabs[idx]
        wlcolors = []
        am = all_airmass[idx]

        for w0 in thewl:
            ibin = GetWLBin(w0)

            if ibin >= 0:
                colorVal = scalarMap.to_rgba(ibin, alpha=1)
            else:
                colorVal = scalarMap.to_rgba(0, alpha=1)

            wlcolors.append(colorVal)

        plt.scatter(np.ones(len(theabs)) * idx, theabs, marker="o", c=wlcolors)
        # plt.errorbar(np.ones(len(theabs))*am,theabs,yerr=theerrabs,ecolor="k",fmt=".")

    plt.plot([292, 292], [0, 35], "k-")
    plt.plot([303, 303], [0, 35], "k-")

    plt.ylim(20, 35.)
    plt.grid(True, color="r")

    plt.title("Grey Abs = $M(\lambda)-K(\lambda).Z$ vs event number ")
    plt.xlabel("Event number")
    plt.ylabel("absorption $m-k(\lambda).z$ (mag)")

    plt.show()

    # ---------------------------------------
    #  Figure 5 : Magnitude corrigée de Rayleigh for star  vs airmass
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    # loop on wavelength bins

    # loop on events
    for idx in all_indexes:
        thewl = all_lambdas[idx]
        theabs = all_abs[idx]



        theerrabs = all_errabs[idx]
        wlcolors = []
        am = all_airmass[idx]

        for w0 in thewl:
            ibin = GetWLBin(w0)

            if ibin >= 0:
                colorVal = scalarMap.to_rgba(ibin, alpha=1)
            else:
                colorVal = scalarMap.to_rgba(0, alpha=1)

            wlcolors.append(colorVal)

        plt.scatter(np.ones(len(theabs)) * am, theabs, marker="o", c=wlcolors)
        # plt.errorbar(np.ones(len(theabs))*am,theabs,yerr=theerrabs,ecolor="k",fmt=".")



    plt.ylim(20, 35.)
    plt.grid(True, color="r")

    plt.title("Grey Abs = $M(\lambda)-K(\lambda).Z$ vs airmass ")
    plt.xlabel("Airmass")
    plt.ylabel("absorption $m-k(\lambda).z$ (mag)")

    plt.show()




    #------------------------------------------------------------------------------------
    #  REFERENCE POINT :::: Wavelength dependence of reference magnitude point
    #----------------------------------------------------------------------------------

    # For Referencepoint
    IDXMINREF=293
    IDXMAXREF=302
    NBIDXREF=IDXMAXREF-IDXMINREF+1

    # the attenuation will be average inside a wavelength bin
    Attenuation_Ref=np.zeros((NBWLBIN,NBIDXREF))  # sum of attenuation for each point (wl,abs) in the bins
    NAttenuation_Ref = np.zeros((NBWLBIN, NBIDXREF)) # counters on the number of entries inside these bins
    Attenuation_Ref_Err = np.zeros((NBWLBIN, NBIDXREF)) # sum of error squared


    # double loop on reference idx, wlbin to compute attenuation at reference point
    for idx in np.arange(IDXMINREF,IDXMAXREF):
        print("---------------------------------------------------------------------------------------")
        print(idx)
        thewl = all_lambdas[idx]
        theabs = all_abs[idx]
        theerrabs = all_errabs[idx]

        # loop on wavelength
        iw0=0
        for w0 in thewl:
            iwlbin = GetWLBin(w0)
            if iwlbin>=0 and theabs[iw0]!=0:
                if iwlbin==0:
                    print("iwlbin={}, .... theabs={}".format(iwlbin,theabs[iw0]))
                Attenuation_Ref[iwlbin,idx-IDXMINREF]+=theabs[iw0]
                NAttenuation_Ref[iwlbin, idx-IDXMINREF] +=1
                Attenuation_Ref_Err[iwlbin, idx - IDXMINREF] += theerrabs[iw0]**2
            iw0+=1

    # normalize the attenuation and error sum squared to the number of entries
    Attenuation_Ref=np.where(NAttenuation_Ref>=1, Attenuation_Ref/NAttenuation_Ref,0)
    Attenuation_Ref_Err = np.where(NAttenuation_Ref >= 1, Attenuation_Ref_Err / NAttenuation_Ref, 0)

    #print("Attenuation_Ref[0,:]=",Attenuation_Ref[0,:])

    #because some observation are empty in a given bin, the empty bins are masked
    mask=Attenuation_Ref==0
    print("mask=",mask)

    #masked attenuation
    MAttenuation_Ref = np.ma.masked_array(Attenuation_Ref, mask)

    #compute average attenuation over good obs
    Attenuation_Ref_mean=np.average(MAttenuation_Ref,axis=1)

    print("Attenuation_Ref_mean[0]=", Attenuation_Ref_mean[0])

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

    # ---------------------------------------
    #  Figure 6 : Wavelength dependance of Reference point attenuation
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    plt.errorbar(Lambdas_ref+1.0,Attenuation_Ref_std,yerr=Attenuation_Ref_std,ecolor="k",fmt=".")
    plt.errorbar(Lambdas_ref-1.0, Attenuation_Ref_mean, yerr=Attenuation_Ref_err, ecolor="r", fmt=".")
    plt.plot(Lambdas_ref, Attenuation_Ref_mean, "o-b")
    plt.xlabel("$\lambda$ (nm)")
    plt.ylabel("Absorption at z=1")
    plt.title("Absorption reference Point wrt wavelength")
    plt.ylim(10, 40.)
    plt.grid(True, color="r")
    plt.show()

    #---------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    # Relative attenuation compared to reference point
    # ----------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------

    IDXMIN = 0
    IDXMAX = 272
    NBIDX = IDXMAX - IDXMIN + 1

    Attenuation_godown = np.zeros((NBWLBIN, NBIDX))
    NAttenuation_godown = np.zeros((NBWLBIN, NBIDX))
    Attenuation_Err_godown = np.zeros((NBWLBIN, NBIDX))
    Attenuation_mean_GD=np.zeros((NBWLBIN, NBIDX))

    for idx in np.arange(IDXMIN,IDXMAX+1):
        print("---------------------------------------------------------------------------------------")
        print(idx)
        thewl = all_lambdas[idx]
        theabs = all_abs[idx]
        theerrabs = all_errabs[idx]

        # loop on wavelength
        iw0=0
        for w0 in thewl:
            iwlbin = GetWLBin(w0)
            if iwlbin>=0 and theabs[iw0]!=0:
                Attenuation_godown[iwlbin,idx-IDXMIN]+=theabs[iw0]
                NAttenuation_godown[iwlbin, idx-IDXMIN] +=1
                Attenuation_Err_godown[iwlbin, idx - IDXMIN] += theerrabs[iw0]**2
            iw0+=1

    Attenuation_godown=np.where(NAttenuation_godown>=1, Attenuation_godown/NAttenuation_godown,0)
    Attenuation_Err_godown = np.sqrt(np.where(NAttenuation_godown >= 1, Attenuation_Err_godown / NAttenuation_godown, 0))

    print("Attenuation_godown.shape:", Attenuation_godown.shape)
    print("Attenuation_Err_godown.shape:", Attenuation_Err_godown.shape)


    #print("Attenuation_godown:", Attenuation_godown)
    #print("Attenuation_Err_godown:",Attenuation_Err_godown)


    # Express attenuation wrt reference point
    # Reference point array extended for array broadcasting
    Attenuation_mean_GD=Attenuation_godown-Attenuation_Ref_mean[:,np.newaxis]

    #print("Attenuation_mean_GD.shape:",Attenuation_mean_GD.shape)
    #print("Attenuation_mean_GD:", Attenuation_mean_GD)

    #print("airmass_godown.shape:",airmass_godown.shape)
    #print("airmass_godown:", airmass_godown)

    # ---------------------------------------
    #  Figure 7 : Relative attenuation wrt airmass for star rising part
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    #
    for iwlbin in np.arange(NBWLBIN):
        colorVal = scalarMap.to_rgba(iwlbin, alpha=1)

        print(iwlbin," : ",Attenuation_mean_GD[iwlbin,:])

        #plt.errorbar(airmass_godown,Attenuation_mean_GD[iwlbin,:],yerr= Attenuation_Err_godown[iwlbin,:],color=colorVal,fmt="o",label=WLLABELS[iwlbin])
        plt.plot(airmass_godown, Attenuation_mean_GD[iwlbin, :],"o",color=colorVal)

    plt.grid(True, color="r")
    plt.xlabel("airmass")
    plt.ylabel("Attenuation (mag)")
    plt.title("Attenuation relative to reference point")
    plt.ylim(-1., 1.)
    plt.legend()
    plt.show()

    # ---------------------------------------
    #  Figure 8 :  Relative attenuation wrt event number for star rising part
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    #
    for iwlbin in np.arange(NBWLBIN):
        colorVal = scalarMap.to_rgba(iwlbin, alpha=1)

        print(iwlbin, " : ", Attenuation_mean_GD[iwlbin, :])

        #plt.errorbar(event_number_godown, Attenuation_mean_GD[iwlbin, :], yerr=Attenuation_Err_godown[iwlbin, :],color=colorVal, fmt="o",label=WLLABELS[iwlbin])
        plt.plot(event_number_godown, Attenuation_mean_GD[iwlbin, :],"o",color=colorVal)

    plt.grid(True, color="r")
    plt.xlabel("Event Number")
    plt.ylabel("Attenuation (mag)")
    plt.title("Attenuation relative to reference point")
    plt.ylim(-1.,1.)
    plt.legend()
    plt.show()



#-------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # Relative Attenuation for the whole observation range A(t)-A(t0)
    # ----------------------------------------------------------------------------------


    # produce the data
    IDXMIN = all_indexes.min()
    IDXMAX = all_indexes.max()
    NBIDX = IDXMAX - IDXMIN + 1

    Attenuation_all = np.zeros((NBWLBIN, NBIDX))
    NAttenuation_all = np.zeros((NBWLBIN, NBIDX))
    Attenuation_Err_all = np.zeros((NBWLBIN, NBIDX))
    Attenuation_mean_all = np.zeros((NBWLBIN, NBIDX))

    for idx in np.arange(IDXMIN, IDXMAX + 1):
        print("---------------------------------------------------------------------------------------")
        print(idx)
        thewl = all_lambdas[idx]
        theabs = all_abs[idx]
        theerrabs = all_errabs[idx]

        # loop on wavelength
        iw0 = 0
        for w0 in thewl:
            iwlbin = GetWLBin(w0)
            if iwlbin >= 0 and theabs[iw0] != 0:
                Attenuation_all[iwlbin, idx - IDXMIN] += theabs[iw0]
                NAttenuation_all[iwlbin, idx - IDXMIN] += 1
                Attenuation_Err_all[iwlbin, idx - IDXMIN] += theerrabs[iw0] ** 2
            iw0 += 1

    Attenuation_all = np.where(NAttenuation_all >= 1, Attenuation_all / NAttenuation_all, 0)
    # root squared of average squared errors
    Attenuation_Err_all = np.sqrt(np.where(NAttenuation_all >= 1, Attenuation_Err_all / NAttenuation_all, 0))


    # Mask for exactly zero attenuation
    mask = Attenuation_all == 0
    print("mask=", mask)


    # Express attenuation wrt reference point
    Attenuation_mean_ALL = Attenuation_all - Attenuation_Ref_mean[:, np.newaxis]
    Attenuation_Err_ALL= Attenuation_Err_all

    print("Attenuation_mean_ALL.shape:", Attenuation_mean_ALL.shape)
    print("Attenuation_Err_all.shape:", Attenuation_Err_all.shape)

    # masked attenuations
    MAttenuation_mean_ALL = np.ma.masked_array(Attenuation_mean_ALL, mask)
    MAttenuation_Err_ALL  = np.ma.masked_array(Attenuation_Err_ALL,mask)


    # ---------------------------------------
    #  Figure 9 : Masked attenuation without error bars
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    # Loop on wavelength bins
    for iwlbin in np.arange(NBWLBIN):
        colorVal = scalarMap.to_rgba(iwlbin, alpha=1)


        #plt.plot(all_indexes, Attenuation_mean_ALL[iwlbin, :], "o", color=colorVal)
        plt.plot(all_indexes, MAttenuation_mean_ALL[iwlbin, :], "o", color=colorVal)



    #plt.plot([292,292],[-1,1],"k-")
    #plt.plot([303, 303], [-1, 1], "k-")

    plt.grid(True, color="r")
    plt.xlabel("Event Number")
    plt.ylabel("Attenuation (mag)")
    plt.title("Attenuation relative to reference point")
    plt.ylim(-1., 1.)
    plt.legend()
    plt.show()

    # ---------------------------------------
    #  Figure 10 :  Masked attenuation with error bars
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    # Loop on wavelength bins
    for iwlbin in np.arange(NBWLBIN):
        colorVal = scalarMap.to_rgba(iwlbin, alpha=1)



        #plt.errorbar(all_indexes, Attenuation_mean_ALL[iwlbin, :], yerr=Attenuation_Err_ALL[iwlbin, :], ecolor="grey",
                     #color=colorVal,fmt=".")

        plt.errorbar(all_indexes, MAttenuation_mean_ALL[iwlbin, :], yerr=MAttenuation_Err_ALL[iwlbin, :], ecolor="grey",
                     color=colorVal, fmt=".")


    #plt.plot([292, 292], [-1, 1], "k-")
    #plt.plot([303, 303], [-1, 1], "k-")

    plt.grid(True, color="r")
    plt.xlabel("Event Number")
    plt.ylabel("Attenuation (mag)")
    plt.title("Attenuation relative to reference point")
    plt.ylim(-1., 1.)
    plt.legend()
    plt.show()


# ---------------------------------------
    #  Figure 11 :  Masked attenuation with error bars wrt time
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    # Loop on wavelength bins
    for iwlbin in np.arange(NBWLBIN):
        colorVal = scalarMap.to_rgba(iwlbin, alpha=1)



        #plt.errorbar(all_indexes, Attenuation_mean_ALL[iwlbin, :], yerr=Attenuation_Err_ALL[iwlbin, :], ecolor="grey",
                     #color=colorVal,fmt=".")

        plt.errorbar(all_dt, MAttenuation_mean_ALL[iwlbin, :], yerr=MAttenuation_Err_ALL[iwlbin, :], ecolor="grey",
                     color=colorVal, fmt=".")


    #plt.plot([292, 292], [-1, 1], "k-")
    #plt.plot([303, 303], [-1, 1], "k-")

    plt.grid(True, color="r")
    plt.xlabel("Relative time (hours)")
    plt.ylabel("Attenuation (mag)")
    plt.title("Attenuation relative to reference point")
    plt.ylim(-1., 1.)
    plt.legend()
    plt.show()


#----------------------------------------------------------------------------------------------------------------------