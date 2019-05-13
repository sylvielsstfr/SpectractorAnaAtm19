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

WLMIN = 300.0
WLMAX = 1100.0
NBWLBIN = 80
WLBINWIDTH = (WLMAX - WLMIN) / float(NBWLBIN)

WLMINBIN = np.arange(WLMIN, WLMAX, WLBINWIDTH)
WLMAXBIN = np.arange(WLMIN + WLBINWIDTH, WLMAX + WLBINWIDTH, WLBINWIDTH)
WLMEANBIN=(WLMINBIN + WLMAXBIN)/2.



print('WLMINBIN..................................=', WLMINBIN.shape, WLMINBIN)
print('WLMAXBIN..................................=', WLMAXBIN.shape, WLMAXBIN)
print('NBWLBIN...................................=', NBWLBIN)
print('WLBINWIDTH................................=', WLBINWIDTH)


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

    # Extract information from files
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

            data=hdu[0].data

            wavelength=data[0,:]
            spec = data[1,:]
            err=data[2,: ]



            wl_sorted_idx=np.argsort(wavelength)

            wl=wavelength[wl_sorted_idx]
            fl=spec[wl_sorted_idx]
            errfl=err[wl_sorted_idx]

            goodpoints=np.where(np.logical_and(fl != 0, errfl != 0))

            wl=wl[goodpoints]
            fl=fl[goodpoints]
            errfl=errfl[goodpoints]



            mag = -2.5 * np.log10(fl)
            errmag = errfl /fl
            od_adiab = RayOptDepth_adiabatic(wl,altitude=2890.5, costh=1./am)  # optical depth
            abs=mag-2.5/np.log(10.)*od_adiab
            errabs=errmag

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

        #except:
        if 0:
            print("Unexpected error:", sys.exc_info()[0])
            pass

    all_indexes=np.array(all_indexes)
    all_airmass = np.array(all_airmass)

    all_flux = np.array(all_flux)
    all_flux_err = np.array(all_flux_err)
    all_lambdas = np.array(all_lambdas)
    all_mag=np.array(all_mag)
    all_errmag=np.array(all_errmag)

    print(all_airmass)

    # get a sign for airmass

    zmin_idx=np.where(all_airmass==all_airmass.min())[0][0]
    zmin=all_airmass[zmin_idx]

    all_airmass_sgn=np.where(np.arange(len(all_airmass))<zmin_idx,-all_airmass,all_airmass)

    print("zmin_idx...............=",zmin_idx)
    print("zmin...................=", zmin)

    #godown_idx = np.where(np.arange(len(all_airmass) )<= zmin_idx)[0]
    #goup_idx = np.where(np.arange(len(all_airmass)) >= zmin_idx)[0]

    godown_idx = np.where(np.arange(len(all_airmass)) <= zmin_idx)[0]
    goup_idx = np.where(np.arange(len(all_airmass)) >= zmin_idx)[0]


    print('godown_idx.............=', godown_idx)
    print('goup_idx...............=', goup_idx)

    airmass_godown = all_airmass[godown_idx]
    airmass_goup = all_airmass[goup_idx]

    event_number_godown=all_indexes[godown_idx]
    event_number_goup = all_indexes[goup_idx]

    print('event num godown_idx.............=', event_number_godown)
    print('event num goup_idx...............=', event_number_goup)



    ifig=600



    # Search for the reference point

    # ---------------------------------------
    #  Figure attenuation vs airmass goup
    # ------------------------------------


    # bins in wevelength
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=NBWLBIN)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(NBWLBIN), alpha=1)

    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    # loop on wavelength bins

    # loop on events
    for idx in goup_idx:
        thewl = all_lambdas[idx]
        themag = all_mag[idx]
        theerrmag = all_errmag[idx]
        wlcolors = []
        am = all_airmass[idx]

        #print(idx)
        #print(themag)
        #print(theerrmag)

        for w0 in thewl:
            ibin = GetWLBin(w0)

            if ibin >= 0:
                colorVal = scalarMap.to_rgba(ibin, alpha=1)
            else:
                colorVal = scalarMap.to_rgba(0, alpha=1)

            wlcolors.append(colorVal)

        plt.scatter(np.ones(len(themag)) * am, themag, marker="o", c=wlcolors)
        #plt.errorbar(np.ones(len(themag)) * am, themag, yerr=theerrmag, ecolor="k", fmt=".")

    plt.ylim(20, 60.)
    plt.grid(True, color="r")
    plt.title("Instrumental Magnitude vs airmass (star falling)")
    plt.xlabel("airmass")
    plt.ylabel("magnitude (mag)")
    plt.show()

    # ---------------------------------------
    #  Figure
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


    plt.ylim(20,60.)
    plt.grid(True,color="r")

    plt.title("Grey Abs = $M(\lambda)-K(\lambda).Z$ vs airmass (star falling)")
    plt.xlabel("airmass")
    plt.ylabel("absorption $m-k(\lambda).z$ (mag)")

    plt.show()

    # ---------------------------------------
    #  Figure
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

    plt.plot([292,292],[0,60],"k-")
    plt.plot([303, 303], [0, 60], "k-")

    plt.ylim(20,60.)
    plt.grid(True,color="r")

    plt.title("Grey Abs = $M(\lambda)-K(\lambda).Z$ vs airmass (star falling)")
    plt.xlabel("Event number")
    plt.ylabel("absorption $m-k(\lambda).z$ (mag)")

    plt.show()


    #------------------------------------------------------------------------------------
    # Wavelength dependence of reference magnitude
    #----------------------------------------------------------------------------------
    # ---------------------------------------
    #  Figure
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    # For Reference
    IDXMINREF=292
    IDXMAXREF=303
    NBIDXREF=IDXMAXREF-IDXMINREF+1


    Attenuation_Ref=np.zeros((NBWLBIN,NBIDXREF))
    NAttenuation_Ref = np.zeros((NBWLBIN, NBIDXREF))
    Attenuation_Ref_Err = np.zeros((NBWLBIN, NBIDXREF))


    # dooble loop on reference idx, wlbin
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
                Attenuation_Ref[iwlbin,idx-IDXMINREF]+=theabs[iw0]
                NAttenuation_Ref[iwlbin, idx-IDXMINREF] +=1
                Attenuation_Ref_Err[iwlbin, idx - IDXMINREF] += theerrabs[iw0]**2
            iw0+=1

    Attenuation_Ref=np.where(NAttenuation_Ref>1, Attenuation_Ref/NAttenuation_Ref,0)
    Attenuation_Ref_Err = np.where(NAttenuation_Ref > 1, Attenuation_Ref_Err / NAttenuation_Ref, 0)

    Attenuation_Ref_mean=np.average(Attenuation_Ref,axis=1)
    Attenuation_Ref_std = np.std(Attenuation_Ref, axis=1)
    Attenuation_Ref_err = np.sqrt(np.average(Attenuation_Ref_Err, axis=1))

    Lambdas_ref=WLMEANBIN

    print("Attenuation_Ref_mean",Attenuation_Ref_mean)
    print("Attenuation_Ref_std", Attenuation_Ref_std)
    print("Attenuation_Ref_err", Attenuation_Ref_err)

    plt.errorbar(Lambdas_ref+1.0,Attenuation_Ref_mean,yerr=Attenuation_Ref_std,ecolor="k",fmt=".")
    plt.errorbar(Lambdas_ref-1.0, Attenuation_Ref_mean, yerr=Attenuation_Ref_err, ecolor="r", fmt=".")
    plt.plot(Lambdas_ref, Attenuation_Ref_mean, "o-b")
    plt.xlabel("$\lambda$ (nm)")
    plt.ylabel("Absorption at z=1")
    plt.title("Absorption reference Point wrt wavelength")
    plt.ylim(10, 40.)
    plt.grid(True, color="r")
    plt.show()

    # ------------------------------------------------------------------------------------
    # Wavelength dependence of reference magnitude
    # ----------------------------------------------------------------------------------
    # ---------------------------------------
    #  Figure
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    #
    IDXMIN = 1
    IDXMAX = 272
    NBIDX = IDXMAX - IDXMIN + 1

    Attenuation_godown = np.zeros((NBWLBIN, NBIDX))
    NAttenuation_godown = np.zeros((NBWLBIN, NBIDX))
    Attenuation_Err_godown = np.zeros((NBWLBIN, NBIDX))

    for idx in np.arange(IDXMIN,IDXMAX):
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

    Attenuation_godown=np.where(NAttenuation_godown>1, Attenuation_godown/NAttenuation_godown,0)
    Attenuation_Err_godown = np.sqrt(np.where(NAttenuation_godown > 1, Attenuation_Err_godown / NAttenuation_godown, 0))


    Attenuation_mean_GD=Attenuation_godown-Attenuation_Ref_mean

    for iwlbin in np.arange(NBWLBIN):
        colorVal = scalarMap.to_rgba(iwlbin, alpha=1)
        plt.errorbar(airmass_godown,Attenuation_mean_GD[iwlbin,:],yerr= Attenuation_Err_godown[iwlbin,:],color=colorVal,fmt="o")

    plt.grid(True, color="r")
    plt.show()
    