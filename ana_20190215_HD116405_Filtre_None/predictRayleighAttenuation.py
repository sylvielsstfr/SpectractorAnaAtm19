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

    ifig = 700


    #########################################
    # Simulation
    ##########################################


    sim_airmass=np.linspace(1,1.5,10.)
    sim_cos=1./sim_airmass
    sim_wavelength=np.arange(300,1000.,10.)

    NBWLBINSIM=len(sim_wavelength)
    NBAMSIM=len(sim_airmass)





    # ---------------------------------------
    #  Figure Yayleigh attenuation vs airmass goup
    # ------------------------------------

    # bins in wevelength
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=NBWLBINSIM)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(NBWLBINSIM), alpha=1)



    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1


    plt.subplot(2,1,1)
    idx=0
    for wl in sim_wavelength:
        od_adiab = RayOptDepth_adiabatic(wl, altitude=2890.5, costh=sim_cos)  # optical depth
        absmag = 2.5 / np.log(10.) * od_adiab
        transm = np.exp(-od_adiab)
        colorVal = scalarMap.to_rgba(idx, alpha=1)
        plt.plot(sim_airmass,transm,"o-",color=colorVal)
        idx+=1
    plt.ylim(0., 1.)
    plt.xlabel("airmass")
    plt.ylabel("transmission")
    plt.grid()

    plt.subplot(2,1,2)
    idx = 0
    for wl in sim_wavelength:
        od_adiab = RayOptDepth_adiabatic(wl, altitude=2890.5, costh=sim_cos)  # optical depth
        absmag = 2.5 / np.log(10.) * od_adiab
        transm = np.exp(-od_adiab)
        colorVal = scalarMap.to_rgba(idx, alpha=1)
        plt.plot(sim_airmass, absmag, "o-", color=colorVal)
        idx += 1
    plt.grid()
    plt.xlabel("airmass")
    plt.ylabel("magnitude (mag)")
    plt.show()

    # ---------------------------------------
    #  Figure Yayleigh attenuation vs wavelength
    # ------------------------------------

    # bins in wevelength
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=NBAMSIM)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(NBAMSIM), alpha=1)

    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1

    plt.subplot(2, 1, 1)
    idx = 0
    for am in sim_airmass:
        od_adiab = RayOptDepth_adiabatic(sim_wavelength, altitude=2890.5, costh=1/am)  # optical depth
        absmag = 2.5 / np.log(10.) * od_adiab
        transm = np.exp(-od_adiab)
        colorVal = scalarMap.to_rgba(idx, alpha=1)
        plt.plot(sim_wavelength, transm, "o-", color=colorVal)
        idx += 1
    plt.ylim(0.,1.)
    plt.xlabel("wavelength (nm)")
    plt.ylabel("transmission")
    plt.grid()
    plt.subplot(2, 1, 2)
    idx = 0
    for am in sim_airmass:
        od_adiab = RayOptDepth_adiabatic(sim_wavelength, altitude=2890.5, costh=1 / am)  # optical depth
        absmag = 2.5 / np.log(10.) * od_adiab
        transm = np.exp(-od_adiab)
        colorVal = scalarMap.to_rgba(idx, alpha=1)
        plt.plot(sim_wavelength, absmag, "o-", color=colorVal)
        idx += 1
    plt.grid()
    plt.xlabel("wavelength (nm)")
    plt.ylabel("magnitude (mag)")

    plt.show()





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
    all_rayleigh=[]

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


            # Sort Wavemength
            wl_sorted_idx=np.argsort(wavelength)

            wl=wavelength[wl_sorted_idx]
            fl=spec[wl_sorted_idx]
            errfl=err[wl_sorted_idx]
            wlbins=[ GetWLBin(w) for w in wl ]
            wlbins=np.array(wlbins)

            # keep points with measured flux and remove wavelength out of binning
            goodpoints=np.where(np.logical_and(fl != 0, wlbins != -1))

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
                all_rayleigh.append(2.5/np.log(10.)*od_adiab)

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

    all_rayleigh=np.array(all_rayleigh)

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










    # bins in wevelength
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=NBWLBIN)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(NBWLBIN), alpha=1)


    #check if wavelengths are correctly set in wavelength bins
    # ---------------------------------------
    #  Figure Yayleigh attenuation vs airmass goup
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    # loop on wavelength bins

    for idx in all_indexes:
        thewl = all_lambdas[idx]

        wlcolors = []

        for w0 in thewl:
            ibin = GetWLBin(w0)

            if ibin >= 0:
                colorVal = scalarMap.to_rgba(ibin, alpha=1)
            else:
                colorVal = scalarMap.to_rgba(0, alpha=1)

            wlcolors.append(colorVal)

        plt.scatter(np.ones(len(thewl)) * idx, thewl, marker="o", c=wlcolors)
        # plt.errorbar(np.ones(len(themag)) * am, themag, yerr=theerrmag, ecolor="k", fmt=".")

        # plt.ylim(20, 60.)
    plt.grid(True, color="r")
    plt.title("Wavelength bins")
    plt.xlabel("Event number")
    plt.ylabel("wavelength (nm)")
    plt.show()






    # ---------------------------------------
    #  Figure Yayleigh attenuation vs airmass goup
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    # loop on wavelength bins

    # loop on events
    for idx in goup_idx:
        thewl = all_lambdas[idx]
        themag = all_mag[idx]
        theerrmag = all_errmag[idx]
        theray=all_rayleigh[idx]

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

        plt.scatter(np.ones(len(theray)) * am, theray, marker="o", c=wlcolors)
        #plt.errorbar(np.ones(len(themag)) * am, themag, yerr=theerrmag, ecolor="k", fmt=".")

    #plt.ylim(20, 60.)
    plt.grid(True, color="r")
    plt.title("Rayleigh attenuation vs airmass (star falling)")
    plt.xlabel("airmass")
    plt.ylabel("magnitude (mag)")
    plt.show()

    # ---------------------------------------
    #  Figure Yayleigh attenuation vs airmass goup
    # ------------------------------------
    plt.figure(num=ifig, figsize=(16, 10))
    ifig += 1
    # loop on wavelength bins

    # loop on events
    for idx in godown_idx:
        thewl = all_lambdas[idx]
        themag = all_mag[idx]
        theerrmag = all_errmag[idx]
        theray = all_rayleigh[idx]

        wlcolors = []
        am = all_airmass[idx]

        # print(idx)
        # print(themag)
        # print(theerrmag)

        for w0 in thewl:
            ibin = GetWLBin(w0)

            if ibin >= 0:
                colorVal = scalarMap.to_rgba(ibin, alpha=1)
            else:
                colorVal = scalarMap.to_rgba(0, alpha=1)

            wlcolors.append(colorVal)

        plt.scatter(np.ones(len(theray)) * am, theray, marker="o", c=wlcolors)
        # plt.errorbar(np.ones(len(themag)) * am, themag, yerr=theerrmag, ecolor="k", fmt=".")

    #plt.ylim(20, 60.)
    plt.grid(True, color="r")
    plt.title("Rayleigh attenuation vs airmass (star rising)")
    plt.xlabel("airmass")
    plt.ylabel("magnitude (mag)")
    plt.show()
