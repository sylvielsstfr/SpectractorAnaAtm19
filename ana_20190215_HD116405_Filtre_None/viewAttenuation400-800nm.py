#########################################################################################################
# View Attenuation
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

import sys
sys.path.append(workbookDir)
sys.path.append(spectractordir)
sys.path.append(os.path.dirname(workbookDir))

from spectractor import parameters
from spectractor.extractor.extractor import Spectractor
from spectractor.logbook import LogBook
from spectractor.extractor.dispersers import *
from spectractor.extractor.spectrum import *


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
                print(">>>>> output filename : {} already exists with size {} ! Skip Spectractor".format(output_filename_psf,filesize))

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
    WLMIN=400.0
    WLMAX=800.0
    NBWLBIN=10
    WLBINWIDTH=(WLMAX-WLMIN)/float(NBWLBIN)

    WLMINBIN=np.arange(WLMIN,WLMAX,WLBINWIDTH)
    WLMAXBIN =np.arange(WLMIN+WLBINWIDTH, WLMAX + WLBINWIDTH, WLBINWIDTH)

    print('NBSPEC....................................= ',NBSPEC)
    print('WLMINBIN..................................=',WLMINBIN.shape, WLMINBIN)
    print('WLMAXBIN..................................=',WLMAXBIN.shape,WLMAXBIN)
    print('NBWLBIN...................................=',NBWLBIN)
    print('WLBINWIDTH................................=', WLBINWIDTH)



    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=NBSPEC)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(NBSPEC), alpha=1)


    theimage=np.zeros((NBWLBIN,NBSPEC),dtype=float)
    print("image.shape=",theimage.shape)
    print("image.type=", theimage.dtype)

    #assert False



    all_airmass=[]
    all_flag=[]

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
            all_airmass.append(header["AIRMASS"])

            data=hdu[0].data

            wavelength=data[0,:]
            spec = data[1,:]
            err=data[2,: ]

            wl_sorted_idx=np.argsort(wavelength)

            wl=wavelength[wl_sorted_idx]
            fl=spec[wl_sorted_idx]


            colorVal = scalarMap.to_rgba(idx, alpha=1)

            # cut bad amblitude spectra
            if spec.max() < 0.6e-10 :

                all_flag.append(True)
                # loop on wavelength bins
                for bin in np.arange(0,NBWLBIN,1):

                    fluxes_idx=np.where(np.logical_and(wl>=WLMINBIN[bin],wl<WLMAXBIN[bin]))[0]

                    if len(fluxes_idx)>0:

                        fluxes=fl[fluxes_idx]
                        fluxes_av=fluxes.mean()

                        theimage[bin,idx]=fluxes_av
            else:
                all_flag.append(False)


        #except:
        if 0:
            print("Unexpected error:", sys.exc_info()[0])
            pass

    all_flag=np.array(all_flag)

    # get a sign for airmass
    all_airmass=np.array(all_airmass)
    zmin_idx=np.where(all_airmass==all_airmass.min())[0][0]
    zmin=all_airmass[zmin_idx]

    all_airmass_sgn=np.where(np.arange(len(all_airmass))<zmin_idx,-all_airmass,all_airmass)

    print("zmin_idx...............=",zmin_idx)
    print("zmin...................=", zmin)

    #godown_idx = np.where(np.arange(len(all_airmass) )<= zmin_idx)[0]
    #goup_idx = np.where(np.arange(len(all_airmass)) >= zmin_idx)[0]

    godown_idx = np.where(np.logical_and(np.arange(len(all_airmass)) <= zmin_idx, all_flag))[0]
    goup_idx = np.where(np.logical_and(np.arange(len(all_airmass)) >= zmin_idx,all_flag))[0]


    print('godown_idx.............=', godown_idx)
    print('goup_idx...............=', goup_idx)

    airmass_godown = all_airmass[godown_idx]
    airmass_goup = all_airmass[goup_idx]

    event_number_godown=np.arange(0, NBSPEC)[godown_idx]
    event_number_goup = np.arange(0, NBSPEC)[goup_idx]


    ifig=0


    #-------------------------------------------------------------------------------------------------------------
    #
    # 2D image of attenuation
    #------------------------------------------------------------------------------------------------------
    plt.figure(num=ifig,figsize=(20, 20))
    ifig+=1

    plt.subplot(2,1,1)

    plt.scatter(np.arange(NBSPEC),all_airmass,color=all_colors)
    plt.grid(True,color="k")
    plt.xlabel("event number")
    plt.ylabel("airmass")
    plt.title("airmasses")
    plt.xlim(0,NBSPEC)

    plt.subplot(2,1,2)

    theextent = [0, NBSPEC, WLMIN, WLMAX]

    img=plt.imshow(theimage,origin="lower",cmap="jet",extent=theextent,aspect='auto')

    #plt.colorbar(img,orientation="horizontal",)

    plt.grid(True,color="white")
    plt.title("all spectra")
    plt.xlabel(" event number")
    plt.ylabel("$\lambda$ (nm)")
    #plt.axes().set_aspect('equal', 'datalim')

    plt.suptitle("night 2019-02-15, HD116405 Filter None")

    plt.show()



    #------------------------------------------------------------------------------------------------------
    # Plot attenuation vs airmass
    #-----------------------------------------------------------------------------------------------------------

    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=NBWLBIN)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(NBWLBIN), alpha=1)

    # ------------------------------------
    #  Figure
    # ------------------------------------
    plt.figure(num=ifig,figsize=(20, 8))
    ifig += 1

    # loop on wavelength bins
    plt.subplot(2, 1, 1)
    for ibinwl in np.arange(0, NBWLBIN, 1):
        amplitudes=theimage[ibinwl,:]   # array having the same dimension of airmass
        amplitudes_godown=amplitudes[godown_idx]
        #airmass_godown = all_airmass[godown_idx]
        plt.semilogy(airmass_godown,amplitudes_godown,"o-",color=all_colors[ibinwl],markersize=5)
        plt.grid(True, color="k")
        plt.ylim(1e-12, 5e-11)
        plt.title("star rising")
        plt.xlabel("airmass")
        plt.ylabel("flux")

    plt.subplot(2, 1, 2)
    for ibinwl in np.arange(0, NBWLBIN, 1):
        amplitudes=theimage[ibinwl,:]   # array having the same dimension of airmass
        amplitudes_goup = amplitudes[goup_idx]
        #airmass_goup = all_airmass[goup_idx]
        plt.semilogy(airmass_goup, amplitudes_goup, "o-", color=all_colors[ibinwl],markersize=5)
        plt.grid(True, color="k")
        plt.ylim(1e-12, 5e-11)
        plt.title("star falling")
        plt.xlabel("airmass")
        plt.ylabel("flux")

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


    #---------------------------------------
    #  Figure
    #------------------------------------
    plt.figure(num=ifig,figsize=(16, 10))
    ifig += 1
    # loop on wavelength bins
    for ibinwl in np.arange(0, NBWLBIN, 1):
        amplitudes = theimage[ibinwl, :]  # array having the same dimension of airmass
        amplitudes_godown = amplitudes[godown_idx]
        #airmass_godown = all_airmass[godown_idx]
        label="{:3.0f}-{:3.0f}nm".format(WLMINBIN[ibinwl],WLMAXBIN[ibinwl])
        plt.semilogy(airmass_godown, amplitudes_godown, "o-", color=all_colors[ibinwl], markersize=5,label=label)
        plt.grid(True, color="k")
        plt.ylim(1e-12, 5e-11)
        plt.xlim(1,1.7)
        plt.title("star rising")
        plt.xlabel("airmass")
        plt.ylabel("flux")
        plt.legend()
    plt.show()

    # --------------------------------------
    #  Figure
    # ------------------------------------
    plt.figure(num=ifig,figsize=(16, 10))
    ifig += 1
    for ibinwl in np.arange(0, NBWLBIN, 1):
        amplitudes=theimage[ibinwl,:]   # array having the same dimension of airmass
        amplitudes_goup = amplitudes[goup_idx]
        #airmass_goup = all_airmass[goup_idx]
        label = "{:3.0f}-{:3.0f}nm".format(WLMINBIN[ibinwl], WLMAXBIN[ibinwl])
        plt.semilogy(airmass_goup, amplitudes_goup, "o-", color=all_colors[ibinwl],markersize=5,label=label)
        plt.grid(True, color="k")
        plt.ylim(1e-12, 5e-11)

        plt.xlim(1,1.06)
        plt.title("star falling")
        plt.xlabel("airmass")
        plt.ylabel("flux")
        plt.legend()
    plt.show()


    #-----------------------------------------------------------------------------------------------------------------
    #
    # Attenuation in time
    #----------------------------------------------------------------------------------------------------------------

    # ---------------------------------------
    #  Figure
    # ------------------------------------
    plt.figure(num=ifig,figsize=(16, 10))
    ifig += 1

    NBAM=len(airmass_godown)

    ampl_ratio_godown = np.zeros((NBWLBIN, NBAM), dtype=float)

    # loop on wavelength bins
    for ibinwl in np.arange(0, NBWLBIN, 1):
        amplitudes = theimage[ibinwl, :]  # array having the same dimension of airmass
        amplitudes_godown = amplitudes[godown_idx]
        amplitudes_shifted=np.roll(amplitudes_godown,1)
        amplitudes_shifted[0]=amplitudes_godown[0]
        amplitudes_ratio=np.where(amplitudes_shifted>0,amplitudes_godown/amplitudes_shifted,0)
        idx_sel=np.where(amplitudes_shifted>0)[0]
        ampl_ratio_godown[ibinwl,:]=amplitudes_ratio
        #airmass_godown = all_airmass[godown_idx]

        label = "{:3.0f}-{:3.0f}nm".format(WLMINBIN[ibinwl], WLMAXBIN[ibinwl])
        plt.plot(airmass_godown[idx_sel], amplitudes_ratio[idx_sel], "o-", color=all_colors[ibinwl], markersize=5, label=label)
        plt.grid(True,color="k")
        plt.ylim(0.5, 2.)
        plt.xlim(1, 1.7)
        plt.title("star rising")
        plt.xlabel("airmass")
        plt.ylabel("flux ratio")
        plt.legend()
    plt.show()

    # --------------------------------------
    #  Figure
    # ------------------------------------
    plt.figure(num=ifig,figsize=(16, 10))
    ifig += 1

    NBAM = len(airmass_goup)
    ampl_ratio_goup = np.zeros((NBWLBIN, NBAM), dtype=float)

    for ibinwl in np.arange(0, NBWLBIN, 1):
        amplitudes = theimage[ibinwl, :]  # array having the same dimension of airmass
        amplitudes_goup = amplitudes[goup_idx]

        amplitudes_shifted = np.roll(amplitudes_goup, 1)
        amplitudes_shifted[0] = amplitudes_goup[0]
        amplitudes_ratio =  np.where(amplitudes_shifted>0,amplitudes_goup/amplitudes_shifted,0)

        ampl_ratio_goup[ibinwl, :] = amplitudes_ratio

        idx_sel = np.where(amplitudes_shifted > 0)[0]

        #airmass_goup = all_airmass[goup_idx]
        label = "{:3.0f}-{:3.0f}nm".format(WLMINBIN[ibinwl], WLMAXBIN[ibinwl])
        plt.plot(airmass_goup[idx_sel], amplitudes_ratio[idx_sel], "o-", color=all_colors[ibinwl], markersize=5, label=label)
        plt.grid(True, color="k")
        plt.ylim(0.2, 2.)

        plt.xlim(1, 1.06)
        plt.title("star falling")
        plt.xlabel("airmass")
        plt.ylabel("flux ratio")
        plt.legend()
    plt.show()


    #--------------------------------------------------------------------------------------------------------------------------
    #
    #  amplitude ratio in 2D image
    #--------------------------------------------------------------------------------------------------------------------------
    plt.figure(num=ifig,figsize=(16, 16))
    ifig += 1

    plt.subplot(2, 1, 1)
    theextent=[event_number_godown.min(),event_number_godown.max(),WLMIN,WLMAX]
    plt.imshow(ampl_ratio_godown,cmap="jet",origin="lower",vmin=0,vmax=1.5,aspect='auto',extent=theextent)
    plt.grid(True,color="white")
    plt.xlabel("Event number")
    plt.ylabel("$\lambda$ (nm)")
    plt.title("flux ratio for rising star")
    plt.subplot(2, 1, 2)
    theextent = [event_number_goup.min(), event_number_goup.max(), WLMIN, WLMAX]
    plt.imshow(ampl_ratio_goup, cmap="jet", origin="lower",vmin=0.,vmax=1.5,aspect='auto',extent=theextent)
    plt.grid(True,color="white")
    plt.xlabel("Event number")
    plt.ylabel("$\lambda$ (nm)")
    plt.title("flux ratio for falling star")
    #plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)
    plt.suptitle("Flux ratio in wavelength band (relative to previous obs)")

    plt.show()

    # --------------------------------------------------------------------------------------------------------------------------
    #
    #  amplitude ratio in scatter plot vs event number
    # --------------------------------------------------------------------------------------------------------------------------
    plt.figure(num=ifig,figsize=(16, 8))
    ifig += 1

    plt.subplot(2, 1, 1)

    idx_sel=np.where(ampl_ratio_godown>0)[0]

    ratio_mean=np.mean(ampl_ratio_godown,axis=0)
    ratio_std = np.std(ampl_ratio_godown, axis=0)
    plt.errorbar(event_number_godown,ratio_mean,yerr=ratio_std,fmt="o",color="r",ecolor='k')
    plt.ylim(0.75,1.25)
    plt.grid(True,color="k")
    plt.xlabel("Event number")
    plt.ylabel("Flux ratio")
    plt.title("flux ratio for rising star")
    plt.subplot(2, 1, 2)

    ratio_mean = np.mean(ampl_ratio_goup, axis=0)
    ratio_std = np.std(ampl_ratio_goup, axis=0)
    plt.errorbar(event_number_goup, ratio_mean, yerr=ratio_std, fmt="o",color="r",ecolor='k')
    plt.ylim(0.75, 1.25)
    plt.grid(True,color="k")
    plt.xlabel("Event number")
    plt.ylabel("Flux ratio")
    plt.title("flux ratio for falling star")
    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)

    plt.show()


    # --------------------------------------------------------------------------------------------------------------------------
    #
    #  amplitude ratio in scatter plot vs airmass
    # --------------------------------------------------------------------------------------------------------------------------
    plt.figure(num=ifig,figsize=(16, 8))
    ifig += 1

    plt.subplot(2, 1, 1)

    ratio_mean=np.mean(ampl_ratio_godown,axis=0)
    ratio_std = np.std(ampl_ratio_godown, axis=0)
    plt.errorbar( airmass_godown,ratio_mean,yerr=ratio_std,fmt="o",color="r",ecolor='k')
    plt.ylim(0.75,1.25)
    plt.grid(True,color="k")
    plt.xlabel("airmass")
    plt.ylabel("Flux ratio")
    plt.title("flux ratio for rising star")
    plt.subplot(2, 1, 2)

    ratio_mean = np.mean(ampl_ratio_goup, axis=0)
    ratio_std = np.std(ampl_ratio_goup, axis=0)
    plt.errorbar( airmass_goup, ratio_mean, yerr=ratio_std, fmt="o",color="r",ecolor='k')
    plt.ylim(0.75, 1.25)
    plt.grid(True,color="k")
    plt.xlabel("airmass")
    plt.ylabel("Flux ratio")
    plt.title("flux ratio for falling star")
    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)

    plt.show()

    # --------------------------------------------------------------------------------------------------------------------------
    # CUT WL above 800 nm
    #  amplitude ratio in scatter plot vs event number
    # --------------------------------------------------------------------------------------------------------------------------
    plt.figure(num=ifig,figsize=(16, 8))
    ifig += 1

    plt.subplot(2, 1, 1)

    idx_sel_wl = np.where(WLMAXBIN < 800)[0]

    ratio_mean = np.mean(ampl_ratio_godown[idx_sel_wl,:], axis=0)
    ratio_std = np.std(ampl_ratio_godown[idx_sel_wl,:], axis=0)
    plt.errorbar(event_number_godown, ratio_mean, yerr=ratio_std, fmt="o", color="r", ecolor='k')
    plt.ylim(0.75, 1.25)
    plt.grid(True, color="k")
    plt.xlabel("Event number")
    plt.ylabel("Flux ratio")
    plt.title("flux ratio for rising star $\lambda < 800 nm$")
    plt.subplot(2, 1, 2)

    ratio_mean = np.mean(ampl_ratio_goup[idx_sel_wl,:], axis=0)
    ratio_std = np.std(ampl_ratio_goup[idx_sel_wl,:], axis=0)
    plt.errorbar(event_number_goup, ratio_mean, yerr=ratio_std, fmt="o", color="r", ecolor='k')
    plt.ylim(0.75, 1.25)
    plt.grid(True, color="k")
    plt.xlabel("Event number")
    plt.ylabel("Flux ratio")
    plt.title("flux ratio for falling star $\lambda < 800 nm$")
    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)

    plt.show()

    # --------------------------------------------------------------------------------------------------------------------------
    # UT WL above 800 nm
    #  amplitude ratio in scatter plot vs airmass
    # --------------------------------------------------------------------------------------------------------------------------
    plt.figure(num=ifig,figsize=(16, 8))
    ifig += 1

    plt.subplot(2, 1, 1)

    idx_sel_wl = np.where(WLMAXBIN < 800)[0]

    ratio_mean1 = np.mean(ampl_ratio_godown[idx_sel_wl,:], axis=0)
    ratio_std1 = np.std(ampl_ratio_godown[idx_sel_wl,:], axis=0)
    plt.errorbar(airmass_godown, ratio_mean1, yerr=ratio_std1, fmt="o", color="r", ecolor='k')
    plt.ylim(0.8, 1.2)
    plt.grid(True, color="k")
    plt.xlabel("airmass")
    plt.ylabel("Flux ratio")
    plt.title("flux ratio for rising star $\lambda < 800 nm$")
    plt.subplot(2, 1, 2)

    ratio_mean2 = np.mean(ampl_ratio_goup[idx_sel_wl,:], axis=0)
    ratio_std2 = np.std(ampl_ratio_goup[idx_sel_wl,:], axis=0)
    plt.errorbar(airmass_goup, ratio_mean2, yerr=ratio_std2, fmt="o", color="r", ecolor='k')
    plt.ylim(0.8, 1.2)
    plt.grid(True, color="k")
    plt.xlabel("airmass")
    plt.ylabel("Flux ratio")
    plt.title("flux ratio for falling star $\lambda < 800 nm$")
    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)

    plt.show()

    # --------------------------------------------------------------------------------------------------------------------------
    # UT WL above 800 nm
    #  histograms
    # --------------------------------------------------------------------------------------------------------------------------

    plt.figure(num=ifig,figsize=(16, 8))
    ifig += 1
    plt.subplot(2, 3, 1)
    plt.hist(ratio_mean1,bins=50,range=(0.75,1.25),color="b")
    plt.grid(True,color="k")
    plt.xlabel("mean")

    plt.subplot(2, 3, 2)
    plt.hist(ratio_std1,bins=50,range=(0,0.2),color="b")
    plt.grid(True, color="k")
    plt.xlabel("$\sigma$")

    plt.subplot(2, 3, 3)
    plt.scatter(ratio_mean1,ratio_std1,color="b")
    plt.grid(True, color="k")
    plt.ylabel("$\sigma$")
    plt.xlabel("mean")
    plt.xlim(0.75,1.25)
    plt.ylim(0., 0.2)

    plt.subplot(2, 3, 4)
    plt.hist(ratio_mean2,bins=50,range=(0.75,1.25),color="b")
    plt.grid(True, color="k")
    plt.xlabel("mean")

    plt.subplot(2, 3, 5)
    plt.hist(ratio_std2,bins=50,range=(0,0.2),color="b")
    plt.grid(True, color="k")
    plt.xlabel("$\sigma$")

    plt.subplot(2, 3, 6)
    plt.scatter(ratio_mean2, ratio_std2,color="b")
    plt.grid(True, color="k")
    plt.ylabel("$\sigma$")
    plt.xlabel("mean")
    plt.xlim(0.75, 1.25)
    plt.ylim(0., 0.2)

    plt.suptitle("Flux ratio")
    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)
    plt.show()

