#########################################################################################################
# View spectra produced in Spectractor
#########################################################################################################

import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import re



import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import matplotlib.colors as colors
import matplotlib.cm as cmx



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

plt.rcParams["figure.figsize"] = (20,10)



#----------------------------------------------------------------------------
WLMIN = 400.0
WLMAX = 800.0
NBWLBIN = 62
WLBINWIDTH = (WLMAX - WLMIN) / float(NBWLBIN)

WLMINBIN = np.arange(WLMIN, WLMAX, WLBINWIDTH)
WLMAXBIN = np.arange(WLMIN + WLBINWIDTH, WLMAX + WLBINWIDTH, WLBINWIDTH)
#--------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------
def GetImage(NSPEC,theWL,theVal):


    image=np.zeros((NBWLBIN,NSPEC),dtype=float)

    # loop on spectra
    for idx in np.arange(0, NSPEC):

        # for this spectrum
        thewl=theWL[idx]
        theval=theVal[idx]

        wl_sorted_idx = np.argsort(thewl)
        wl =thewl[wl_sorted_idx]
        val = theval[wl_sorted_idx]

        # loop on wavelength bins for this spectrum
        for bin in np.arange(0,NBWLBIN,1):

            val_idx=np.where(np.logical_and(wl>=WLMINBIN[bin],wl<WLMAXBIN[bin]))[0]

            val_sel=val[val_idx]
            val_av=val_sel.mean()
            val_std=val_sel.std()

            image[bin,idx]=val_av
    return image
#--------------------------------------------------------------------------------------


flatten = lambda l: [item for sublist in l for item in sublist]


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
    #output_directory = "/Users/dagoret/DATA/PicDuMidiFev2019/spectractor_output_deco/"+ thedate
    #output_directory = "/Users/dagoret/DATA/PicDuMidiFev2019/spectractor_output_deco2/" + thedate
    # dernière production sur deco
    output_directory = "/Users/dagoret/DATA/PicDuMidiFev2019/spectractor_output_prod3/" + thedate

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
    onlyfilestable = []

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
                onlyfilestable.append(file_name.replace('spectrum.fits','table.csv'))



    # sort again all the files
    onlyfilesspectrum = np.array(onlyfilesspectrum)
    sortedindexes = np.argsort(onlyfilesspectrum)
    onlyfilesspectrum = onlyfilesspectrum[sortedindexes]

    onlyfilestable = np.array(onlyfilestable)
    onlyfilestable = onlyfilestable[sortedindexes]


    #get basnemae of files for later use (to check if _table.csv and _spectrogram.fits exists
    onlyfilesbasename=[]
    for f in onlyfilesspectrum:
        onlyfilesbasename.append(re.findall("(^T.*)_spectrum.fits$",f)[0])


    basenamecut=[]
    for f in onlyfilesspectrum:
        print(f)
        basenamecut.append(f.split("_HD")[0])


    #############################################
    # 3) Plot Spectra
    ##########################################

    NBSPEC = len(sortedindexes)


    all_lambdas=[]
    all_Dx=[]
    all_Dy=[]
    all_Dy_mean=[]
    all_flux_sum=[]
    all_flux_integral=[]
    all_flux_err=[]
    all_fwhm=[]
    all_Dy_fwhm_sup=[]
    all_Dy_fwhm_inf = []
    all_Dx_rot = []
    all_amplitude_moffat = []
    all_x_mean=[]
    all_gamma=[]
    all_alpha=[]
    all_eta_gauss=[]
    all_stddev=[]
    all_saturation=[]


    all_airmass=[]

    for idx in np.arange(0, NBSPEC):

        #print("{}) : {}".format(idx,onlyfilesspectrum[idx]))
        #print("{}) : {}".format(idx, onlyfilestable[idx]))

        fullfilename1 = os.path.join(output_directory, onlyfilesspectrum[idx])
        fullfilename2 = os.path.join(output_directory, onlyfilestable[idx])

        #try:
        if 1:
            #s = Spectrum()
            #s.load_spectrum(fullfilename)
            #am=s.header["AIRMASS"]
            #thetable=s.chromatic_psf.table


            #print(thetable.dtype)
            #('lambdas', '<f8'), ('Dx', '<f8'), ('Dy', '<f8'), ('Dy_mean', '<f8'), ('flux_sum', '<f8'), (
            #'flux_integral', '<f8'), ('flux_err', '<f8'), ('fwhm', '<f8'), ('Dy_fwhm_sup', '<f8'), (
            #'Dy_fwhm_inf', '<f8'), ('Dx_rot', '<f8'), ('amplitude_moffat', '<f8'), ('x_mean', '<f8'), (
            #'gamma', '<f8'), ('alpha', '<f8'), ('eta_gauss', '<f8'), ('stddev', '<f8'), ('saturation', '<f8')]

            hdu = fits.open(fullfilename1)
            header = hdu[0].header
            all_airmass.append(header["AIRMASS"])
            hdu.close()

            thetable=pd.read_csv(fullfilename2)




            lambdas= np.array(thetable["lambdas"])
            Dx = np.array(thetable["Dx"])
            Dy = np.array(thetable["Dy"])
            Dy_mean = np.array(thetable["Dy_mean"])
            flux_sum=np.array(thetable["flux_sum"])
            flux_integral = np.array(thetable["flux_integral"])
            flux_err = np.array(thetable["flux_err"])
            fwhm = np.array(thetable["fwhm"])
            Dy_fwhm_sup = np.array(thetable["Dy_fwhm_sup"])
            Dy_fwhm_inf = np.array(thetable["Dy_fwhm_inf"])
            Dx_rot = np.array(thetable["Dx_rot"])
            amplitude_moffat = np.array(thetable["amplitude_moffat"])
            x_mean = np.array(thetable["x_mean"])
            gamma = np.array(thetable["gamma"])
            alpha = np.array(thetable["alpha"])
            eta_gauss = np.array(thetable["eta_gauss"])
            stddev = np.array(thetable["stddev"])
            saturation = np.array(thetable["saturation"])

            good_indexes=np.where(np.logical_and(lambdas>=WLMIN,lambdas<WLMAX))[0]




            all_lambdas.append(lambdas[good_indexes])
            all_Dx.append(Dx[good_indexes])
            all_Dy.append(Dy[good_indexes])
            all_Dy_mean.append(Dy_mean[good_indexes])
            all_flux_sum.append(flux_sum[good_indexes])
            all_flux_integral.append(flux_integral[good_indexes])
            all_flux_err.append(flux_err[good_indexes])
            all_fwhm.append(fwhm[good_indexes])
            all_Dy_fwhm_sup.append(Dy_fwhm_sup[good_indexes])
            all_Dy_fwhm_inf.append(Dy_fwhm_inf[good_indexes])
            all_Dx_rot.append(Dx_rot[good_indexes])
            all_amplitude_moffat.append(amplitude_moffat[good_indexes])
            all_x_mean.append(x_mean[good_indexes])
            all_gamma.append(gamma[good_indexes])
            all_alpha.append(alpha[good_indexes])
            all_eta_gauss.append(eta_gauss[good_indexes])
            all_stddev.append(stddev[good_indexes])
            all_saturation.append(saturation[good_indexes])


        #except:
        if 0:
            print("Unexpected error:", sys.exc_info()[0])
            pass


    print(len(all_lambdas))
    print(len(all_Dx))


    ifig = 420

    #############################################
    # 3) Process
    ##########################################

    NBSPEC = len(sortedindexes)
    WLMIN = 380.0
    WLMAX = 1000.0
    NBWLBIN = 200
    WLBINWIDTH = (WLMAX - WLMIN) / float(NBWLBIN)

    WLMINBIN = np.arange(WLMIN, WLMAX, WLBINWIDTH)
    WLMAXBIN = np.arange(WLMIN + WLBINWIDTH, WLMAX + WLBINWIDTH, WLBINWIDTH)

    print('NBSPEC....................................= ', NBSPEC)
    print('WLMINBIN..................................=', WLMINBIN.shape, WLMINBIN)
    print('WLMAXBIN..................................=', WLMAXBIN.shape, WLMAXBIN)
    print('NBWLBIN...................................=', NBWLBIN)
    print('WLBINWIDTH................................=', WLBINWIDTH)

    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=NBSPEC)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange(NBSPEC), alpha=1)





    # ------------------------------------
    #  Figure 1: fmwh 2D image
    # ------------------------------------
    theimage_fwhm = GetImage(NBSPEC, all_lambdas, all_fwhm)

    fig = plt.figure(num=ifig, figsize=(20, 20))
    ifig += 1

    theextent = [0, NBSPEC, WLMIN, WLMAX]

    img = plt.imshow(theimage_fwhm, origin="lower", cmap="jet", vmin=0,vmax=30.,extent=theextent, aspect='auto')

    plt.grid(True, color="white")
    plt.title("all spectra")
    plt.xlabel(" event number")
    plt.ylabel("$\lambda$ (nm)")
    # plt.axes().set_aspect('equal', 'datalim')

    plt.suptitle("night 2019-02-15, HD116405 Filter None (lin scale)")
    cbar = fig.colorbar(img, orientation="horizontal")
    cbar.set_label('fwhm', rotation=0)

    plt.show()
    #------------------------------------------------------------------------------------

    all_wlmin=[]
    all_wlmax=[]
    all_col=[]
    idx=0
    for wls in all_lambdas:

        if len(wls)>0:
            all_wlmin.append(wls.min())
            all_wlmax.append(wls.max())
            colorVal = scalarMap.to_rgba(idx, alpha=1)
            all_col.append(colorVal)


        idx+=1

    # ------------------------------------
    #  Figure 2: lambdas histo2D
    # ------------------------------------
    fig = plt.figure(num=ifig, figsize=(10, 10))
    ifig += 1

    plt.scatter(all_wlmin,all_wlmax,marker="o",c=all_col)
    plt.axes().set_aspect('auto')
    plt.grid()
    plt.xlabel("$\lambda_{min}$ (nm)")
    plt.ylabel("$\lambda_{max}$ (nm)")
    plt.suptitle("START-STOP of spectra")
    plt.show()



    # ------------------------------------
    #  Figure 1: lambdas histo1D
    # ------------------------------------
    fig = plt.figure(num=ifig, figsize=(14, 6))
    ifig += 1


    plt.subplot(1,2,1)
    plt.hist( all_wlmin)
    plt.xlabel("$\lambda_{min}$ (nm)")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.hist(all_wlmax)
    plt.xlabel("$\lambda_{max}$ (nm)")
    plt.grid()

    plt.suptitle("START-STOP of spectra")
    plt.show()


    # ------------------------------------
    #  Figure 1: fmwh histo
    # ------------------------------------
    fig = plt.figure(num=ifig, figsize=(20, 20))
    ifig += 1

    all_fmin = []
    all_fmax = []
    all_fmean=[]
    all_fstd = []

    for fw in all_fwhm:
        fmean= fw.mean()
        fstd= fw.std()
        fwmin = fw.min()
        fwmax = fw.max()
        all_fmin.append(fwmin)
        all_fmax.append(fwmax)
        all_fmean.append(fmean)
        all_fstd.append(fstd)

    plt.subplot(3,2,1)
    #all_fwhm=np.array(all_fwhm)
    all_fwhm_flat=flatten(all_fwhm)

    plt.hist(all_fwhm_flat,bins=50,range=(0,80.))
    plt.xlabel("fwhm (pix)")
    plt.title("FWHM distribution for all wavelength")
    plt.grid(True,color="k")
    plt.yscale("log")

    plt.subplot(3, 2, 2)
    plt.hist(all_fmean, bins=50, range=(0, 80.))
    plt.xlabel("fwhm mean (pix)")
    plt.title("Mean FWHM distribution for all wavelength")
    plt.grid(True, color="k")
    plt.yscale("log")

    plt.subplot(3, 2, 3)
    plt.hist(all_fstd, bins=50, range=(0, 80.))
    plt.xlabel("fwhm std (pix)")
    plt.title("STD FWHM distribution for all wavelength")
    plt.grid(True, color="k")
    plt.yscale("log")

    plt.subplot(3, 2, 4)
    plt.hist(all_fmin, bins=50, range=(0, 80.))
    plt.xlabel("fwhm min (pix)")
    plt.title("MIN FWHM distribution for all wavelength")
    plt.grid(True, color="k")
    plt.yscale("log")

    plt.subplot(3, 2, 5)
    plt.hist(all_fmax, bins=50, range=(0, 80.))
    plt.xlabel("fwhm max (pix)")
    plt.title("MAX FWHM distribution for all wavelength")
    plt.grid(True, color="k")
    plt.yscale("log")

    plt.tight_layout(pad=5., h_pad=5., w_pad=2., rect=None)
    plt.show()


    # File lit


    for idx in np.arange(0, NBSPEC):

        if(all_fmax[idx]>20):
            print("file={} : fwhm_max={} ".format(basenamecut[idx],all_fmax[idx]))
        #print(basenamecut[idx], all_fmax[idx])