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
        # if idx in [0,1,4]:
        #    continue

        print("{}) : {}".format(idx,onlyfilesspectrum[idx]))
        print("{}) : {}".format(idx, onlyfilestable[idx]))

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

            all_lambdas.append(lambdas)
            all_Dx.append(Dx)
            all_Dy.append(Dy)
            all_Dy_mean.append(Dy_mean)
            all_flux_sum.append(flux_sum)
            all_flux_integral.append(flux_integral)
            all_flux_err.append(flux_err)
            all_fwhm.append(fwhm)
            all_Dy_fwhm_sup.append(Dy_fwhm_sup)
            all_Dy_fwhm_inf.append(Dy_fwhm_inf)
            all_Dx_rot.append(Dx_rot)
            all_amplitude_moffat.append(amplitude_moffat)
            all_x_mean.append(x_mean)
            all_gamma.append(gamma)
            all_alpha.append(alpha)
            all_eta_gauss.append(eta_gauss)
            all_stddev.append(stddev)
            all_saturation.append(saturation)


        #except:
        if 0:
            print("Unexpected error:", sys.exc_info()[0])
            pass


        #figfilename="figures/20190215/fig_spec_"+basenamecut[idx]+".png"
        #fig.savefig(figfilename)













