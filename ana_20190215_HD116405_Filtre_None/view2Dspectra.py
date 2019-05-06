#########################################################################################################
# View spectra produced in Spectractor
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

plt.rcParams["figure.figsize"] = (20,12)

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

    NBSPEC = len(sortedindexes)

    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=NBSPEC)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    all_colors = scalarMap.to_rgba(np.arange( NBSPEC), alpha=1)




    plt.figure()


    for idx in np.arange(0, NBSPEC):
        # if idx in [0,1,4]:
        #    continue

        print("{}) : {}".format(idx,onlyfilesspectrum[idx]))

        fullfilename = os.path.join(output_directory, onlyfilesspectrum[idx])
        try:
            hdu = fits.open(fullfilename)

            header=hdu[0].header
            data=hdu[0].data

            lambdas=data[0,:]
            spec = data[1,:]
            err=data[2,: 2]

            colorVal = scalarMap.to_rgba(idx, alpha=1)

            if spec.max() < 0.6e-10 :
                plt.plot(lambdas,spec,"-",color=colorVal)



        except:
            print("Unexpected error:", sys.exc_info()[0])
            pass

    plt.grid(True,color="k")
    plt.title("all spectra")
    plt.xlabel("$\lambda$ (nm)")
    plt.ylabel("spectra")
    plt.ylim(0,0.1e-9)
    plt.show()












