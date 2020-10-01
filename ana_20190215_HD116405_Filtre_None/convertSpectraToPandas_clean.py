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

from astropy.table import Table, Row


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
def test_if_good_file(filename, listrootfilename):
    """
    """

    flag_good = False

    for goodfile in listrootfilename:
        stringsearch = f"^{goodfile}.*"
        if re.search(stringsearch, filename):
            flag_good = True
            break

    return flag_good
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




tablesdir="tables_clean"
tablesdir2="tables_bad"


ensure_dir(tablesdir)
ensure_dir(tablesdir2)
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
    #output_directory = "/Users/dagoret/DATA/PicDuMidiFev2019/spectractor_output_deco/"+ thedate
    #output_directory = "/Users/dagoret/DATA/PicDuMidiFev2019/spectractor_output_deco2/" + thedate
    # dernière production sur deco
    output_directory = "/Users/dagoret/DATA/PicDuMidiFev2019/spectractor_output_prod4/" + thedate

    parameters.VERBOSE = True
    parameters.DISPLAY = True




    ############################
    # 2) Get the config
    #########################

    config = spectractordir+"/config/picdumidi.ini"

    my_logger = set_logger(__name__)
    my_logger.info('\n\tconvertSpectraToPandas')
    # Load config file
    load_config(config)

    ############################
    # 3) Read Selection file
    #########################


    # quality_flag_file="images-pic-15fev19-cut.xlsx"
    input_selected_files = "selected_images-pic-15fev19-cut.xlsx"

    dfgood = pd.read_excel(input_selected_files)

    list_of_goodfiles = dfgood["file"]
    list_of_rootfilename = [file.split("_bin")[0] for file in list_of_goodfiles]





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
    onlyfilesspectrum_qualityflag = []



    for file_name in onlyfiles:
        if re.search("^T.*_spectrum.fits$", file_name):
            # check if other files exits

            quality_flag = False

            if test_if_good_file(file_name, list_of_rootfilename):
                print("*************************************************************************************")
                print("*                ************  GOOD FILE = ", file_name)
                print("*************************************************************************************")
                quality_flag = True
            else:
                print("*************************************************************************************")
                print("*                >>>>>>>>>>>>  BAD FILE = ", file_name)
                print("*************************************************************************************")

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

                onlyfilesspectrum_qualityflag.append(quality_flag)




    # sort again all the files
    onlyfilesspectrum = np.array(onlyfilesspectrum)
    onlyfilesspectrum_qualityflag = np.array(onlyfilesspectrum_qualityflag)


    sortedindexes = np.argsort(onlyfilesspectrum)
    onlyfilesspectrum = onlyfilesspectrum[sortedindexes]
    onlyfilesspectrum_qualityflag = onlyfilesspectrum_qualityflag[sortedindexes]  # flag


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
    all_filename=[]

    #----------------------------------
    # Extract spectra information from files
    # compute magnitudes
    #-----------------------------
    count=-1
    # loop on spectra
    for idx in np.arange(0, NBSPEC):
        if idx==322:
            print("SKIP bad file",onlyfilesspectrum[idx] )
            continue

        count += 1


        strnum = f'{count:03}'
        output_pdfile = "attenuationdata_clean{}.csv".format(strnum)





        print(" read file {}) : {}".format(idx,onlyfilesspectrum[idx]))

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
            DT=(T-T0).sec/(3600.0)  # time in hours

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

            absmin = abs.min()
            absmax = abs.max()


            df=pd.DataFrame()

            nwl=len(wl)
            df["count"]=np.full(nwl,count)
            df["airmass"] = np.full(nwl,am)
            df["time"] = np.full(nwl,DT)
            df["wavelength"] = wl
            df["flux"] = fl
            df["errflux"] = errfl
            df["mag"] = mag
            df["errmag"] = errmag
            df["abs"] = abs
            df["errabs"] = errabs


            #df.to_csv(os.path.join(tablesdir,output_pdfile))


            if onlyfilesspectrum_qualityflag[idx]:
                df.to_csv(os.path.join(tablesdir, output_pdfile))
            else:
                print("*************************************************************************************")
                print("*    >>>> Reject bad quality file {}) : {}".format(idx, onlyfilesspectrum[idx]))
                print("*************************************************************************************")
                df.to_csv(os.path.join(tablesdir2, output_pdfile))



            # save for each observation  { event-id, airmass, set of points (wl,flux,errflux, mag,abs,errabs) }
            if(len(mag>0)):
                #all_indexes.append(idx)
                all_indexes.append(count)
                all_airmass.append(am)
                all_lambdas.append(wl)
                all_flux.append(fl)
                all_flux_err.append(errfl)
                all_mag.append(mag)
                all_errmag.append(errmag)
                all_abs.append(abs)
                all_errabs.append(errabs)
                all_dt.append(DT)

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
    all_airmass = np.array(all_airmass)
    all_flux = np.array(all_flux)
    all_flux_err = np.array(all_flux_err)
    all_lambdas = np.array(all_lambdas)
    all_mag=np.array(all_mag)
    all_errmag=np.array(all_errmag)
    all_dt=np.array(all_dt)
    all_flag=np.array(all_flag)

    all_badidx=np.array(all_badidx)
    all_badfn=np.array(all_badfn)



    #assert False
    if 0:
        print("len(all_mairmass)=", len(all_airmass))
        print("all_airmass=",all_airmass)

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





#----------------------------------------------------------------------------------------------------------------------