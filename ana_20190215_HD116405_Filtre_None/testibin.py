import numpy as np


WLMIN = 400.0
WLMAX = 800.0
NBWLBIN = 40
WLBINWIDTH = (WLMAX - WLMIN) / float(NBWLBIN)

WLMINBIN = np.arange(WLMIN, WLMAX, WLBINWIDTH)
WLMAXBIN = np.arange(WLMIN + WLBINWIDTH, WLMAX + WLBINWIDTH, WLBINWIDTH)


print('WLMINBIN..................................=', WLMINBIN.shape, WLMINBIN)
print('WLMAXBIN..................................=', WLMAXBIN.shape, WLMAXBIN)
print('NBWLBIN...................................=', NBWLBIN)
print('WLBINWIDTH................................=', WLBINWIDTH)


def GetWLBin(wl):
    """

    :param wl: wavelength scalar
    :return: index
    """
    set_ibin=np.where(np.logical_and(WLMINBIN<=wl,WLMAXBIN>wl))[0]


    if len(set_ibin)==1:
        return set_ibin[0]
    else:
        return -1



if __name__ == "__main__":


    wavelengths=np.arange(350.,1200.,50.)

    for wl in wavelengths:
        ibin=GetWLBin(wl)

        print("wavelength = {} ..... iblin= {}".format(wl,ibin))

