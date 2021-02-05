import numpy as np
import pickle, time, sys
from astropy.io import fits
from astropy.table import vstack
from astropy.wcs import WCS
sys.path.append("/data2/SHChoi/phot/python_script/MyModule")
from PanSTARRS import *
from ModuleCommon import dms2deg_header
from ModuleRefDown import getPanSTARRS, download_Pan, download_ATLAS

#"""
Dates = ['20151215', '20151216', '20151217', '20151218', '20151219',
         '20151220', '20151221', '20151222', '20151223', '20151224',
         '20160115', '20160116', '20160117', '20160118', '20160119',
         '20160122', '20161221', '20161222', '20161223', '20161224',
         '20161225', '20161227', '20161228', '20171211', '20171212',
         '20171213', '20171214', '20171215', '20171216', '20171217', '20171218']
#"""

chance = 100
sleeptime = 30

chan = 0
done = np.zeros(len(Dates))

while chan < chance:
    try:
        for da in range(len(Dates)):
            if done[da] == 1:
                pass
            else:
                Date = Dates[da]
                print('Conducting ' + Date)
                t0 = time.time()

                imagelist = np.genfromtxt('/data2/SHChoi/phot/obs' + Date + '/KMTNet_0_input/image' + Date + '.list', dtype=np.str)
                ImgDir = '/data2/SHChoi/data/' + Date + '/WCSCOR/'
                PhoDir = '/data2/SHChoi/data/' + Date + '/PHOT/'
                AstDir = '/data2/SHChoi/phot/obs' + Date + '/KMTNet_SkyBoT/'
                RefDir = '/data2/SHChoi/phot/obs' + Date + '/KMTNet_RefCat/'    # reference star catalog
                AtlDir = '/data2/SHChoi/ATLAS_SkyMapper/'
                OutDir = '/data2/SHChoi/phot/obs' + Date + '/KMTNet_1_match/'
                PhotModeSaveDir = '/data2/SHChoi/phot/obs' + Date + '/KMTNet_0_input/'

                N_img = len(imagelist)
                MODECHECKLIST = ['startpoint']
                MODE_pre = ['startpoint']
                for i in range(N_img):
                    print('[%s/%s]' % (i+1, N_img), imagelist[i])
                    # deg check
                    hdulist = fits.open(ImgDir + imagelist[i] + '.miss.fits')
                    hdr_header = hdulist[0].header
                    DEC_c = dms2deg_header(hdr_header['DEC'])
                    if float(DEC_c) >= -29.5:
                        MODE_for_save = download_Pan(imagelist[i] + '.miss.fits', PanDir=RefDir, ImgDir=ImgDir,
                                                     MODECHECK=np.array(MODECHECKLIST), MODE_pre=MODE_pre)
                    else:
                        MODE_for_save = download_ATLAS(imagelist[i] + '.miss.fits', RefDir=RefDir, ImgDir=ImgDir,
                                                       MainRefDir=AtlDir, MODECHECK=np.array(MODECHECKLIST),
                                                       MODE_pre=MODE_pre)

                    if MODE_for_save != MODE_pre:
                        MODECHECKLIST.append(MODE_for_save)
                    MODE_pre = np.copy(MODE_for_save)

                done[da] = 1

    except FileNotFoundError:
        print('--- retry after %s s [%s] ---' % (sleeptime, chan))
        print(DEC_c)
        time.sleep(sleeptime)
        chan += 1

    if sum(done) == len(done):
        break

print('all done')
