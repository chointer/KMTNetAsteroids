# 20200724 download_ATLAS, download_Pan
#	   -> MODE check part modify
from PanSTARRS import *
import numpy as np
import pickle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import vstack

def MODE_printer(MODE_6, MODE_pre, MODECHECK):
    MODE_6 = str(MODE_6)
    MODE_pre = str(MODE_pre)
    if sum(MODECHECK == MODE_6) == 0:
        if MODE_6 == MODE_pre:
            assert False, 'Error MODE_printer: MODE_6 == MODE_pre'
        else:                               # case A -> B
            MODE_for_save = MODE_6

    elif sum(MODECHECK == MODE_6) == 1:
        if MODE_6 == MODE_pre:              # case A -> A
            MODE_for_save = MODE_pre
        else:
            if MODE_6 == MODE_pre[:6]:      # case A' -> A'
                MODE_for_save = MODE_pre
            else:                           # case B -> A', B' -> A''
                MODE_for_save = MODE_6
                while sum(MODECHECK == MODE_for_save) == 1:
                    MODE_for_save = MODE_for_save + '0'
    else:
        assert False, 'Error MODE_printer: MODECHECK > 1'
    return MODE_for_save


def getPanSTARRS(savename, ra, dec, radius, constraints, columns, PanDir):
    # flag : asteroid number, 0 = main ast
    try:
        with open(PanDir+'Pan.%s.pickle' % savename, 'rb') as f:
            data = pickle.load(f)
    except:
        print('No Requested Pan Data -> download!')
        data = getstarlist(ra, dec, radius, constraints, columns)
        print('Download Complete')
        with open(PanDir+'Pan.%s.pickle' % savename, 'wb') as f:
            pickle.dump(data, f)
    return data


def download_Pan(ImageName, PanDir, ImgDir, MODECHECK, MODE_pre, radius_sub=(0.365)):
    # radius_sub=0.365 for fitszed ref catalog
    constraints = {'nDetections.gt': 4}
    columns = """objID,raMean,decMean,nDetections,ng,nr,ni,nz,
    gMeanKronMag,rMeanKronMag,iMeanKronMag,zMeanKronMag,
    gMeanKronMagErr,rMeanKronMagErr,iMeanKronMagErr,zMeanKronMagErr,objInfoFlag""".split(',')

    sub_center_pix = np.array([[2304, 2304, 6912, 6912], [2308, 6924, 2308, 6924]])
    hdulist = fits.open(ImgDir + ImageName)

    hdr_header = hdulist[0].header
    MODE = hdr_header['OBJECT'][:6]

    MODE_for_save = MODE_Printer(MODE, MODE_pre, MODECHECK)

    for ChipNum in range(1, 5):
        try:
            with open(PanDir + 'Pan.' + MODE_for_save + '_Chip%s' % ChipNum + '.pickle', 'rb') as f:
                print('Pan.' + MODE_for_save + '_Chip%s' % ChipNum + ' already exists.')
            continue
        except:
            pass

        hdr_chip = hdulist[ChipNum].header
        wcs = WCS(hdr_chip)
        sub_center_ra, sub_center_dec = wcs.wcs_pix2world(sub_center_pix[0], sub_center_pix[1], 1)
        print(sub_center_ra, sub_center_dec)
        for k in range(4):
            table_sub = getPanSTARRS(MODE_for_save + '_Chip%s_Sub%s' % (ChipNum, k),
                                     sub_center_ra[k], sub_center_dec[k],
                                     radius_sub, constraints, columns, PanDir=PanDir)
            if k == 0:
                table_tot = table_sub.copy()
            else:
                duplication = np.isin(table_sub, table_tot)
                table_tot = vstack([table_tot, table_sub[np.logical_not(duplication)]], join_type='exact')

        with open(PanDir + 'Pan.' + MODE_for_save + '_Chip%s' % ChipNum + '.pickle', 'wb') as f:
            pickle.dump(table_tot, f)

    return MODE_for_save




def download_ATLAS(ImageName, RefDir, ImgDir, MainRefDir, MODECHECK, MODE_pre, radius_sub=(0.708)):
    # radius_sub = 0.708 for fit-sized ref catalog
    hdulist = fits.open(ImgDir + ImageName)

    hdr_header = hdulist[0].header
    MODE = hdr_header['OBJECT'][:6]

    MODE_for_save = MODE_Printer(MODE, MODE_pre, MODECHECK)
    
    for ChipNum in range(1, 5):
        # existence check
        try:
            with open(RefDir + 'ATLAS.' + MODE_for_save + '_Chip%s' % ChipNum + '.pickle', 'rb') as f:
                print('ATLAS.' + MODE_for_save + '_Chip%s' % ChipNum + ' already exists.')
            continue
        except:
            pass

        hdr_chip = hdulist[ChipNum].header
        wcs = WCS(hdr_chip)
        sub_center_ra, sub_center_dec = wcs.wcs_pix2world(4608, 4616, 1)
        print(sub_center_ra, sub_center_dec)

        # import ref_catalog
        ref_ra = np.array([(sub_center_ra//90) * 90, (sub_center_ra//90 + 1) * 90], dtype=np.int)
        ref_dec = np.array([(sub_center_dec//30) * 30, (sub_center_dec//30 + 1) * 30], dtype=np.int)
        ref_ra_sign = np.array(['p', 'p'])
        ref_dec_sign = np.array(['p', 'p'])
        if ref_ra[0] < 0: ref_ra_sign[0] = 'm'
        if ref_ra[1] < 0: ref_ra_sign[1] = 'm'
        if ref_dec[0] < 0: ref_dec_sign[0] = 'm'
        if ref_dec[1] < 0: ref_dec_sign[1] = 'm'

        checked_ref = [ref_ra_sign[0] + str(np.abs(ref_ra[0])) + '_' +
                       ref_ra_sign[1] + str(np.abs(ref_ra[1])) + '_' +
                       ref_dec_sign[0] + str(np.abs(ref_dec[0])) + '_' +
                       ref_dec_sign[1] + str(np.abs(ref_dec[1]))]

        main_ref = np.genfromtxt(MainRefDir + 'ATLAS_' + checked_ref[0] + '.cat')
        # [0] ra, [1] dec,
        # [2] g, [3] g_err, [4] r, [5] r_err,
        # [6] i, [7] i_err, [8] z, [9] z_err

        # extract star list
        mask = ((np.cos(np.radians(sub_center_dec)) * (main_ref[:, 0] - sub_center_ra)) ** 2
                + (main_ref[:, 1] - sub_center_dec) ** 2) ** 0.5 < radius_sub
        table = main_ref[mask]
        print(sum(mask))

        edges = np.array([[sub_center_ra + radius_sub, sub_center_dec], [sub_center_ra, sub_center_dec + radius_sub],
                         [sub_center_ra - radius_sub, sub_center_dec], [sub_center_ra, sub_center_dec - radius_sub],
                         [sub_center_ra + radius_sub**0.5, sub_center_dec + radius_sub**0.5],
                         [sub_center_ra + radius_sub**0.5, sub_center_dec - radius_sub**0.5],
                         [sub_center_ra - radius_sub**0.5, sub_center_dec + radius_sub**0.5],
                         [sub_center_ra - radius_sub**0.5, sub_center_dec - radius_sub**0.5]])

        # import additional catalog
        for i in range(8):
            ref_side_ra = np.array([(edges[i, 0] // 90) * 90, (edges[i, 0] // 90 + 1) * 90], dtype=np.int)
            ref_side_dec = np.array([(edges[i, 1] // 30) * 30, (edges[i, 1] // 30 + 1) * 30], dtype=np.int)
            ref_side_ra_sign = np.array(['p', 'p'])
            ref_side_dec_sign = np.array(['p', 'p'])
            if ref_side_ra[0] < 0: ref_side_ra_sign[0] = 'm'
            if ref_side_ra[1] < 0: ref_side_ra_sign[1] = 'm'
            if ref_side_dec[0] < 0: ref_side_dec_sign[0] = 'm'
            if ref_side_dec[1] < 0: ref_side_dec_sign[1] = 'm'
            side_ref_name = ref_ra_sign[0] + str(np.abs(ref_ra[0])) + '_' \
                            + ref_ra_sign[1] + str(np.abs(ref_ra[1])) + '_' \
                            + ref_dec_sign[0] + str(np.abs(ref_dec[0])) + '_' \
                            + ref_dec_sign[1] + str(np.abs(ref_dec[1]))
            if side_ref_name in checked_ref:
                continue
            else:
                print(side_ref_name, checked_ref[0])    # temp
                checked_ref = checked_ref + [side_ref_name]
                side_ref = np.genfromtxt(MainRefDir + 'ATLAS_' + side_ref_name + '.cat')
                mask_side = ((np.cos(np.radians(sub_center_dec)) * (side_ref[:, 0] - sub_center_ra)) ** 2 +
                             (side_ref[:, 1] - sub_center_dec) ** 2)**0.5 < radius_sub
                table = np.vstack([table, side_ref[mask_side]])

        with open(RefDir + 'ATLAS.' + MODE_for_save + '_Chip%s' % ChipNum + '.pickle', 'wb') as f:
            pickle.dump(table, f)

    return MODE_for_save


