# sextractor params : FLUX_APER, FLUXERR_APER, MAG_APER, MAGERR_APER, FLUX_AUTO, FLUXERR_AUTO, MAG_AUTO, MAGERR_AUTO,
#                     FLUX_PETRO, FLUXERR_PETRO, MAG_PETRO, MAGERR_PETRO, FLUX_WIN, FLUXERR_WIN, MAG_WIN, MAGERR_WIN,
#                     SNR_WIN, KRON_RADIUS, BACKGROUND, X_IMAGE, Y_IMAGE, X_WIN_IMAGE, YWIN_IMAGE,
#                     ERRAWIN_IMAGE, ERRBWIN_IMAGE, ERRTHETAWIN_IMAGE, FLAGS, ELLIPTICITY, FLUX_RADIUS
# ccd chip number : master[0]  M[1]  K[2]  N[3]  T[4]
# 20200702  match.Match
#           -> Make available using "two types" of reference catalog: PanSTARRS, ATLAS
#           -> Change some parameters' name (ex std_type)
#           match.Match
#           -> Add additional info into the output: asteroid's 'ellipticity' and 'FLAG' from SExtractor, RefType
# 20200723  match.Match
#           -> Make AstName
#           match.Match
#           -> have mag-mag slope 3rd trial, nevertheless, if it fails, reject
# 20200724  match.__init__
#	    -> dealing with MODE, if -> while
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.time import Time
from astropy.table import vstack, Table
from astroquery.jplhorizons import Horizons
import pickle
import requests
import time
import sys
sys.path.append("/data2/SHChoi/phot/python_script/MyModule")
from PanSTARRS import *
from linfit_sigclip import linfit_sigclip, sigma_boundary
from ModuleCommon import hms2deg_header, dms2deg_header, hms2deg_SkyBoT, dms2deg_SkyBoT, err_Add, err_Mul
from ModuleRefDown import MODE_printer

def tran_lin(f, b, r, C0, C1, fe, be, re, C0e, C1e):
    result = f + C0 + C1*(b-r)
    c = b - r
    ce = err_Add([be, re])
    Err_lin = err_Mul([np.ones_like(f) * C1, c], [np.ones_like(f) * C1e, ce])
    return result, err_Add([fe, C0e, Err_lin])

def tran_quad(f, b, r, D0, D1, D2, fe, be, re, D0e, D1e, D2e):
    result = f + D0 + D1*(b-r) + D2*(b-r)**2
    c = b - r
    ce = err_Add([be, re])
    Err_lin = err_Mul([np.ones_like(f) * D1, c], [np.ones_like(f) * D1e, ce])
    Err_quad = err_Mul([np.ones_like(f) * D2, c**2], [np.ones_like(f) * D2e, c*ce*(2**0.5)])
    return result, err_Add([fe, D0e, Err_lin, Err_quad])

def getastlist(ep, ra, dec, rd, mime='text', output='obs', loc='W93', err_pos=2, obj=101):
    query = 'http://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php?' + \
            '&-ep=%f' %ep + '&-ra=%f' %ra + '&-dec=%f' %dec + '&-rd=%f' %rd + \
            '&-mime=%s' %mime + '&-output=%s' %output + '&-loc=%s' %loc + \
            '&-filter=%f' %err_pos + '&-objFilter=%d' %obj
    res = requests.get(query)
    tab = res.content.decode().splitlines()
    while int(tab[0][7:]) == -1:
        print('getastlist : query Failure... retry')
        time.sleep(2)
        res = requests.get(query)
        tab = res.content.decode().splitlines()
    tab = np.char.split(np.array(tab[2:]), sep=' | ').tolist()
    tab = np.array(tab[1:])
    return tab

def match2tables(stdx, stdy, obsx, obsy, d=5):
    idx_std = []
    idx_obs = []
    for a in range(len(stdx)):
        dist = ((stdx[a]-obsx)**2 + (stdy[a]-obsy)**2)**0.5
        idx = np.where(dist < d)[0]
        if len(idx) == 1:
            idx_std.append(a)
            idx_obs.append(idx[0])
    idx_std = np.array(idx_std)
    idx_obs = np.array(idx_obs)
    return idx_std, idx_obs

def PantoBVRI(output, quadfit=False, **kwargs):
    #g, r, i, ge, re, ie):
    if output == 'B':
        g = kwargs['g']; ge = kwargs['ge']
        r = kwargs['r']; re = kwargs['re']
        B, Be = tran_lin(g, g, r, 0.194, 0.561, ge, ge, re, 0.001, 0.002)
        if quadfit == True:
            idx_red = g - r > 1.5
            B_quad, Be_quad = tran_quad(g, g, r, 0.199, 0.540, 0.016, ge, ge, re, 0.001, 0.004, 0.003)
            B[idx_red] = B_quad[idx_red]
            Be[idx_red] = Be_quad[idx_red]
            mask_safe = (g - r >= -0.5)
        else: mask_safe = (g - r >= -0.5) & (g - r <= 1.5)
        return B, Be, mask_safe
    elif output == 'V':
        g = kwargs['g']; ge = kwargs['ge']
        r = kwargs['r']; re = kwargs['re']
        #V = r - 0.017 + 0.492 * (g - r)
        #Ve = err_Add([re, 0.001, err_Mul([np.ones_like(g) * 0.492, g - r], [np.ones_like(g) * 0.001, err_Add([ge, re])])])
        V, Ve = tran_lin(g, g, r, -0.017, -0.508, ge, ge, re, 0.001, 0.001) #0.002) !!! type mistaked !!!
        if quadfit == True:
            idx_red = g - r > 1.5
            V_quad, Ve_quad = tran_quad(g, g, r, -0.020, -0.498, -0.008, ge, ge, re, 0.001, 0.002, 0.002)
            V[idx_red] = V_quad[idx_red]
            Ve[idx_red] = Ve_quad[idx_red]
            mask_safe = (g - r >= -0.5)
        else: mask_safe = (g - r >= -0.5) & (g - r <= 1.5)
        return V, Ve, mask_safe
    elif output == 'R':
        #g = kwargs['g']; ge = kwargs['ge']
        r = kwargs['r']; re = kwargs['re']
        i = kwargs['i']; ie = kwargs['ie']
        #R = r - 0.142 - 0.166 * (g - r)
        #Re = err_Add([re, 0.001, err_Mul([np.ones_like(g) * -0.166, g - r], [np.ones_like(g) * 0.001, err_Add([ge, re])])])
        R, Re = tran_lin(r, r, i, -0.166, -0.275, re, re, ie, 0.000, 0.002)
        if quadfit == True:
            idx_red = r - i > 1.0
            R_quad, Re_quad = tran_quad(r, r, i, -0.172, -0.221, -0.081, re, re, ie, 0.000, 0.002, 0.002)
            R[idx_red] = R_quad[idx_red]
            Re[idx_red] = Re_quad[idx_red]
            mask_safe = (r - i >= -0.4)
        else: mask_safe = (r - i >= -0.4) & (r - i <= 1.0)
        return R, Re, mask_safe
    elif output == 'I':
        g = kwargs['g']; ge = kwargs['ge']
        r = kwargs['r']; re = kwargs['re']
        i = kwargs['i']; ie = kwargs['ie']
        I, Ie = tran_lin(i, r, i, -0.416, -0.214, ie, re, ie, 0.001, 0.003)
        # I, Ie = tran_lin(i, g, r, -0.376, -0.167, ie, ge, re, 0.001, 0.001)
        if quadfit == True:
            idx_red = r - i > 1.0
            #idx_red = g - r > 1.5
            I_quad, Ie_quad = tran_quad(i, r, i, -0.433, -0.040, -0.263, ie, re, ie, 0.001, 0.003, 0.003)
            #I_quad, Ie_quad = tran_quad(i, g, r, -0.387, -0.123, -0.034, ie, ge, re, 0.001, 0.004, 0.003)
            I[idx_red] = I_quad[idx_red]
            Ie[idx_red] = Ie_quad[idx_red]
            mask_safe = (r - i >= -0.4)
            # mask_safe = (g - r >= -0.5)
        else:
            mask_safe = (r - i >= -0.4) & (r - i <= 1.0)
            # mask_safe = (g - r >= -0.5) & (g - r <= 1.5)
        return I, Ie, mask_safe
    else: assert False, 'unacceptable output type : %s' %output

def GetMaskFluxRad(FluxRad, sigclip = 1, Niter = 10, sigmax = 0.5, sigmin= 0.2) :
    mask = np.full(len(FluxRad), True)
    for k in range(Niter):
        median_FR = np.median(FluxRad[mask])
        std_FR = np.std(FluxRad[mask])
        mask = (FluxRad >= median_FR - sigclip * std_FR) * (FluxRad <= median_FR + sigclip * std_FR)
        if std_FR >= sigmax: continue
        elif std_FR <= sigmin:
            std_FR = sigmin
            mask = (FluxRad >= median_FR - sigclip * sigmin) * (FluxRad <= median_FR + sigclip * sigmin)
            break
    return mask, median_FR, std_FR

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


class match:
    def __init__(self, ImageName, ImgDir, PhotDir, MODECHECK, MODE_pre):
        print('=== === === match : %s === === ===' %ImageName)
        ### get image info ###
        self.ImgName = ImageName
        self.hdulist = fits.open(ImgDir + ImageName + '.miss.fits')
        self.hdr_header = self.hdulist[0].header
        self.FILTER = self.hdr_header['FILTER']
        self.EXPTIME = self.hdr_header['EXPTIME']
        self.DATE = self.hdr_header['DATE-OBS']     # UTC
        self.MODE = self.hdr_header['OBJECT'][:6]
        #assert False # check mode handling !
        #while (self.MODE != MODE_pre) and (sum(MODECHECK == self.MODE) == 1):
        #    self.MODE = self.MODE + '0'
        self.MODE = MODE_printer(self.MODE, MODE_pre, MODECHECK)
        self.JD = Time(self.DATE).jd
        self.RA_c = hms2deg_header(self.hdr_header['RA'])
        self.DEC_c = dms2deg_header(self.hdr_header['DEC'])

        self.ImgSet = [0, self.hdulist[1].data, self.hdulist[2].data, self.hdulist[3].data, self.hdulist[4].data]
        phothdulist = fits.open(PhotDir + self.ImgName + '.cat')
        self.ObsTabSet = [0, phothdulist[2].data, phothdulist[4].data,
                          phothdulist[6].data, phothdulist[8].data]

        print('MODE = ', self.MODE)


    def GetAstTab(self, AstTabName, AstDir, MagLim120=22,
                  rd=1.6, mime='text', output='obs', loc='W93', err_pos=2, obj=101, boundary=10):
        """
        return self.AstTab (output='obs')
             [0] number, [1] name, [2] ra, [3] dec, [4] class(loc), [5] Mv,
             [6] err_pos, [7] angular dist, [8] motion, [9] motion,
             [10] geocenetric dist, [11] heliocentric dist,
             [12] Phase angle, [13] Solar elongation, [14] CHIPNUM, [15] AMPNUM
        """
        try:
            with open(AstDir+AstTabName+'.pickle', 'rb') as f:
                AstTab = pickle.load(f)
        except:
            print('No Requested Ast Data -> Download!')
            AstTab = getastlist(self.JD, self.RA_c, self.DEC_c,
                                rd=rd, mime=mime, output=output, loc=loc, err_pos=err_pos, obj=obj)
            # AstTab : [0] number, [1] name, [2] ra, [3] dec, [4] class(loc), [5] Mv,
            #          [6] err_pos, [7] body-to-center angular dist, [8] motion, [9] motion,
            #          [10] geocentric dist, [11] heliocentric dist, [12] Phase angle,
            #          [13] Solar elongation

            #AstNumException = -1
            for k in range(len(AstTab)):
                if AstTab[k, 0] == ' -':
                    print('not-allocated asteroid : %s %s' % (AstTab[k, 0], AstTab[k, 1]))
                    AstTab[k, 0] = AstTab[k, 1].replace(" ", "_")   #AstNumException
                    #AstNumException -= 1

            AstTab[:, 2] = hms2deg_SkyBoT(AstTab[:, 2])
            AstTab[:, 3] = dms2deg_SkyBoT(AstTab[:, 3])
            with open(AstDir+AstTabName+'.pickle', 'wb') as f:
                pickle.dump(AstTab, f)


        ### Chip & Amplifier index ###
        AstTab_CHIPAMP = np.zeros((len(AstTab), 2), dtype=np.int) - 1
        AstTab_xy = np.zeros((len(AstTab), 2), dtype=np.float) - 1
        for ChipNum_temp in range(1, 5):
            hdr_temp = self.hdulist[ChipNum_temp].header
            wcs = WCS(hdr_temp)
            AstTab_x_temp, AstTab_y_temp = wcs.wcs_world2pix(AstTab[:, 2].astype(float), AstTab[:, 3].astype(float), 1)
            AstTab_AMPNUM_temp = (AstTab_x_temp//1152).astype(int)
            idx_inside = np.where((AstTab_AMPNUM_temp >= 0) & (AstTab_AMPNUM_temp <= 7) &
                                  (AstTab_x_temp > 0 + boundary) & (AstTab_x_temp < 9216 - boundary) &
                                  (AstTab_y_temp > 0 + boundary) & (AstTab_y_temp < 9232 - boundary))
            # boundary is applied because of (1) the concern that edge distortion error and
            # (2) that some of (ra,dec) are converted to (0,0), not their correct (x,y)
            # (2) usually happens to asteroids far from the query center
            AstTab_CHIPAMP[idx_inside, 0] = ChipNum_temp
            AstTab_CHIPAMP[idx_inside, 1] = AstTab_AMPNUM_temp[idx_inside]
            AstTab_xy[idx_inside, 0] = AstTab_x_temp[idx_inside]
            AstTab_xy[idx_inside, 1] = AstTab_y_temp[idx_inside]
        AstTab = np.concatenate((AstTab, AstTab_CHIPAMP), axis=1)
        MagLim = MagLim120 + 2.5 * np.log10(self.EXPTIME/120)
        mask = (AstTab[:, 5].astype(float) < MagLim) * (AstTab[:, 14].astype(float) > 0) * (AstTab[:, 15].astype(float) >= 0)
        #mask = (AstTab[:, 8].astype(float) > 0) * (AstTab[:, 9].astype(float) >= 0)
        self.AstTab = AstTab[mask]
        self.AstTab_xy = AstTab_xy[mask]
        print('From SkyBoT, %s/%s' % (len(self.AstTab), len(AstTab)))
        return self.AstTab

    def Match(self, AstInfo, RefDir, RefRad, StdLim_Merr, ObsLim_Merr, IDist, IDist_star,
              OutDir, FLUX_RADIUS_SIG_max, FLUX_RADIUS_SIG_min, Mag_Diff_max, quadfitTF, match_N_min=34,
              CheckImage=True, dpi=150):
        # IDist : identification distance [pix]
        # AstTab : [0] number, [1] name, [2] ra, [3] dec, [4] class(loc), [5] Mv,
        #          [6] err_pos, [7] body-to-center angular dist, [8] motion, [9] motion,
        #          [10] geocentric dist, [11] heliocentric dist, [12] Phase angle,
        #          [13] Solar elongation, [14] CHIPNUM, [15] AMPNUM
        AstNum = AstInfo[0]
        AstRa = float(AstInfo[2])
        AstDec = float(AstInfo[3])
        AstMv = float(AstInfo[5])
        AstCHIPNUM = int(AstInfo[14])
        AstAMPNUM = int(AstInfo[15])
        print(' === === === Match :%s === === === ' % AstNum)

        #####################
        ### get std table ###
        #####################
        AstName_type = False
        try:
            AstName = int(AstNum)
        except ValueError:
            AstName = AstNum.strip().replace("/", "_").replace(" ", "_")
            print(AstName)
            AstName_type = True

        try:
            with open(RefDir + 'Pan.' + self.MODE + '_Chip%s' % AstCHIPNUM + '.pickle', 'rb') as f:
                PanTab = pickle.load(f)
                std_type = 'Pan'
                PanTab['newID'] = np.arange(len(PanTab)) + 1
                PanTab['Var'] = (PanTab['objInfoFlag'] % 512) // 4
                print(' # Var Obj = %s' % sum(PanTab['Var'] > 0))
                idx_Pan = np.where((PanTab['gMeanKronMagErr'] > -10) & (PanTab['rMeanKronMagErr'] > -10) &
                                   (PanTab['iMeanKronMagErr'] > -10) & (PanTab['zMeanKronMagErr'] > -10) &
                                   (PanTab['gMeanKronMagErr'] <= StdLim_Merr) &
                                   (PanTab['rMeanKronMagErr'] <= StdLim_Merr) &
                                   (PanTab['iMeanKronMagErr'] <= StdLim_Merr) &
                                   (PanTab['zMeanKronMagErr'] <= StdLim_Merr) &
                                   (PanTab['Var'] == 0) &
                                   (((np.cos(np.radians(AstDec)) * (PanTab['raMean'] - AstRa)) ** 2 +
                                     (PanTab['decMean'] - AstDec) ** 2) ** 0.5 < RefRad))
                PanTab = PanTab[idx_Pan]
                wcs = WCS(self.hdulist[int(AstInfo[14])].header)
                StdX, StdY = wcs.wcs_world2pix(PanTab['raMean'], PanTab['decMean'], 1)
                # table ['ID', 'ra', 'dec', 'x', 'y', 'g', 'ge', 'r', 're', 'i', 'ie', 'z', 'ze']
                StdTab = Table([PanTab['newID'], PanTab['raMean'], PanTab['decMean'], StdX, StdY,
                                PanTab['gMeanKronMag'], PanTab['gMeanKronMagErr'],
                                PanTab['rMeanKronMag'], PanTab['rMeanKronMagErr'],
                                PanTab['iMeanKronMag'], PanTab['iMeanKronMagErr'],
                                PanTab['zMeanKronMag'], PanTab['zMeanKronMagErr']],
                               names=('ID', 'ra', 'dec', 'x', 'y', 'g', 'ge', 'r', 're', 'i', 'ie', 'z', 'ze'))

        except FileNotFoundError:
            with open(RefDir + 'ATLAS.' + self.MODE + '_Chip%s' % AstCHIPNUM + '.pickle', 'rb') as f:
                AtlTab = pickle.load(f)
                # ATLAS: [0] ra, [1] dec, [2] g, [3] g_err, [4] r, [5] r_err, [6] i, [7] i_err, [8] z, [9] z_err
                std_type = 'ATLAS'
                AtlID = np.arange(len(AtlTab)) + 1
                idx_Atl = np.where((AtlTab[:, 3] > -10) & (AtlTab[:, 5] > -10) &
                                   (AtlTab[:, 7] > -10) & (AtlTab[:, 9] > -10) &
                                   (AtlTab[:, 2] > StdLim_Merr) & (AtlTab[:, 4] > StdLim_Merr) &
                                   (AtlTab[:, 6] > StdLim_Merr) & (AtlTab[:, 8] > StdLim_Merr) &
                                   (((np.cos(np.radians(AstDec)) * (AtlTab[:, 0] - AstRa)) ** 2 +
                                     (AtlTab[:, 1] - AstDec) ** 2) ** 0.5 < RefRad))
                AtlID = AtlID[idx_Atl]
                AtlTab = AtlTab[idx_Atl]
                wcs = WCS(self.hdulist[int(AstInfo[14])].header)
                StdX, StdY = wcs.wcs_world2pix(AtlTab[:, 0], AtlTab[:, 1], 1)
                # table ['ID', 'ra', 'dec', 'x', 'y', 'g', 'ge', 'r', 're', 'i', 'ie', 'z', 'ze']
                StdTab = Table([AtlID, AtlTab[:, 0], AtlTab[:, 1], StdX, StdY,
                                AtlTab[:, 2], AtlTab[:, 3], AtlTab[:, 4], AtlTab[:, 5],
                                AtlTab[:, 6], AtlTab[:, 7], AtlTab[:, 8], AtlTab[:, 9]],
                               names=('ID', 'ra', 'dec', 'x', 'y', 'g', 'ge', 'r', 're', 'i', 'ie', 'z', 'ze'))

        #####################
        ### get obs table ###
        #####################
        idx_Obs = ((self.ObsTabSet[AstCHIPNUM]['X_IMAGE'] >= AstAMPNUM*1152) *
                   (self.ObsTabSet[AstCHIPNUM]['X_IMAGE'] < (AstAMPNUM+1)*1152) *
                   (self.ObsTabSet[AstCHIPNUM]['MAGERR_AUTO'] <= ObsLim_Merr))
        # FLUX RADIUS constraint would be performed later
        if CheckImage is True:
            idx_Obs_CheckImage = np.where((self.ObsTabSet[AstCHIPNUM]['X_IMAGE'] >= AstAMPNUM*1152) &
                                          (self.ObsTabSet[AstCHIPNUM]['X_IMAGE'] < (AstAMPNUM+1)*1152))
            fig_ObsX = self.ObsTabSet[AstCHIPNUM]['X_IMAGE'][idx_Obs_CheckImage]
            fig_ObsY = self.ObsTabSet[AstCHIPNUM]['Y_IMAGE'][idx_Obs_CheckImage]
            fig_ObsMag = self.ObsTabSet[AstCHIPNUM]['MAG_AUTO'][idx_Obs_CheckImage]
            fig_ObsMerr = self.ObsTabSet[AstCHIPNUM]['MAGERR_AUTO'][idx_Obs_CheckImage]
        ObsTab = self.ObsTabSet[AstCHIPNUM][idx_Obs]

        ### find asteroid in table ###
        AstX, AstY = wcs.wcs_world2pix(AstRa, AstDec, 1)
        idx_AstInObsTab = np.where(((ObsTab['X_IMAGE'] - AstX)**2 + (ObsTab['Y_IMAGE'] - AstY)**2)**0.5 <= IDist)[0]
        idx_AstOutStdTab = np.where(((StdTab['x'] - AstX)**2 + (StdTab['y'] - AstY)**2)**0.5 <= IDist)[0]
        idx_AstAdj = np.where(((self.AstTab_xy[:, 0] - AstX)**2 + (self.AstTab_xy[:, 1] - AstY)**2)**0.5 <= IDist)[0]
        if len(idx_AstInObsTab) == 1 and len(idx_AstOutStdTab) == 0 and len(idx_AstAdj) == 1:
            AstRa_Obs, AstDec_Obs = wcs.wcs_pix2world(ObsTab['X_IMAGE'][idx_AstInObsTab], ObsTab['Y_IMAGE'][idx_AstInObsTab], 1)
        else:
            print('!!! asteroid %s rejected : N_idx_ast = %s / N_idx_star = %s !!!'
                  % (AstNum, len(idx_AstInObsTab), len(idx_AstOutStdTab)))
            print('[ Match Failure ] \n')
            return np.array([-1])
        # asteroid should be matched with a single source in ObsTable and matched with no source in StdTable

        ### match ###
        """
        plt.figure()
        print(AstInfo[8])
        plt.scatter(AstX, AstY, s=20, alpha=0.7, label='ast')
        plt.scatter(StdTab['x'], StdTab['y'], s=5, c='k', alpha=0.5, label='std')
        plt.scatter(ObsTab['X_IMAGE'], ObsTab['Y_IMAGE'], s=10, c='tomato', alpha=0.5, label='obs')
        plt.legend()
        plt.figure()
        plt.scatter(AstRa, AstDec, s=20, alpha=0.7)
        plt.scatter(StdTab['ra'], StdTab['dec'], s=5, c='gray', alpha=0.6, label='std')
        obsx, obsy = wcs.wcs_pix2world(ObsTab['X_IMAGE'], ObsTab['Y_IMAGE'], 1)
        plt.scatter(obsx, obsy, s=10, c='tomato', alpha=0.5, label='obs')
        plt.legend()
        plt.show()
        """
        idx_Std_match, idx_Obs_match = \
            match2tables(StdTab['x'], StdTab['y'], ObsTab['X_IMAGE'], ObsTab['Y_IMAGE'], d=IDist_star)

        # Matched Data Number Check 1
        if len(idx_Std_match) == 0:
            print('!!! [match] asteroid %s rejected : Too Small Num of Matched Stars, %s !!!' 
                  % (AstNum, len(idx_Std_match)))
            return np.array([len(idx_Std_match)])

        ### Get Matched Table ###
        #print(len(idx_Std_match))
        MatTab = np.zeros((len(idx_Std_match), 9))
        # [0] newID, [1] X_obs, [2] Y_obs, [3] M_std, [4] Me_std, [5] M_obs, [6] Me_obs, [7] ra_obs, [8] dec_obs
        MatTab[:, 0] = StdTab['ID'][idx_Std_match]
        MatTab[:, 1] = ObsTab['X_IMAGE'][idx_Obs_match]
        MatTab[:, 2] = ObsTab['Y_IMAGE'][idx_Obs_match]

        if self.FILTER == 'B' or self.FILTER == 'V' or self.FILTER == 'R' or self.FILTER == 'I':
            PanMag_tr, PanMerr_tr, mask_safe = \
                PantoBVRI(self.FILTER, quadfit=quadfitTF,
                          g=StdTab['g'][idx_Std_match], ge=StdTab['ge'][idx_Std_match],
                          r=StdTab['r'][idx_Std_match], re=StdTab['re'][idx_Std_match],
                          i=StdTab['i'][idx_Std_match], ie=StdTab['ie'][idx_Std_match])
            MatTab[:, 3] = PanMag_tr
            MatTab[:, 4] = PanMerr_tr
        elif self.FILTER == 'g' or self.FILTER == 'r' or self.FILTER == 'i' or self.FILTER == 'z':
            MatTab[:, 3] = StdTab[self.FILTER][idx_Std_match]
            MatTab[:, 4] = StdTab[self.FILTER + 'e'][idx_Std_match]
        else:
            assert False, 'filter error : %s        check savetxt too!??' % self.FILTER

        MatTab[:, 5] = ObsTab['MAG_AUTO'][idx_Obs_match]
        MatTab[:, 6] = ObsTab['MAGERR_AUTO'][idx_Obs_match]
        ObsRa, ObsDec = wcs.wcs_pix2world(MatTab[:, 1], MatTab[:, 2], 1)
        MatTab[:, 7] = ObsRa
        MatTab[:, 8] = ObsDec
        Mat_FLUX_RADIUS = ObsTab['FLUX_RADIUS'][idx_Obs_match]

        # Matched Data Number Check 2
        if len(idx_Std_match) < match_N_min:
            print('!!! [match] asteroid %s rejected : Too Small Num of Matched Stars, %s !!!'
                  % (AstNum, len(idx_Std_match)))
            figLN = plt.figure(figsize=(7, 7))  # low number of matched stars
            figLN.suptitle('%s\tRA%.4f\tDEC%.4f' % (AstName, AstRa, AstDec))
            axLN = figLN.add_subplot(1, 1, 1)
            axLN.errorbar(MatTab[:, 5], MatTab[:, 3],
                          xerr=MatTab[:, 6], yerr=MatTab[:, 4],
                          fmt='o', markersize=2, c='k', ecolor='k', elinewidth=0.3, capsize=1, capthick=0.5)
            axLN.set_xlabel('Instrumental Magnitude [mag]')
            axLN.set_ylabel('Standard Magnitude [mag]')
            plt.savefig(OutDir + 'LowNumStar_' + self.MODE + '_%s.png' % AstName, dpi=dpi)
            plt.close('all')
            return np.array([len(idx_Std_match)])


        ### 'FLUX_RADIUS' sigma clipping ###
        mask_FR, FR_median, FR_std = GetMaskFluxRad(Mat_FLUX_RADIUS, Niter=4,
                                                    sigmax=FLUX_RADIUS_SIG_max, sigmin=FLUX_RADIUS_SIG_min)

        ### 'M_obs - M_std' slope check & outlier clipping ###
        mag_bright = np.percentile(MatTab[:, 5], 5)        # mag_bright = 11 + 2.5 * np.log10(self.EXPTIME/60)
        # 5% due to bright stars distribute more sparsely than faint stars on the mag-mag plot.
        mag_faint = np.percentile(MatTab[:, 5], 90)         # mag_faint = 18 + 2.5 * np.log10(self.EXPTIME/60)
        mask_fit = (MatTab[:, 5] >= mag_bright) * (MatTab[:, 5] <= mag_faint) * (mask_FR == True)
        # mask_fit = (MatTab[:, 3] >= mag_bright) * (MatTab[:, 3] <= mag_faint) * (mask_FR == True)
        mask, fit, Niter = linfit_sigclip(MatTab[:, 5][mask_fit], MatTab[:, 3][mask_fit],
                                          MatTab[:, 6][mask_fit], MatTab[:, 4][mask_fit],
                                          sigmaclip=10, maxiter=1)
        mask_fit_magcut = np.absolute(MatTab[:, 3][mask_fit] - (fit[0][0] + fit[0][1]*MatTab[:, 5][mask_fit])) \
                          <= Mag_Diff_max
        mask, fit, Niter = linfit_sigclip(MatTab[:, 5][mask_fit][mask_fit_magcut], MatTab[:, 3][mask_fit][mask_fit_magcut],
                                          MatTab[:, 6][mask_fit][mask_fit_magcut], MatTab[:, 4][mask_fit][mask_fit_magcut],
                                          sigmaclip=10, maxiter=1)
        coe = fit[0]        # coesig = np.diag(fit[1])**0.5
        if coe[1] < 0.9 or coe[1] > 1.1:
            mask_FR_not = np.logical_not(mask_FR)
            fig = plt.figure(figsize=(16, 7))
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_xlabel('FLUX_RADIUS')
            ax1.set_ylabel('Instrumental Magnitude [mag]')
            ax1.scatter(Mat_FLUX_RADIUS[mask_FR], MatTab[:, 5][mask_FR], s=3, c='k')
            ax1.scatter(Mat_FLUX_RADIUS[mask_FR_not], MatTab[:, 5][mask_FR_not], s=2, c='gray', alpha=0.5)
            ax1.axvline(FR_median, linewidth=1.2, linestyle='--', c='k')
            ax1.axvline(FR_median + FR_std, linewidth=1, linestyle=':', c='k')
            ax1.axvline(FR_median - FR_std, linewidth=1, linestyle=':', c='k')
            ax1.set_ylim(np.max(MatTab[:, 5])+1, np.min(MatTab[:, 5])-1)
            ax1.set_xlim(FR_median-0.5, FR_median+1.5)

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.errorbar(MatTab[mask_FR, 5], MatTab[mask_FR, 3],
                         xerr=MatTab[mask_FR, 6], yerr=MatTab[mask_FR, 4],
                         fmt='o', markersize=2, c='k', ecolor='k', elinewidth=0.3, capsize=1, capthick=0.5)
            if sum(mask_FR_not) != 0:
                ax2.errorbar(MatTab[mask_FR_not, 5], MatTab[mask_FR_not, 3],
                             xerr=MatTab[mask_FR_not, 6], yerr=MatTab[mask_FR_not, 4], fmt='o', markersize=2,
                             c='gray', ecolor='gray', elinewidth=0.3, capsize=1, capthick=0.5, alpha=0.6)
            fig_fitx = np.linspace(min(MatTab[:, 5]) - 0.1, max(MatTab[:, 5]) + 0.1)
            ax2.plot(fig_fitx, coe[0] + coe[1] * fig_fitx, linestyle='--', c='k', linewidth=0.7,
                     label='%.4f+%.4fx' % (coe[0], coe[1]))
            ax2.vlines(mag_bright, ymin=coe[0]+coe[1]*mag_bright-1,
                        ymax=coe[0]+coe[1]*mag_bright+1, linewidth=0.7, colors='k')
            ax2.vlines(mag_faint, ymin=coe[0]+coe[1]*mag_faint-1,
                        ymax=coe[0]+coe[1]*mag_faint+1, linewidth=0.7, colors='k')
            ax2.set_xlabel('Instrumental Magnitude [mag]')
            ax2.set_ylabel('Standard Magnitude [mag]')

            # Fitting using central data (central of instrumental mag) (2nd trial)
            fit_central_mean = np.mean(MatTab[:, 5][mask_fit][mask_fit_magcut])
            fit_central_std = np.std(MatTab[:, 5][mask_fit][mask_fit_magcut])
            fit_central_sigma = 2
            mask_fit_mag_central = np.absolute(MatTab[:, 5][mask_fit][mask_fit_magcut] - fit_central_mean) \
                                   <= fit_central_sigma * fit_central_std
            mask_cen, fit_cen, Niter_cen = \
                linfit_sigclip(MatTab[:, 5][mask_fit][mask_fit_magcut][mask_fit_mag_central],
                               MatTab[:, 3][mask_fit][mask_fit_magcut][mask_fit_mag_central],
                               MatTab[:, 6][mask_fit][mask_fit_magcut][mask_fit_mag_central],
                               MatTab[:, 4][mask_fit][mask_fit_magcut][mask_fit_mag_central], sigmaclip=10, maxiter=1)
            ax2.plot(fig_fitx, fit_cen[0][0]+fit_cen[0][1]*fig_fitx, linestyle='--', c='b', linewidth=0.7,
                     label='%.4f+%.4fx' % (fit_cen[0][0], fit_cen[0][1]))
            mag_fit_cen_bright = np.min(MatTab[:, 5][mask_fit][mask_fit_magcut][mask_fit_mag_central])
            mag_fit_cen_faint = np.max(MatTab[:, 5][mask_fit][mask_fit_magcut][mask_fit_mag_central])
            ax2.vlines(mag_fit_cen_bright, ymin=fit_cen[0][0]+fit_cen[0][1]*mag_fit_cen_bright-1,
                        ymax=fit_cen[0][0]+fit_cen[0][1]*mag_fit_cen_bright+1, linewidth=0.7, colors='b')
            print(mag_fit_cen_faint, fit_cen[0][0]+fit_cen[0][1]*mag_fit_cen_faint-1, fit_cen[0][0]+fit_cen[0][1]*mag_fit_cen_faint+1)
            ax2.vlines(mag_fit_cen_faint, ymin=fit_cen[0][0]+fit_cen[0][1]*mag_fit_cen_faint-1,
                        ymax=fit_cen[0][0]+fit_cen[0][1]*mag_fit_cen_faint+1, linewidth=0.7, colors='b')
            # excluded member which were members of mask_FR
            ax2.legend(loc='upper left')

            # 3rd trial
            if np.absolute(fit_cen[0][1] - 1) > 0.1:
                fit_central_sigma = fit_central_sigma - 0.5
                mask_fit_mag_central = np.absolute(MatTab[:, 5][mask_fit][mask_fit_magcut] - fit_central_mean) \
                                       <= fit_central_sigma * fit_central_std
                mask_cen, fit_cen, Niter_cen = \
                    linfit_sigclip(MatTab[:, 5][mask_fit][mask_fit_magcut][mask_fit_mag_central],
                                   MatTab[:, 3][mask_fit][mask_fit_magcut][mask_fit_mag_central],
                                   MatTab[:, 6][mask_fit][mask_fit_magcut][mask_fit_mag_central],
                                   MatTab[:, 4][mask_fit][mask_fit_magcut][mask_fit_mag_central], sigmaclip=10, maxiter=1)
                ax2.plot(fig_fitx, fit_cen[0][0] + fit_cen[0][1] * fig_fitx, linestyle='--', c='b', linewidth=0.7,
                         label='%.4f+%.4fx' % (fit_cen[0][0], fit_cen[0][1]))
                mag_fit_cen_bright = np.min(MatTab[:, 5][mask_fit][mask_fit_magcut][mask_fit_mag_central])
                mag_fit_cen_faint = np.max(MatTab[:, 5][mask_fit][mask_fit_magcut][mask_fit_mag_central])
                ax2.vlines(mag_fit_cen_bright, ymin=fit_cen[0][0] + fit_cen[0][1] * mag_fit_cen_bright - 1,
                           ymax=fit_cen[0][0] + fit_cen[0][1] * mag_fit_cen_bright + 1, linewidth=0.7, colors='b')
                print(mag_fit_cen_faint, fit_cen[0][0] + fit_cen[0][1] * mag_fit_cen_faint - 1,
                      fit_cen[0][0] + fit_cen[0][1] * mag_fit_cen_faint + 1)
                ax2.vlines(mag_fit_cen_faint, ymin=fit_cen[0][0] + fit_cen[0][1] * mag_fit_cen_faint - 1,
                           ymax=fit_cen[0][0] + fit_cen[0][1] * mag_fit_cen_faint + 1, linewidth=0.7, colors='b')
                plt.savefig(OutDir + 'linfit_retry_3.' + self.ImgName + '.' + str(AstName) + '.png')
                if np.absolute(fit_cen[0][1] - 1) > 0.1:
                    print('!!! clipping error : slope = %.3f !!!' %fit_cen[0][1])
                    return np.array([-2])
                else:
                    mask = mask_cen
                    fit = fit_cen
                    Niter = Niter_cen
                    coe = fit[0]
                    np.diag(fit[1]) ** 0.5
            else:
                plt.savefig(OutDir + 'linfit_retry_2.' + self.ImgName + '.' + str(AstName) + '.png')
                mask = mask_cen
                fit = fit_cen
                Niter = Niter_cen
                coe = fit[0]
                np.diag(fit[1])**0.5

        mask_MM = np.absolute(MatTab[:, 3] - (coe[0] + coe[1]*MatTab[:, 5])) <= Mag_Diff_max

        ### Asteroid Check : Magnitude ###
        AstMag_exp = coe[0] + coe[1] * ObsTab['MAG_AUTO'][idx_AstInObsTab[0]]
        if np.absolute(AstMv - AstMag_exp) > 2:
            print('!!! asteroid %s rejected : large diff. btw AstMv(%s) & AstExp(%.3f) !!!' %(AstNum, AstMv, AstMag_exp))
            return np.array([-1])
        """
        boundary_obs = np.array([min(MatTab[mask_sc, 5]) - 1.5, max(MatTab[mask_sc, 5]) + 1.5])
        if (boundary_obs[0] > ObsTab['MAG_AUTO'][idx_AstInObsTab[0]]) or \
                (boundary_obs[1] < ObsTab['MAG_AUTO'][idx_AstInObsTab[0]]) :
            print('!!! asteroid %s rejected : Too Faint or Bright InstMag = %.3f (%.3f %.3f)'
                  %(AstNum, ObsTab['MAG_AUTO'][idx_AstInObsTab[0]], boundary_obs[0], boundary_obs[1]))
            print('[ Match Failure ] \n')
            return np.array([-1])
        """

        ### SAVE ###
        # name : [imgname] + [filter] + [mode] + [ast_id] + .cat
        # header : image name, exptime, mode, chip, chip number, (final) filter, ref_cat_type,
        #          ast_id, ellipticity, FLAG_SEx, ast_ra, ast_dec, ast_x, ast_y
        # table : ID, X, Y, mag_std, merr_std, mag_obs, merr_obs, ra, dec
        # append asteroid info
        if self.FILTER == 'B' or self.FILTER == 'V' or self.FILTER == 'R' or self.FILTER == 'I' :
            mask_clip = mask_FR * mask_MM * mask_safe
        elif self.FILTER == 'g' or self.FILTER == 'r' or self.FILTER == 'i' or self.FILTER == 'z' :
            mask_clip = mask_FR * mask_MM

        print('N_mat, N_mat_clip : %s -> %s' %(len(mask_clip), sum(mask_clip)))
        chipnames = ['master', 'M', 'K', 'N', 'T']
        if AstName_type is True:
            SaveMatTabName = self.ImgName + self.FILTER + '.' + self.MODE + '.' + AstName
            AstOutput = np.array([[-1, ObsTab['X_IMAGE'][idx_AstInObsTab[0]],
                                   ObsTab['Y_IMAGE'][idx_AstInObsTab[0]], -1, -1,
                                   ObsTab['MAG_AUTO'][idx_AstInObsTab[0]], ObsTab['MAGERR_AUTO'][idx_AstInObsTab[0]],
                                   AstRa_Obs[0], AstDec_Obs[0]]])
        elif int(AstNum) < 0:
            assert False
        else:
            SaveMatTabName = self.ImgName + self.FILTER + '.' + self.MODE + '.%s' % AstName
            AstOutput = np.array([[-int(AstNum), ObsTab['X_IMAGE'][idx_AstInObsTab[0]],
                                   ObsTab['Y_IMAGE'][idx_AstInObsTab[0]], -1, -1,
                                   ObsTab['MAG_AUTO'][idx_AstInObsTab[0]], ObsTab['MAGERR_AUTO'][idx_AstInObsTab[0]],
                                   AstRa_Obs[0], AstDec_Obs[0]]])

        # AstTab : [0] number, [1] name, [2] ra, [3] dec, [4] class(loc), [5] Mv,
        #          [6] err_pos, [7] body-to-center angular dist, [8] motion, [9] motion,
        #          [10] geocentric dist, [11] heliocentric dist, [12] Phase angle,
        #          [13] Solar elongation
        np.savetxt(OutDir + SaveMatTabName + '.cat', np.concatenate((AstOutput, MatTab[mask_clip]), axis=0),
                   fmt='%d\t%.4f\t%.4f\t%.4f\t%.6f\t%.4f\t%.4f\t%.4f\t%.4f',
                   header='image     %s\n' % self.ImgName   + 'EXPTIME   %s\n' % self.EXPTIME +
                          'MODE      %s\n' % self.MODE      + 'CHIP      %s\n' % chipnames[AstCHIPNUM] +
                          'CHIPNUM   %s\n' % AstCHIPNUM     + 'FILTER    %s\n' % self.FILTER +
                          'RefType   %s\n' % std_type       + 'ast_id    %s\n' % AstNum.strip() +
                          'ast_nam_S %s\n' % AstInfo[1]     + 'ast_class %s\n' % AstInfo[4] +
                          'dist_GC   %s\n' % AstInfo[10]    + 'dist_HC   %s\n' % AstInfo[11] +
                          'PhAngle   %s\n' % AstInfo[12]    + 'SolElong  %s\n' % AstInfo[13] +
                          'ellipti   %s\n' % ObsTab['ELLIPTICITY'][idx_AstInObsTab[0]] +
                          'FLAG_SEx  %s\n' % ObsTab['FLAGS'][idx_AstInObsTab[0]] +
                          'JD        %s\n' % self.JD +
                          '*ast_ra   %s\n' % AstRa + '*ast_dec  %s\n' % AstDec +
                          '**ast_X   %s\n' % AstX + '**ast_Y   %s\n' % AstY + ' * SkyBoT, ** from SkyBot Ra & Dec\n' +
                          'id\tx\t\ty\t\tmag_std\tme_std\t\tmag_obs\tme_obs\tra_obs\t\tdec_obs')

        if CheckImage is True:
            fig = plt.figure(figsize=(13, 13))
            fig.suptitle(SaveMatTabName)
            # image window
            ax1 = fig.add_subplot(2, 2, 1)
            ax1_w = 1000                         # width
            ax1_c = [np.round(AstX), np.round(AstY)]  # center
            ax1_b = [np.maximum(0, int(ax1_c[1]-ax1_w-10)), np.minimum(9232, int(ax1_c[1]+ax1_w+10)),
                     np.maximum(0, int(ax1_c[0]-ax1_w-10)), np.minimum(9216, int(ax1_c[0]+ax1_w+10))]   # boundary
            ax1_window = self.ImgSet[AstCHIPNUM][ax1_b[0] : ax1_b[1], ax1_b[2] : ax1_b[3]]
            # norm = simple_norm(self.ImgSet[AstCHIPNUM][:, AstAMPNUM * 1152:(AstAMPNUM + 1) * 1152], percent=99)
            norm = simple_norm(ax1_window[ax1_w+10-100:ax1_w+10+100,ax1_w+10-100:ax1_w+10+100], percent=99)
            ax1.imshow(ax1_window, norm=norm, origin='center')
            ax1.scatter(fig_ObsX-ax1_b[2], fig_ObsY-ax1_b[0], s=30, alpha=0.7, marker='o',
                        facecolor='None', edgecolor='mediumblue', label='source extract')
            ax1.scatter(StdTab['x']-ax1_b[2], StdTab['y']-ax1_b[0], s=10, marker='s',
                        alpha=0.7, facecolor='None', edgecolor='cyan', label='Pan DR2')
            ax1.scatter(AstOutput[0,1]-ax1_b[2], AstOutput[0,2]-ax1_b[0],
                        s=100, marker='o', facecolor='None', edgecolor='red', label='asteroid')
            ax1.set_xlim(10, 2*ax1_w)
            ax1.set_ylim(10, 2*ax1_w)
            ax1.legend(loc='upper left')

            # mag - merr
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.scatter(fig_ObsMag, fig_ObsMerr, s=4, c='k', alpha=0.8)
            ax2.axhline(ObsLim_Merr, linewidth=1, linestyle=':', c='k', label='MagErr Limit')
            ax2.set_xlabel('Inst. Magnitude')
            ax2.set_ylabel('Inst. Magnitude Error')
            fig_mask_err = fig_ObsMerr < 0.1
            ax2.set_xlim(np.min(fig_ObsMag)-0.5, np.max(fig_ObsMag[fig_mask_err])+1)
            ax2.set_ylim(-0.01, 0.21)

            # FLUX_RADIUS vs MAG
            mask_clip_not = np.logical_not(mask_clip)
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.set_xlabel('FLUX_RADIUS')
            ax3.set_ylabel('Instrumental Magnitude [mag]')
            ax3.scatter(ObsTab['FLUX_RADIUS'][idx_Obs_match][mask_clip], ObsTab['MAG_AUTO'][idx_Obs_match][mask_clip],
                        s=3, c='k')
            ax3.scatter(ObsTab['FLUX_RADIUS'][idx_Obs_match][mask_clip_not],
                        ObsTab['MAG_AUTO'][idx_Obs_match][mask_clip_not],
                        s=2, c='gray', alpha=0.5)
            ax3.axvline(FR_median, linewidth=1.2, linestyle='--', c='k')
            ax3.axvline(FR_median + FR_std, linewidth=1, linestyle=':', c='k')
            ax3.axvline(FR_median - FR_std, linewidth=1, linestyle=':', c='k')
            ax3.set_ylim(np.max(ObsTab['MAG_AUTO'][idx_Obs_match])+1, np.min(ObsTab['MAG_AUTO'][idx_Obs_match]-1))
            ax3.set_xlim(FR_median-0.5, FR_median+1.5)
            """
            ax3.set_xlim(np.minimum(np.maximum(np.min(ObsTab['FLUX_RADIUS'][idx_Obs_match])-0.3, FR_median-3*FR_std),
                                    FR_median-1*FR_std-0.3),
                         np.minimum(np.max(ObsTab['FLUX_RADIUS'][idx_Obs_match])+0.3, FR_median+5*FR_std))
            """
            # mag_inst - mag_std
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.errorbar(MatTab[mask_clip, 5], MatTab[mask_clip, 3], xerr=MatTab[mask_clip, 6], yerr=MatTab[mask_clip, 4],
                         fmt='o', markersize=2, c='k', ecolor='k', elinewidth=0.3, capsize=1, capthick=0.5)
            if sum(mask_clip_not) != 0:
                ax4.errorbar(MatTab[mask_clip_not, 5], MatTab[mask_clip_not, 3],
                             xerr=MatTab[mask_clip_not, 6], yerr=MatTab[mask_clip_not, 4], fmt='o', markersize=2,
                             c='gray', ecolor='gray', elinewidth=0.3, capsize=1, capthick=0.5, alpha=0.6)
            fig_fitx = np.linspace(min(MatTab[:, 5]) - 0.1, max(MatTab[:, 5]) + 0.1)
            ax4.plot(fig_fitx, coe[0] + coe[1] * fig_fitx, linestyle='--', c='k', linewidth=0.7,
                     label='%.4f+%.4fx' % (coe[0], coe[1]))
            ax4.vlines(AstOutput[0, 5], ymin=AstMag_exp - 1, ymax=AstMag_exp + 1, colors='red', linewidth=1.5,
                       linestyles='--', alpha=0.8, label='asteroid')
            #fig_boundary = sigma_boundary(fig_fitx, coe[0], coe[1], coesig[0], coesig[1], sigmaclip=10)
            #ax4.fill_between(fig_fitx, fig_boundary[0], fig_boundary[1], alpha=0.2)
            ax4.legend(loc='upper left')
            ax4.set_xlabel('Instrumental Magnitude [mag]')
            ax4.set_ylabel('Standard Magnitude [mag]')
            plt.savefig(OutDir+SaveMatTabName+'.png', dpi=dpi)
        plt.close('all')
        print('# %s matched\n[ Match END ] \n' % len(MatTab[mask_clip]))
        return np.array(len(MatTab[mask_clip]))


constraints = {'nDetections.gt': 4}
columns = """objID,raMean,decMean,nDetections,ng,nr,ni,nz,
gMeanKronMag,rMeanKronMag,iMeanKronMag,zMeanKronMag,
gMeanKronMagErr,rMeanKronMagErr,iMeanKronMagErr,zMeanKronMagErr,objInfoFlag""".split(',')
