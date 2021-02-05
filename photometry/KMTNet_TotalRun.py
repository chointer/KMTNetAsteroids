import numpy as np
import pandas as pd
import time
from KMTNet_1_match import *
from ModuleRefDown import *

constraints = {'nDetections.gt': 4}
columns = """objID,raMean,decMean,nDetections,ng,nr,ni,nz,
gMeanKronMag,rMeanKronMag,iMeanKronMag,zMeanKronMag,
gMeanKronMagErr,rMeanKronMagErr,iMeanKronMagErr,zMeanKronMagErr,objInfoFlag""".split(',')

t0 = time.time()


Datelist = ['20151215', '20151216', '20151217', '20151218', '20151219', 
            '20151220', '20151221', '20151222', '20151223', '20151224',
            '20160115', '20160116', '20160117', '20160118', '20160119',
            '20160122', '20161221', '20161222', '20161223', '20161224',
            '20161225', '20161227', '20161228', '20171211', '20171212',
            '20171213', '20171214', '20171215', '20171216', '20171217', '20171218']

FilterFlag = [True, True, True, True, True, 
              True, False, False, False, False,
              False, False, False, False, False,
              False, True, True, True, True,
              False, False, False, True, True,
              True, True, False, False, False, False]
# True: griz / False: BVRI


for dates in range(len(Datelist)):
    Date = Datelist[dates]
    if FilterFlag[dates] is True:
        FILTERS = ['g', 'r', 'i', 'z']
    else:
        FILTERS = ['B', 'V', 'R', 'I']

    #####################
    #  Phase 1 : Match  #
    #####################

    imagelist = np.genfromtxt('/data2/SHChoi/phot/obs' + Date + '/KMTNet_0_input/image' + Date + '.list', dtype=np.str)
    ImgDir = '/data2/SHChoi/data/' + Date + '/WCSCOR/'
    PhoDir = '/data2/SHChoi/data/' + Date + '/PHOT/'
    AstDir = '/data2/SHChoi/phot/obs' + Date + '/KMTNet_SkyBoT/'
    RefDir = '/data2/SHChoi/phot/obs' + Date + '/KMTNet_RefCat/'
    OutDir = '/data2/SHChoi/phot/obs' + Date + '/KMTNet_1_match/'
    AtlDir = '/data2/SHChoi/ATLAS_SkyMapper'
    PhotModeSaveDir = '/data2/SHChoi/phot/obs' + Date + '/KMTNet_0_input/'

    N_img = len(imagelist)

    MODECHECKLIST = ['startpoint']
    MODE_pre = ['startpoint']
    for i in range(N_img):
        print('[%s/%s]' % (i + 1, N_img), imagelist[i])
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


    MODECHECKLIST = ['startpoint']
    MODE_pre = ['startpoint']
    LowNumMatchList = []    # date img mode astnum N_match

    for i in range(N_img):
        #if imagelist[i] != 'kmtc.20151221.043786': continue
        #if np.logical_not(imagelist[i] == 'kmtc.20151216.042530') : continue
        #if i != 46 : continue
        # Make a Match Class
        proj = match(imagelist[i], ImgDir, PhoDir, np.array(MODECHECKLIST), MODE_pre)

        # Avoid MODE Duplication
        if proj.MODE != MODE_pre: MODECHECKLIST.append(proj.MODE)
        MODE_pre = np.copy(proj.MODE)

        # Getting Asteroid Table from SkyBoT
        proj.GetAstTab(proj.ImgName+'.asttab', AstDir)
        N = len(proj.AstTab)

        # Run Match for Each Asteroid
        for j in range(N):
            #if proj.AstTab[j,0] != ' 160114' : continue
            #if j < 4 & j > 1 : continue
            print('image %s/%s  asteroid %s/%s' %(i+1, N_img, j+1, N))
            process = proj.Match(proj.AstTab[j], RefDir, RefRad=0.3,
                                 StdLim_Merr=0.1, ObsLim_Merr=0.1, IDist=5, IDist_star=5, OutDir=OutDir,
                                 FLUX_RADIUS_SIG_max=0.5, FLUX_RADIUS_SIG_min=0.2, Mag_Diff_max=1, quadfitTF=False, dpi=80)
            if (process >= 0) & (process < 34):
                LowNumMatchList.append(np.array([Date, imagelist[i], proj.MODE, proj.AstTab[j][0], np.str(int(process))], dtype=np.str))


    #assert False
    # Save Mode Set
    np.savetxt(PhotModeSaveDir + 'obs' + Date + '.list', np.array(MODECHECKLIST)[1:], fmt='%s')
    # Save LowNumMatchListi
    if len(LowNumMatchList) != 0:
        np.savetxt(PhotModeSaveDir + 'LowNumMatchList' + Date + '.list', np.array(LowNumMatchList, dtype=np.str), fmt='%s\t%s\t%s\t%s\t%s')

    #"""

    #####################
    #  Phase 2 : Std.   #
    #####################

    from KMTNet_2_stdzation import *
    #TarAstTabName = 'asteroids_' + Date + '_' + FILTERS[0] + FILTERS[1] + FILTERS[2] + FILTERS[3]
    #TarAstTabDir = '/data2/SHChoi/phot/obs' + Date + '/KMTNet_0_input/'
    MatTabDir = '/data2/SHChoi/phot/obs' + Date + '/KMTNet_1_match/'
    SaveDir = '/data2/SHChoi/phot/obs' + Date + '/KMTNet_2_stdzation/'

    # !!! This Part Should be Modified for independent running from Targeting !!!
    #_, _, TarAstMode = GetTargetAstTab(TarAstTabName, TarAstTabDir)
    PhotMode = np.genfromtxt(PhotModeSaveDir + 'obs' + Date + '.list', dtype=np.str)
    TotalAsteroids = 0
    FLAG_AstPhotSet = False

    for i in range(len(PhotMode)):                                                # for each Mode
        # Make a Std. Class & Ast. Table corresponding to the Image Set
        project = asteroid_standardization(PhotMode[i], FILTERS, Date)            # class creation.
        AstNumTab = project.GetAstNumTab(MatTabDir)                                 # Get Asteroid Number List
        if AstNumTab[0] == 'FailureFlag': continue
        #if project.MODE != 'S00084' : continue

        # Run Std. for Each Asteroid
        for j in range(len(AstNumTab)):                                             # for each asteroid (each AstNum)
            print(' === === === Asteroid %s === === ===' %AstNumTab[j])
            Flag_GetMatTabSet = project.GetMatTabSet(AstNumTab[j], MatTabDir)       # Get Matched Table Set
            if Flag_GetMatTabSet == -1: continue
            # Standardization
            Flag_StdAsteroid = \
                project.StdAsteroid(sig_err=[1, 1, 1, 1], sig_eq1=[3, 3, 3, 3], sig_eq2=[3, 3, 3, 3],
                                    maxiter=[2, 2], fit_losstype=['basic', 'basic'], fit_cliptype=['chi', 'chi'],
                                    fit_clip_err=False, Mconsist=2)
            if Flag_StdAsteroid == -1: continue

            # Store Data
            project.SaveAst()                                                       # Save AstPhot
            project.SaveCoeff()                                                     # Save Coeffs (MODE, AstNum, coeffs)
            project.SaveErr()

            # Draw Figures
            project.Figure_Fit(save=True, dir_save=SaveDir, AddSaveName='')         # fitting figure
            project.Figure_Std(save=True, dir_save=SaveDir, AddSaveName='')         # std_dif figure

            # Count Asteroid Number
            TotalAsteroids += 1
            print('')

        # Store Data to the Higher Arrays
        if len(project.AstPhotSet) == 0:
            continue
        elif FLAG_AstPhotSet is False:
            AstPhotSet_tot = np.array(project.AstPhotSet)
            TabCoe = np.array(project.CoeffSet)
            TabErr = np.array(project.AstErrSet)
            FLAG_AstPhotSet = True
        else:
            AstPhotSet_tot = np.concatenate((AstPhotSet_tot, np.array(project.AstPhotSet)))
            TabCoe = np.concatenate((TabCoe, np.array(project.CoeffSet)))
            TabErr = np.concatenate((TabErr, np.array(project.AstErrSet)))

    print('Total Asteroids : %s' %TotalAsteroids)




    #####################
    #  Phase 3 : Save   #
    #####################
    # [00] MODE, [01] AstNum, [02] F1, [03] F1e, [04] F2, [05] F2e, [06] F3, [07] F3e, [08] F4, [09] F4e,
    # [10] FLAG, [11] CHIPNUM, [12] AMPNUM

    # Save Asteroids Data Table
    AstPhotSet_pd = pd.DataFrame(AstPhotSet_tot)
    AstPhotSet_pd.to_csv(SaveDir + 'AstPhot' + Date + '.dat', sep='\t', index=False, header=False)

    # Draw the Coeff. Histogram
    figure_coeff(TabCoe, FILTERS, 'test', bins_gap=0.025, save=True, dir_save=SaveDir, dpi=150, AddSaveName='')

    # Save Std. Coeff. Data Table
    CoeTab_pd = pd.DataFrame(TabCoe)
    CoeTab_pd.to_csv(SaveDir + 'Coeff' + Date + '.dat', sep='\t', index=False, header=False)

    # Save Error Data Table
    ErrTab_pd = pd.DataFrame(TabErr)
    ErrTab_pd.to_csv(SaveDir + 'Error' + Date + '.dat', sep='\t', index=False, header=False)

    print('\n\ndone')
    print(time.time()-t0)
