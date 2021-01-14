# 20200626  asteroid_standardization.saveast
#           -> output chip & amp number are changed from the first values to the mean values
# 20200702  asteroid_standardizaiton
#           -> add reftype, ellipticity, flag_sex to output
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits as fi
from astropy.visualization import simple_norm
import os, sys
sys.path.insert(0, '/data2/SHChoi/phot/python_script/MyModule')
from standardization import standardization_fit, sigma_boundary
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import time
import pandas as pd

def err_Add(errors):
    errors = np.array(errors)
    return (np.sum(errors**2, axis=0))**0.5

def err_Mul(elements, errors):
    x = np.prod(elements, axis=0)
    elements = np.array(elements)
    errors = np.array(errors)
    xerr = x * (np.sum((errors/elements)**2, axis=0))**0.5
    return xerr

def HeadReader(PhotTabDirName):
    dic = {}
    with open(PhotTabDirName, 'r') as f:
        for ii in range(15):
            line = f.readline().split()
            dic[line[1]] = line[2]
    return dic

def GetTargetAstTab(TarAstTabName, TarAstTabDir):
    date, num, mode = np.genfromtxt(TarAstTabDir + TarAstTabName, dtype=np.str).T
    mode = np.array(np.char.split(mode, sep='-').tolist())[:, 0]
    return date, num, mode

def figure_coeff(CoeTab, FILTERS, title='', bins_gap=0.2, save=False, dir_save='', AddSaveName='', dpi=150) :
    # [00] MODE, [01] AstNum,
    # [02] F1C1, [03] F1C1e, [04] F1C2, [05] F1C2e, [06] F1C3, [07] F1C3e, [08] F1C4, [09] F1C4e
    # [10] F2C1, [11] F2C1e, [12] F2C2, [13] F2C2e, [14] F2C3, [15] F2C3e, [16] F2C4, [17] F2C4e
    # [18] F3C1, [19] F3C1e, [20] F3C2, [21] F3C2e, [22] F3C3, [23] F3C3e, [24] F3C4, [25] F3C4e
    # [26] F4C1, [27] F4C1e, [28] F4C2, [29] F4C2e, [30] F4C3, [31] F4C3e, [32] F4C4, [33] F4C4e
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('Coeff. Distr. (C1, C3) '+title)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.10, right=0.95)
    axs = []
    # C1 histograms
    ax_range = []; ax_center = []; ax_height = []
    for k in range(4) :
        axs.append(fig.add_subplot(2, 4, k+1))
        if k == 0 : axs[k].set_ylabel('C1 [#]')
        axs[k].set_title(FILTERS[k])
        C1_temp = CoeTab[:, (k * 8) + 2].astype(float)
        C1e_temp = CoeTab[:, (k * 8) + 3].astype(float)
        bins_range = np.array([bins_gap * (np.min(C1_temp)//bins_gap),
                               bins_gap * (1 + (np.max(C1_temp)//bins_gap))])
        bins_N = int(0.01 + 1 + (np.max(C1_temp)//bins_gap) - (np.min(C1_temp)//bins_gap))
        n, bins, _ = axs[k].hist(C1_temp, bins_N, bins_range,
                                 label='%.3f $\pm$ %.4f' %(np.mean(C1_temp), np.std(C1_temp)))
        #axs[k].errorbar(C1_temp, np.linspace(0.2, 0.8, len(C1_temp)), xerr=C1e_temp, fmt='o', markersize=2,
        #                ecolor='k', elinewidth=1, capsize=1.2, capthick=0.8, c='k')
        axs[k].legend()
        ax_height.append(np.max(n))
        ax_center.append(np.mean(bins_range))
        ax_range.append(bins_range[1]-bins_range[0])
    for k in range(4) :
        axs[k].set_ylim(0, np.max(ax_height)+2)
        axs[k].set_xlim(-bins_gap + ax_center[k] - np.max(ax_range) / 2,
                        bins_gap + ax_center[k] + np.max(ax_range) / 2)
    # C2 histograms
    ax_range = []; ax_center = []; ax_height = []
    for k in range(4):
        axs.append(fig.add_subplot(2, 4, k + 5))
        if k == 0: axs[k + 4].set_ylabel('C3 [#]')
        C3_temp = CoeTab[:, (k * 8) + 6].astype(float)
        C3e_temp = CoeTab[:, (k * 8) + 7].astype(float)
        bins_range = np.array([bins_gap * (np.min(C3_temp) // bins_gap),
                               bins_gap * (1 + (np.max(C3_temp) // bins_gap))])
        bins_N = int(0.01 + 1 + (np.max(C3_temp) // bins_gap) - (np.min(C3_temp) // bins_gap))
        n, bins, _ = axs[k+4].hist(C3_temp, bins_N, bins_range,
                                 label='%.3f $\pm$ %.4f' % (np.mean(C3_temp), np.std(C3_temp)))
        #axs[k+4].errorbar(C3_temp, np.linspace(0.2, 0.8, len(C3_temp)), xerr=C3e_temp, fmt='o', markersize=2,
        #                ecolor='k', elinewidth=1, capsize=1.2, capthick=0.8, c='k')
        axs[k+4].legend()
        ax_height.append(np.max(n))
        ax_center.append(np.mean(bins_range))
        ax_range.append(bins_range[1] - bins_range[0])
    for k in range(4):
        axs[k+4].set_ylim(0, np.max(ax_height) + 2)
        axs[k+4].set_xlim(-bins_gap + ax_center[k] - np.max(ax_range) / 2,
                          bins_gap + ax_center[k] + np.max(ax_range) / 2)

    if save is True:
        plt.savefig(dir_save+'CoeffHist' + AddSaveName + '.png', dpi=dpi)
    plt.close('all')

class asteroid_standardization:
    def __init__(self, MODE, FILTERS, Date):
        self.MODE = MODE
        self.Date = Date
        #if Date =='20151216' and self.MODE == 'S06532': self.MODE = 'S06352'
        self.FILTERS = FILTERS
        print('\n === === === Standardization %s === === ===' % self.MODE)
        # coefficient storage
        self.CoeffSet = []
        self.AstPhotSet = []
        self.FlagTargetTab = False

    def GetAstNumTab(self, MatTabDir):
        os.system('ls %s*%s.%s.*.cat > temp.lis' % (MatTabDir, self.FILTERS[1], self.MODE))
        try:
            AstNumTab = np.array(np.char.split(np.genfromtxt('temp.lis', dtype=str), sep='.').tolist())[:, 4]
        except IndexError:
            try:
                if len(np.genfromtxt('temp.lis', dtype=str)) == 0:
                    print('GetAstNumTab - No Asteroids. There might be a Problem. [PASS this Set]')
                    return ['FailureFlag']
            except TypeError: pass
            print('(GetAstNumTab - only one asteroid might be found)')
            print('ls %s*%s.%s.*.cat' %(MatTabDir, self.FILTERS[1], self.MODE))
            print('/media/astro/KMTNet/obs20151215/KMTNet_1_match')
            AstNumTab = np.array([np.array(np.char.split(np.genfromtxt('temp.lis', dtype=str), sep='.').tolist())[4]])
        print(AstNumTab)
        return AstNumTab

    def GetMatTabSet(self, AstNum, MatTabDir, fit_Nlim_min=34):
        # return 4 filter matched-tables
        # if the table set is "not perfect" for some reason (eg. there are only 3 tables), it returns -1
        ### get matched tables ###
        print(' --- GetMatTabSet ---')
        tables = []
        self.tableheaders = []
        os.system('ls %s*.%s.%s.cat > temp.lis' % (MatTabDir, self.MODE, AstNum))
        TableNames = np.genfromtxt('temp.lis', dtype=str)
        try: TableNames_part = np.array(np.char.split(TableNames, sep='.').tolist())[:, 2]
        except IndexError:
            print('GetMatTabSet - IndexError : only one table might be found =', TableNames)
            return -1
        if len(TableNames) < 4:
            print('GetMatTabSet - Not Perfect Table Set')
            print(TableNames)
            return -1

        TableNames_filter = np.zeros(4, dtype='str')
        for k in range(4):
            TableNames_filter[k] = TableNames_part[k][-1]
        for k in range(4):
            idx_filter = np.where(TableNames_filter == self.FILTERS[k])
            if len(idx_filter[0]) != 1:
                print('GetMatTabSet - Not Perfect Table Set : ', TableNames_filter)
                return -1
            tables.append(np.genfromtxt(TableNames[idx_filter][0]))
            self.tableheaders.append(HeadReader(TableNames[idx_filter][0]))
        print('Tabs\t', tables[0].shape, tables[1].shape, tables[2].shape, tables[3].shape)

        ### match tables ###
        # most important points of this algorithm are
        # (1) that all tables used the same PanSTARRS queried data -> newID is shared
        # (2) that newIDs in tables are already ordered in ascending power. -> masking is enough to match
        maskset = []
        for k in range(3): maskset.append(np.in1d(tables[0][:, 0], tables[k+1][:, 0]))
        mask0 = maskset[0] * maskset[1] * maskset[2]
        tables_output = []
        tables_output.append(tables[0][mask0])
        for k in range(3):
            mask_temp = np.in1d(tables[k+1][:, 0], tables_output[0][:, 0])
            tables_output.append(tables[k+1][mask_temp])
        print('Tabs-mat', tables_output[0].shape, tables_output[1].shape, tables_output[2].shape, tables_output[3].shape)
        if sum(mask0) < fit_Nlim_min :
            print('GetMatTabSet - too few matched stars (%s < %s)   * matching failure?' %(sum(mask0), fit_Nlim_min))
            return -1
        self.AstNum = AstNum
        self.MatTabSet = tables_output
        self.FLAG = 0
        return 0

    def StdAsteroid(self, sig_err=[1, 1, 1, 1], sig_eq1=[5, 5, 5, 5], sig_eq2=[2, 2, 2, 2],
                   maxiter=[10, 10], fit_losstype=['basic', 'basic'], fit_cliptype=['fit', 'fit'],
                    fit_clip_err=True, Mconsist=2, fit_Nlim_min=34):
        """
        fittype=['basic', 'basic']
        !!! Recent tables from GetMatTabSet are used !!!
        :output: self.FitSet, self.AstF_cal, self.TabF_cal
        """
        TabF1, TabF2, TabF3, TabF4 = self.MatTabSet
        AstF1 = TabF1[0]; AstF2 = TabF2[0]; AstF3 = TabF3[0]; AstF4 = TabF4[0]
        TabF1 = TabF1[1:]; TabF2 = TabF2[1:]; TabF3 = TabF3[1:]; TabF4 = TabF4[1:]
        # [0] newID, [1] X_Image, [2] Y_Image,
        # [3] MagStd, [4] MerrStd, [5] MagIns, [6] MerrIns, [7] ra_obs, [8] dec_obs
        # standardization_fit(m_ins, e_ins, m_std, e_std, r_ins, re_ins, r_std, re_std, b_ins, be_ins, b_std, be_std)
        ### FLAGING ###
        # FLAG_edge
        if AstF1[1] < 100 or AstF1[1] > 9116 or AstF1[2] < 100 or AstF1[2] > 9132 or \
                AstF2[1] < 100 or AstF2[1] > 9116 or AstF2[2] < 100 or AstF2[2] > 9132 or \
                AstF3[1] < 100 or AstF3[1] > 9116 or AstF3[2] < 100 or AstF3[2] > 9132 or \
                AstF4[1] < 100 or AstF4[1] > 9116 or AstF4[2] < 100 or AstF4[2] > 9132: self.FLAG += 1
        # FLAG_extrapolation_bright
        if AstF1[5] < np.min(TabF1[:, 5]) or AstF2[5] < np.min(TabF2[:, 5]) or \
                AstF3[5] < np.min(TabF3[:, 5]) or AstF4[5] < np.max(TabF4[:, 5]): self.FLAG += 2
        # FALG_extrapolation_faint
        if AstF1[5] > np.max(TabF1[:, 5]) or AstF2[5] > np.min(TabF2[:, 5]) or \
                AstF3[5] > np.max(TabF3[:, 5]) or AstF4[5] > np.max(TabF4[:, 5]): self.FLAG += 4
        # 9 mag is the brightest in 60s EXPTIME img.
        #if AstF1[5] > 11 : mag Flaging

        sigset = np.array([sig_eq1, sig_eq2, sig_err]).T
        ### Fitting ###
        # FitF1 : g, g-r [F1, F1-F2]
        FitF1 = standardization_fit(TabF1[:, 5], TabF1[:, 6], TabF1[:, 3], TabF1[:, 4],
                                    TabF1[:, 5], TabF1[:, 6], TabF1[:, 3], TabF1[:, 4],
                                    TabF2[:, 5], TabF2[:, 6], TabF2[:, 3], TabF2[:, 4])
        FitF1.fit(coe1=np.array([np.mean(TabF1[:, 3] - TabF1[:, 5]), 0.15]), coe2=np.array([0.1, 1]),
                  sigmaclip=sigset[0], maxiter=maxiter, losstype=fit_losstype, cliptype=fit_cliptype,
                  clip_err=fit_clip_err, Nlim_min=fit_Nlim_min)
        # FitF2 : r, g-r [F2, F1-F2]
        FitF2 = standardization_fit(TabF2[:, 5], TabF2[:, 6], TabF2[:, 3], TabF2[:, 4],
                                    TabF1[:, 5], TabF1[:, 6], TabF1[:, 3], TabF1[:, 4],
                                    TabF2[:, 5], TabF2[:, 6], TabF2[:, 3], TabF2[:, 4])
        FitF2.fit(coe1=np.array([np.mean(TabF2[:, 3] - TabF2[:, 5]), 0.1]), coe2=np.array([0.1, 1]),
                   sigmaclip=sigset[1], maxiter=maxiter, losstype=fit_losstype, cliptype=fit_cliptype,
                  clip_err=fit_clip_err, Nlim_min=fit_Nlim_min)
        # FitF3 : i, r-i [F3, F2-F3]
        FitF3 = standardization_fit(TabF3[:, 5], TabF3[:, 6], TabF3[:, 3], TabF3[:, 4],
                                    TabF2[:, 5], TabF2[:, 6], TabF2[:, 3], TabF2[:, 4],
                                    TabF3[:, 5], TabF3[:, 6], TabF3[:, 3], TabF3[:, 4])
        FitF3.fit(coe1=np.array([np.mean(TabF3[:, 3] - TabF3[:, 5]), 0.1]), coe2=np.array([0.1, 1]),
                  sigmaclip=sigset[2], maxiter=maxiter, losstype=fit_losstype, cliptype=fit_cliptype,
        clip_err=fit_clip_err, Nlim_min=fit_Nlim_min)
        # FitF4 : z, i-z [F4, F3-F4]
        FitF4 = standardization_fit(TabF4[:, 5], TabF4[:, 6], TabF4[:, 3], TabF4[:, 4],
                                    TabF3[:, 5], TabF3[:, 6], TabF3[:, 3], TabF3[:, 4],
                                    TabF4[:, 5], TabF4[:, 6], TabF4[:, 3], TabF4[:, 4])
        FitF4.fit(coe1=np.array([np.mean(TabF4[:, 3] - TabF4[:, 5]), 0.1]), coe2=np.array([0.1, 1]),
                  sigmaclip=sigset[3], maxiter=maxiter, losstype=fit_losstype, cliptype=fit_cliptype,
                  clip_err=fit_clip_err, Nlim_min=fit_Nlim_min)
        if (FitF1.FitFailureFlag == True) or (FitF2.FitFailureFlag == True) \
                or (FitF3.FitFailureFlag == True) or (FitF4.FitFailureFlag == True):
            return -1
        self.sig_eq1 = sig_eq1
        self.FitSet = [FitF1, FitF2, FitF3, FitF4]

        ### Asteroid ###
        # standardization(self, m_ins, e_ins, c_ins, ce_ins)
        self.AstF1_cal = FitF1.standardization(AstF1[5], AstF1[6], AstF1[5] - AstF2[5], err_Add([AstF1[6], AstF2[6]]))
        self.AstF2_cal = FitF2.standardization(AstF2[5], AstF2[6], AstF1[5] - AstF2[5], err_Add([AstF1[6], AstF2[6]]))
        self.AstF3_cal = FitF3.standardization(AstF3[5], AstF3[6], AstF2[5] - AstF3[5], err_Add([AstF2[6], AstF3[6]]))
        self.AstF4_cal = FitF4.standardization(AstF4[5], AstF4[6], AstF3[5] - AstF4[5], err_Add([AstF3[6], AstF4[6]]))
        print('AstMag\t%.3f\t%.3f\t%.3f\t%.3f' %(self.AstF1_cal[0], self.AstF2_cal[0],
                                                 self.AstF3_cal[0], self.AstF4_cal[0]))
        print('AstMerr\t%.4f\t%.4f\t%.4f\t%.4f' %(self.AstF1_cal[1], self.AstF2_cal[1],
                                                  self.AstF3_cal[1], self.AstF4_cal[1]))

        ### Stars ###
        self.TabF1_cal = FitF1.standardization(TabF1[:, 5], TabF1[:, 6], TabF1[:, 5] - TabF2[:, 5],
                                               err_Add([TabF1[:, 6], TabF2[:, 6]]))
        self.TabF2_cal = FitF2.standardization(TabF2[:, 5], TabF2[:, 6], TabF1[:, 5] - TabF2[:, 5],
                                               err_Add([TabF1[:, 6], TabF2[:, 6]]))
        self.TabF3_cal = FitF3.standardization(TabF3[:, 5], TabF3[:, 6], TabF2[:, 5] - TabF3[:, 5],
                                               err_Add([TabF2[:, 6], TabF3[:, 6]]))
        self.TabF4_cal = FitF4.standardization(TabF4[:, 5], TabF4[:, 6], TabF3[:, 5] - TabF4[:, 5],
                                               err_Add([TabF3[:, 6], TabF4[:, 6]]))

        # consistency
        mag_diff = np.max([self.AstF1_cal[0], self.AstF2_cal[0], self.AstF3_cal[0], self.AstF4_cal[0]])-\
                np.min([self.AstF1_cal[0], self.AstF2_cal[0], self.AstF3_cal[0], self.AstF4_cal[0]])
        if mag_diff > Mconsist:
            print('StdAsteroid - consistency error %.3f' % mag_diff)
            return -1
        else: return 0

    def SaveCoeff(self):
        # [00] MODE, [01] AstNum,
        # [02] F1C1, [03] F1C1e, [04] F1C2, [05] F1C2e, [06] F1C3, [07] F1C3e, [08] F1C4, [09] F1C4e
        # [10] F2C1, [11] F2C1e, [12] F2C2, [13] F2C2e, [14] F2C3, [15] F2C3e, [16] F2C4, [17] F2C4e
        # [18] F3C1, [19] F3C1e, [20] F3C2, [21] F3C2e, [22] F3C3, [23] F3C3e, [24] F3C4, [25] F3C4e
        # [26] F4C1, [27] F4C1e, [28] F4C2, [29] F4C2e, [30] F4C3, [31] F4C3e, [32] F4C4, [33] F4C4e
        self.CoeffSet.append([self.MODE, self.AstNum,
                              self.FitSet[0].C1, self.FitSet[0].C1err, self.FitSet[0].C2, self.FitSet[0].C2err,
                              self.FitSet[0].C3, self.FitSet[0].C3err, self.FitSet[0].C4, self.FitSet[0].C4err,
                              self.FitSet[1].C1, self.FitSet[1].C1err, self.FitSet[1].C2, self.FitSet[1].C2err,
                              self.FitSet[1].C3, self.FitSet[1].C3err, self.FitSet[1].C4, self.FitSet[1].C4err,
                              self.FitSet[2].C1, self.FitSet[2].C1err, self.FitSet[2].C2, self.FitSet[2].C2err,
                              self.FitSet[2].C3, self.FitSet[2].C3err, self.FitSet[2].C4, self.FitSet[2].C4err,
                              self.FitSet[3].C1, self.FitSet[3].C1err, self.FitSet[3].C2, self.FitSet[3].C2err,
                              self.FitSet[3].C3, self.FitSet[3].C3err, self.FitSet[3].C4, self.FitSet[3].C4err])

    def SaveAst(self):
        # [0] Date, [1] MODE, [2] AstNum,
        # [3] F1, [4] F1e, [5] F2, [6] F2e, [7] F3, [8] F3e, [9] F4, [10] F4e,
        # [11] F1_FLAG_SEx, [12] F2_FLAG_SEx, [13] F3_FLAG_SEx, [14] F4_FLAG_SEx,
        # [15] FLAG, [16] CHIPNUM, [17] AMPNUM, [18] ellipticity, [19] ref_type, [20] observation_span
        CHIPNUM = (np.int(self.tableheaders[0]['CHIPNUM']) + np.int(self.tableheaders[1]['CHIPNUM']) +
                   np.int(self.tableheaders[2]['CHIPNUM']) + np.int(self.tableheaders[3]['CHIPNUM']))/4
        AMPNUM = (float(self.tableheaders[0]['**ast_X'])//1152 + float(self.tableheaders[1]['**ast_X'])//1152 +
                  float(self.tableheaders[2]['**ast_X'])//1152 + float(self.tableheaders[3]['**ast_X'])//1152) / 4
        ellip = (float(self.tableheaders[0]['ellipti']) + float(self.tableheaders[1]['ellipti']) +
                 float(self.tableheaders[2]['ellipti']) + float(self.tableheaders[3]['ellipti'])) / 4
        obsJDs = np.array([float(self.tableheaders[0]['JD']), float(self.tableheaders[1]['JD']),
                           float(self.tableheaders[2]['JD']), float(self.tableheaders[3]['JD'])])
        obsspan = np.max(obsJDs) - np.min(obsJDs)
        if self.tableheaders[0]['RefType'] == 'Pan': reftype = 0.
        elif self.tableheaders[0]['RefType'] == 'ATLAS': reftype = 1.
        else: assert False, 'SaveAst RefType %s' % self.tableheaders[0]['RefType']

        self.AstPhotSet.append([self.Date, self.MODE, self.AstNum,
                                self.AstF1_cal[0], self.AstF1_cal[1], self.AstF2_cal[0], self.AstF2_cal[1],
                                self.AstF3_cal[0], self.AstF3_cal[1], self.AstF4_cal[0], self.AstF4_cal[1],
                                float(self.tableheaders[0]['FLAG_SEx']), float(self.tableheaders[1]['FLAG_SEx']),
                                float(self.tableheaders[2]['FLAG_SEx']), float(self.tableheaders[3]['FLAG_SEx']),
                                self.FLAG, CHIPNUM, AMPNUM, round(ellip, 3), reftype, obsspan])

    def Figure_Fit(self, dpi=150, save=False, dir_save='', AddSaveName=''):
        TabF1, TabF2, TabF3, TabF4 = self.MatTabSet
        AstFSet = [0, TabF1[0], TabF2[0], TabF3[0], TabF4[0]]
        F1, F2, F3, F4 = self.FILTERS
        eq1_Xlabels = ['%s_{std} - %s_{std}' % (F1, F2), '%s_{std} - %s_{std}' % (F1, F2),
                       '%s_{std} - %s_{std}' % (F2, F3), '%s_{std} - %s_{std}' % (F3, F4)]
        eq1_Ylabels = ['%s_{std} - %s_{ins}' % (F1, F1), '%s_{std} - %s_{ins}' % (F2, F2),
                       '%s_{std} - %s_{ins}' % (F3, F3), '%s_{std} - %s_{ins}' % (F4, F4)]
        eq2_Xlabels = ['%s_{ins} - %s_{ins}' % (F1, F2), '%s_{ins} - %s_{ins}' % (F1, F2),
                       '%s_{ins} - %s_{ins}' % (F2, F3), '%s_{ins} - %s_{ins}' % (F3, F4)]
        eq2_Ylabels = ['%s_{std} - %s_{std}' % (F1, F2), '%s_{std} - %s_{std}' % (F1, F2),
                       '%s_{std} - %s_{std}' % (F2, F3), '%s_{std} - %s_{std}' % (F3, F4)]
        fig = plt.figure(figsize=(19, 9))
        fig.subplots_adjust(top=0.90, bottom=0.10, left=0.05, right=0.97)
        fig.suptitle('Std. Fit. %s #%s' %(self.MODE, self.AstNum))
        axs = []
        for k in range(8): axs.append(fig.add_subplot(2, 4, k+1))
        for k in range(4):
            axs[k].set_title(self.FILTERS[k])
            self.FitSet[k].plot_fitting(axs[k], axs[k+4], eq1_Xlabel=eq1_Xlabels[k], eq1_Ylabel=eq1_Ylabels[k],
                                        eq2_Xlabel=eq2_Xlabels[k], eq2_Ylabel=eq2_Ylabels[k])
        # ast g #
        axs[0].errorbar(self.AstF1_cal[0] - self.AstF2_cal[0], self.AstF1_cal[0] - AstFSet[1][5],
                        xerr=err_Add([self.AstF1_cal[1], self.AstF2_cal[1]]),
                        yerr=err_Add([self.AstF1_cal[1], AstFSet[1][6]]), fmt='o', markersize=4, ecolor='r',
                        elinewidth=1, capsize=1.2, capthick=0.8, c='r')
        axs[4].errorbar(AstFSet[1][5] - AstFSet[2][5], self.AstF1_cal[0] - self.AstF2_cal[0],
                        xerr=err_Add([AstFSet[1][6], AstFSet[2][6]]),
                        yerr=err_Add([self.AstF1_cal[1], self.AstF2_cal[1]]), fmt='o', markersize=4, ecolor='r',
                        elinewidth=1, capsize=1.2, capthick=0.8, c='r')
        # ast r #
        axs[1].errorbar(self.AstF1_cal[0] - self.AstF2_cal[0], self.AstF2_cal[0] - AstFSet[2][5],
                        xerr=err_Add([self.AstF1_cal[1], self.AstF2_cal[1]]),
                        yerr=err_Add([self.AstF2_cal[1], AstFSet[2][6]]), fmt='o', markersize=4, ecolor='r',
                        elinewidth=1, capsize=1.2, capthick=0.8, c='r')
        axs[5].errorbar(AstFSet[1][5] - AstFSet[2][5], self.AstF1_cal[0] - self.AstF2_cal[0],
                        xerr=err_Add([AstFSet[1][6], AstFSet[2][6]]),
                        yerr=err_Add([self.AstF1_cal[1], self.AstF2_cal[1]]), fmt='o', markersize=4, ecolor='r',
                        elinewidth=1, capsize=1.2, capthick=0.8, c='r')
        # ast i #
        axs[2].errorbar(self.AstF2_cal[0] - self.AstF3_cal[0], self.AstF3_cal[0] - AstFSet[3][5],
                        xerr=err_Add([self.AstF2_cal[1], self.AstF3_cal[1]]),
                        yerr=err_Add([self.AstF3_cal[1], AstFSet[3][6]]), fmt='o', markersize=4, ecolor='r',
                        elinewidth=1, capsize=1.2, capthick=0.8, c='r')
        axs[6].errorbar(AstFSet[2][5] - AstFSet[3][5], self.AstF2_cal[0] - self.AstF3_cal[0],
                        xerr=err_Add([AstFSet[2][6], AstFSet[3][6]]),
                        yerr=err_Add([self.AstF2_cal[1], self.AstF3_cal[1]]), fmt='o', markersize=4, ecolor='r',
                        elinewidth=1, capsize=1.2, capthick=0.8, c='r')
        # ast z #
        axs[3].errorbar(self.AstF3_cal[0] - self.AstF4_cal[0], self.AstF4_cal[0] - AstFSet[4][5],
                        xerr=err_Add([self.AstF3_cal[1], self.AstF4_cal[1]]),
                        yerr=err_Add([self.AstF4_cal[1], AstFSet[4][6]]), fmt='o', markersize=4, ecolor='r',
                        elinewidth=1, capsize=1.2, capthick=0.8, c='r')
        axs[7].errorbar(AstFSet[3][5] - AstFSet[4][5], self.AstF3_cal[0] - self.AstF4_cal[0],
                        xerr=err_Add([AstFSet[3][6], AstFSet[4][6]]),
                        yerr=err_Add([self.AstF3_cal[1], self.AstF4_cal[1]]), fmt='o', markersize=4, ecolor='r',
                        elinewidth=1, capsize=1.2, capthick=0.8, c='r')
        #plt.show()
        if save is True:
            fig.savefig(dir_save + 'StdFit' + AddSaveName + '_%s_%s.png' %(self.MODE, self.AstNum), dpi=dpi)
            plt.close('all')


    def Figure_Std_subplots(self, axes, x, y, xlabel, ylabel, mask, ylim=0.3):
        # member_in, member_out, outlier_in, outlier_out
        mask_m_in = (mask == True) * (np.absolute(y) < ylim)
        mask_m_out_u = (mask == True) * (y >= ylim)
        mask_m_out_l = (mask == True) * (y <= -ylim)
        mask_o_in = (mask == False) * (np.absolute(y) < ylim)
        mask_o_out_u = (mask == False) * (y >= ylim)
        mask_o_out_l = (mask == False) * (y <= -ylim)

        if sum(mask_m_in) != 0: axes.scatter(x[mask_m_in], y[mask_m_in], c='k', s=5, alpha=0.6)
        if sum(mask_m_out_u) != 0: axes.scatter(x[mask_m_out_u], np.zeros(sum(mask_m_out_u)) + ylim - 0.01, c='k', s=13,
                                                marker=10, alpha=0.6)
        if sum(mask_m_out_l) != 0: axes.scatter(x[mask_m_out_l], np.zeros(sum(mask_m_out_l)) - ylim + 0.01, c='k', s=13,
                                                marker=11, alpha=0.6)
        if sum(mask_o_in) != 0: axes.scatter(x[mask_o_in], y[mask_o_in], c='gray', s=5, alpha=0.6)
        if sum(mask_o_out_u) != 0: axes.scatter(x[mask_o_out_u], np.zeros(sum(mask_o_out_u)) + ylim - 0.01, c='gray',
                                                s=13, marker=10, alpha=0.6)
        if sum(mask_o_out_l) != 0: axes.scatter(x[mask_o_out_l], np.zeros(sum(mask_o_out_l)) - ylim + 0.01, c='gray',
                                                s=13, marker=11, alpha=0.6)

        # if sum(mask_outlier) != 0 :
        #    axes.scatter(x[mask_outlier], y[mask_outlier], c='grey', s=3, alpha=0.6)
        # axes.scatter(x[mask], y[mask], c='k', s=5)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.axhline(np.mean(y[mask]), linestyle='--', linewidth=1, c='k')
        axes.axhline(np.mean(y[mask]) - np.std(y[mask]), linestyle='--', linewidth=0.7, c='k')
        axes.axhline(np.mean(y[mask]) + np.std(y[mask]), linestyle='--', linewidth=0.7, c='k')
        axes.axhline(np.mean(y) - np.std(y), linestyle=':', linewidth=0.7, c='gray')
        axes.axhline(np.mean(y) + np.std(y), linestyle=':', linewidth=0.7, c='gray')
        axes.set_ylim(-ylim, ylim)
        axes.yaxis.set_major_locator(MultipleLocator(0.05))
        axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axes.yaxis.set_minor_locator(MultipleLocator(0.01))

    def Figure_Std(self, dpi=150, save=False, dir_save='', ylim=0.15, AddSaveName=''):
        TabF1, TabF2, TabF3, TabF4 = self.MatTabSet
        TabFSet = [0, TabF1[1:], TabF2[1:], TabF3[1:], TabF4[1:]]
        F1, F2, F3, F4 = self.FILTERS
        fig = plt.figure(figsize=(16, 8))
        fig.subplots_adjust(top=0.90, bottom=0.10, left=0.10, right=0.95, wspace=0.3, hspace=0.45)
        fig.suptitle('Std. Diff. %s #%s' %(self.MODE, self.AstNum))
        axs = []
        for k in range(7): axs.append(fig.add_subplot(4, 2, k+1))
        ### filter ###
        self.Figure_Std_subplots(axs[0], TabFSet[1][:, 3] - TabFSet[2][:, 3], self.TabF1_cal[0] - TabFSet[1][:, 3],
                                 '(%s-%s)$_{std}$' %(F1, F2), '$\Delta$%s' %F1, self.FitSet[0].mask, ylim=ylim)
        self.Figure_Std_subplots(axs[2], TabFSet[1][:, 3] - TabFSet[2][:, 3], self.TabF2_cal[0] - TabFSet[2][:, 3],
                                 '(%s-%s)$_{std}$' %(F1, F2), '$\Delta$%s' %F2, self.FitSet[1].mask, ylim=ylim)
        self.Figure_Std_subplots(axs[4], TabFSet[2][:, 3] - TabFSet[3][:, 3], self.TabF3_cal[0] - TabFSet[3][:, 3],
                                 '(%s-%s)$_{std}$' %(F2, F3), '$\Delta$%s' %F3, self.FitSet[2].mask, ylim=ylim)
        self.Figure_Std_subplots(axs[6], TabFSet[3][:, 3] - TabFSet[4][:, 3], self.TabF4_cal[0] - TabFSet[4][:, 3],
                                 '(%s-%s)$_{std}$' %(F3, F4), '$\Delta$%s' %F4, self.FitSet[3].mask, ylim=ylim)
        ### color ###
        self.Figure_Std_subplots(axs[1], TabFSet[1][:, 3] - TabFSet[2][:, 3],
                                 (self.TabF1_cal[0] - self.TabF2_cal[0]) - (TabFSet[1][:, 3] - TabFSet[2][:, 3]),
                                 '(%s-%s)$_{std}$' %(F1, F2), '$\Delta$(%s-%s)' %(F1, F2), self.FitSet[0].mask, ylim=ylim)
        self.Figure_Std_subplots(axs[3], TabFSet[2][:, 3] - TabFSet[3][:, 3],
                                 (self.TabF2_cal[0] - self.TabF3_cal[0]) - (TabFSet[2][:, 3] - TabFSet[3][:, 3]),
                                 '(%s-%s)$_{std}$' %(F2, F3), '$\Delta$(%s-%s)' %(F2, F3), self.FitSet[2].mask, ylim=ylim)
        self.Figure_Std_subplots(axs[5], TabFSet[3][:, 3] - TabFSet[4][:, 3],
                                 (self.TabF3_cal[0] - self.TabF4_cal[0]) - (TabFSet[3][:, 3] - TabFSet[4][:, 3]),
                                 '(%s-%s)$_{std}$' %(F3, F4), '$\Delta$(%s-%s)' %(F3, F4), self.FitSet[3].mask, ylim=ylim)
        if save==True :
            fig.savefig(dir_save + 'StdDiff' + AddSaveName + '_%s_%s.png' %(self.MODE, self.AstNum), dpi=dpi)
            plt.close('all')
        #plt.show()

    def Figure_chi_distribution(self, dpi=150, save=False, dir_save='', AddSaveName=''):
        F1, F2, F3, F4 = self.FILTERS
        eq1_Ylabels = ['%s$_{std}$ - %s$_{std}$' % (F1, F2), '%s$_{std}$ - %s$_{std}$' % (F1, F2),
                       '%s$_{std}$ - %s$_{std}$' % (F2, F3), '%s$_{std}$ - %s$_{std}$' % (F3, F4)]
        eq2_Ylabels = ['%s$_{ins}$ - %s$_{ins}$' % (F1, F2), '%s$_{ins}$ - %s$_{ins}$' % (F1, F2),
                       '%s$_{ins}$ - %s$_{ins}$' % (F2, F3), '%s$_{ins}$ - %s$_{ins}$' % (F3, F4)]
        fig = plt.figure(figsize=(19,9))
        fig.subplots_adjust(top=0.90, bottom=0.10, left=0.05, right=0.97)
        fig.suptitle('Std. Fit. %s #%s' % (self.MODE, self.AstNum))
        axs = []
        for k in range(8): axs.append(fig.add_subplot(2, 4, k + 1))
        for k in range(4):
            axs[k].set_title(self.FILTERS[k])
            axs[k].set_ylabel(eq1_Ylabels[k])
            x = self.FitSet[k].chi_eq2
            y = self.FitSet[k].b_ins - self.FitSet[k].r_ins
            axs[k].scatter(x, y, s=5, c='k')
            axs[k].axvline(np.mean(x) - 3 * np.std(x), linewidth=0.7, linestyle='--', c='k')
            axs[k].axvline(np.mean(x), linewidth=1, linestyle='--', c='k')
            axs[k].axvline(np.mean(x) + 3 * np.std(x), linewidth=0.7, linestyle='--', c='k')
            axs[k+4].set_ylabel(eq2_Ylabels[k])
            axs[k+4].set_xlabel('chi')
            x = self.FitSet[k].chi_eq1
            y = self.FitSet[k].b_std - self.FitSet[k].r_std
            axs[k+4].scatter(x, y, s=5, c='k')
            axs[k+4].axvline(np.mean(x) - 3 * np.std(x), linewidth=0.7, linestyle='--', c='k')
            axs[k+4].axvline(np.mean(x), linewidth=1, linestyle='--', c='k')
            axs[k+4].axvline(np.mean(x) + 3 * np.std(x), linewidth=0.7, linestyle='--', c='k')
        if save == True:
            fig.savefig(dir_save + 'ChiDist' + AddSaveName + '_%s_%s.png' %(self.MODE, self.AstNum), dpi=dpi)
            plt.close('all')
    """
    def Figure_Outlier(self, dpi=150):
        TabF1, TabF2, TabF3, TabF4 = self.MatTabSet
        F1, F2, F3, F4 = self.FILTERS
        Tab_m = [TabF1[1:], TabF2[1:], TabF3[1:], TabF4[1:]]
        Tab_b = [TabF1[1:], TabF1[1:], TabF2[1:], TabF3[1:]]
        Tab_r = [TabF2[1:], TabF2[1:], TabF3[1:], TabF4[1:]]
        b_labels = [F1, F1, F2, F3]
        r_labels = [F2, F2, F3, F4]
        fig = plt.figure(figsize=(16, 9))
        fig.subplots_adjust(top=0.90, bottom=0.10, left=0.10, right=0.95, wspace=0.3, hspace=0.33)
        axs = []
        fig.suptitle('Outliers %s #%s' %(self.MODE, self.AstNum))
        for k in range(4):
            axs.append(fig.add_subplot(4, 4, k * 4 + 1))
            axs.append(fig.add_subplot(4, 4, k * 4 + 2))
            axs.append(fig.add_subplot(4, 2, k * 2 + 2))
            mask_out_up = (self.FitSet[k].mask == False) * \
                          (Tab_m[k][:, 3] - Tab_m[k][:, 5] > self.FitSet[k].C2 +
                           self.FitSet[k].C1 * (Tab_b[k][:, 3] - Tab_r[k][:, 3]))
            mask_out_lo = (self.FitSet[k].mask == False) * \
                          (Tab_m[k][:, 3] - Tab_m[k][:, 5] < self.FitSet[k].C2 +
                           self.FitSet[k].C1 * (Tab_b[k][:, 3] - Tab_r[k][:, 3]))
            # blue-color samples in the image
        """
    """
    def FindTarget(self, TargetTab_Num, TargetTab_MODE):
        AstPhotTab = np.array(self.AstPhotSet)
        idx_target = np.where(TargetTab_MODE == self.MODE)[0]
        if len(idx_target) != 1:
            print('!!! no (or many) target in the list !!!')
            return -1
        self.TargetAstNum = TargetTab_Num[idx_target][0]
        print(self.TargetAstNum)
        idx_target_phot = np.where((AstPhotTab[:, 0] == self.MODE) & (AstPhotTab[:, 1] == self.TargetAstNum))
        if len(idx_target_phot) != 1:
            print('!!! no (or many) target in the AstPhotSet !!!')
            return -1
        self.TargetAstPhot = AstPhotTab[idx_target_phot]
        return 0
    """
font = {'size' : 10}
matplotlib.rc('font', **font)
