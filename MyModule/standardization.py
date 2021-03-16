# modified regression algorithm : fit eq2, masking, fit eq1, masking -- repeat
# modified regression algorithm : loss function inverse density weighting
import numpy as np
from scipy.optimize import leastsq
import sys
sys.path.append("/home/astro/PycharmProjects/KMTNet_git/KMTNet/MyModule")
from ModuleCommon import err_Add, err_Mul


def func_linear(x, a, b): return a+b*x

def func_loss(coe, x, y, xerr, yerr, type='none', Narea=5):
    w = np.ones(len(x))
    if type == 'inverse density' :
        inverse_density = np.zeros(Narea)           # number density = N_data
        area_boundary = np.linspace(np.min(x)*0.99, np.max(x)*1.01, Narea+1)
        for ii in range(Narea) :
            idx = np.where((x >= area_boundary[ii]) & (x < area_boundary[ii + 1]))[0]
            if len(idx) == 0 : inverse_density[ii] = 0
            else : inverse_density[ii] = 1/len(idx)
        inverse_density = inverse_density * len(x) / np.sum(inverse_density)
        for ii in range(Narea) :
            idx = np.where((x >= area_boundary[ii]) & (x < area_boundary[ii + 1]))
            w[idx] = inverse_density[ii]**0.5
    elif type == 'inverse density 0.5' :
        root_inverse_density = np.zeros(Narea)           # number density = N_data
        area_boundary = np.linspace(np.min(x)*0.99, np.max(x)*1.01, Narea+1)
        for ii in range(Narea) :
            idx = np.where((x >= area_boundary[ii]) & (x < area_boundary[ii + 1]))[0]
            if len(idx) == 0 : root_inverse_density[ii] = 0
            else : root_inverse_density[ii] = 1/(len(idx)**0.5)
        root_inverse_density = root_inverse_density * len(x) / np.sum(root_inverse_density)
        for ii in range(Narea) :
            idx = np.where((x >= area_boundary[ii]) & (x < area_boundary[ii + 1]))
            w[idx] = root_inverse_density[ii]**0.5
    elif type == 'inverse density 1.5' :
        root_inverse_density = np.zeros(Narea)           # number density = N_data
        area_boundary = np.linspace(np.min(x)*0.99, np.max(x)*1.01, Narea+1)
        for ii in range(Narea) :
            idx = np.where((x >= area_boundary[ii]) & (x < area_boundary[ii + 1]))[0]
            if len(idx) == 0 : root_inverse_density[ii] = 0
            else : root_inverse_density[ii] = 1/(len(idx)**1.5)
        root_inverse_density = root_inverse_density * len(x) / np.sum(root_inverse_density)
        for ii in range(Narea) :
            idx = np.where((x >= area_boundary[ii]) & (x < area_boundary[ii + 1]))
            w[idx] = root_inverse_density[ii]**0.5
    return w * (y - func_linear(x, coe[0], coe[1]))/((yerr**2 + coe[1]**2*xerr**2)**0.5)

def sigma_boundary(x, a, b, aerr, berr, sigmaclip):
    table = np.zeros((len(x), 4))
    A = np.array([a-sigmaclip*aerr, a+sigmaclip*aerr, a-sigmaclip*aerr, a+sigmaclip*aerr])
    B = np.array([b-sigmaclip*berr, b-sigmaclip*berr, b+sigmaclip*berr, b+sigmaclip*berr])
    for i in range(4) :
        table[:, i] = func_linear(x, A[i], B[i])
    boundary_max = np.max(table, axis=1)
    boundary_min = np.min(table, axis=1)
    return boundary_min, boundary_max


# inputs : for each star)
# mag_inst, merr_inst, mag_std, merr_std,
# red_inst, rerr_inst, red_std, rerr_std,
# blue_inst, berr_inst, blue_std, berr_std

class standardization_fit:
    def __init__(self, m_ins, e_ins, m_std, e_std, b_ins, be_ins, b_std, be_std, r_ins, re_ins, r_std, re_std):
        self.m_ins = m_ins
        self.e_ins = e_ins
        self.m_std = m_std
        self.e_std = e_std
        self.b_ins = b_ins
        self.be_ins = be_ins
        self.b_std = b_std
        self.be_std = be_std
        self.r_ins = r_ins
        self.re_ins = re_ins
        self.r_std = r_std
        self.re_std = re_std
        self.clip_err = False

    def fit(self, coe1=np.array([0, 0.2]), coe2=np.array([0.2, 1]), sigmaclip=[3.0, 3.0, 3.0],
            maxiter=[10, 10], losstype=['basic', 'basic'], cliptype=['fit', 'fit'], clip_err=True, N_limiting = True, Nlim_min=34,
            value_sig_add=0.5):
        self.FitFailureFlag = False
        """
        fit Coefficients of these equations
            m_std - m_ins = C2 + C1 (b_std - r_std)  --- eq 1
            b_std - r_std = C4 + C3 (b_ins - r_ins)  --- eq 2
        coe1 : eq1 (C2, C1) fit starting values
        coe2 : eq2 (C4, C2) fit starting values
        sigmaclip : [ clip_eq1, clip_eq2, clip_err ]
        maxiter : [ maxiter_eq1, maxiter_eq2 ]
        losstype : [ losstype_eq1, losstype_eq2 ]
        cliptype : [cliptype_eq1, cliptype_eq2 ]
        clip_err : if ture, data with large error are clipped before fitting
        """
        mask = np.logical_not(np.isnan(self.be_ins)) * np.logical_not(np.isnan(self.re_ins)) * \
               np.logical_not(np.isnan(self.be_std)) * np.logical_not(np.isnan(self.re_std)) * \
               np.logical_not(np.isnan(self.e_ins)) * np.logical_not(np.isnan(self.e_std))

        self.err_color_ins = (self.be_ins ** 2 + self.re_ins ** 2) ** 0.5
        self.err_color_std = (self.be_std ** 2 + self.re_std ** 2) ** 0.5
        self.err_m_stdins = (self.e_std ** 2 + self.e_ins ** 2) ** 0.5

        #mask = np.full(len(self.m_ins), True)
        eq1_fit = np.array([coe1]); eq2_fit = np.array([coe2])

        # Sample Size Check
        if N_limiting is True and sum(mask) < Nlim_min :
            print('StdFitFailure - Too Small Sample Size : # %s' % sum(mask))
            self.FitFailureFlag = True
            return -1

        ### Error Clipping ###
        if clip_err is True:
            self.clip_err = True
            self.mean_eci = np.mean(self.err_color_ins)
            self.std_eci = np.std(self.err_color_ins)
            self.mean_ecs = np.mean(self.err_color_std)
            self.std_ecs = np.std(self.err_color_std)
            self.mean_ems = np.mean(self.err_m_stdins)
            self.std_ems = np.std(self.err_m_stdins)

            mask_wo_err = (mask == True) * \
                          (self.mean_eci + self.std_eci * sigmaclip[2] >= self.err_color_ins) * \
                          (self.mean_ecs + self.std_ecs * sigmaclip[2] >= self.err_color_std) * \
                          (self.mean_ems + self.std_ems * sigmaclip[2] >= self.err_m_stdins)
            self.mask_err = np.logical_not(mask_wo_err.copy())
            if sum(mask) >= Nlim_min: mask = mask_wo_err
            else: print('\tError Clipping Failure ( # %s are remained )' %sum(mask))

        Nsample0 = sum(mask)
        if N_limiting == True: Nlim = np.maximum(Nsample0/2, Nlim_min)
        else: Nlim = 0

        ### fit eq 2 ###
        Niter = 0
        FLAG_Sample = False
        for i in range(maxiter[1]):
            Niter += 1
            ### fitting ###
            eq2_fit = leastsq(func_loss, eq2_fit[0],
                              args=(self.b_ins[mask] - self.r_ins[mask], self.b_std[mask] - self.r_std[mask],
                                    self.err_color_ins[mask], self.err_color_std[mask], losstype[1]), full_output=True)
            if eq2_fit[4] > 4: assert False, '\tEq2 fit flag : %s' % eq2_fit[4]

            eq2_a_temp, eq2_b_temp = eq2_fit[0]
            eq2_aerr_temp, eq2_berr_temp = np.diag(eq2_fit[1]) ** 0.5
            eq2_boundary = sigma_boundary(self.b_ins - self.r_ins, eq2_a_temp, eq2_b_temp,
                                          eq2_aerr_temp, eq2_berr_temp, sigmaclip=sigmaclip[1])
            ### clipping ###
            if cliptype[1] == 'fit':
                mask_eq2 = (mask == True) * \
                           (self.b_std-self.r_std >= eq2_boundary[0]) * (self.b_std-self.r_std <= eq2_boundary[1])
                ### guarantee sample number ###
                sigma_add = 0
                while sum(mask_eq2) < Nlim:
                    if FLAG_Sample == False: FLAG_Sample = True
                    sigma_add += value_sig_add
                    eq2_boundary = sigma_boundary(self.b_ins - self.r_ins, eq2_a_temp, eq2_b_temp,
                                                  eq2_aerr_temp, eq2_berr_temp, sigmaclip=sigmaclip[1] + sigma_add)
                    mask_eq2 = (mask == True) * \
                               (self.b_std - self.r_std >= eq2_boundary[0]) * (self.b_std - self.r_std <= eq2_boundary[1])
            elif cliptype[1] == 'chi':
                chi_eq2 = func_loss(eq2_fit[0], self.b_ins - self.r_ins, self.b_std - self.r_std,
                                      self.err_color_ins, self.err_color_std, losstype[1])
                chi_eq2_nanmask = np.logical_not(np.isnan(chi_eq2))
                chi_eq2_mean = np.mean(chi_eq2[chi_eq2_nanmask])
                chi_eq2_std = np.std(chi_eq2[chi_eq2_nanmask])
                mask_eq2 = (mask == True) * (chi_eq2 >= chi_eq2_mean - chi_eq2_std * sigmaclip[1]) * \
                           (chi_eq2 <= chi_eq2_mean + chi_eq2_std * sigmaclip[1])
                ### guarantee sample number ###
                sigma_add = 0
                while sum(mask_eq2) < Nlim:
                    if FLAG_Sample == False: FLAG_Sample = True
                    sigma_add += value_sig_add
                    mask_eq2 = (mask == True) * (chi_eq2 >= chi_eq2_mean - chi_eq2_std * (sigmaclip[1] + sigma_add)) *\
                               (chi_eq2 <= chi_eq2_mean + chi_eq2_std * (sigmaclip[1] + sigma_add))
            elif cliptype[1] == 'none':  mask_eq2 = mask.copy()

            ### iteration decision ###
            if i == maxiter[1] - 1: print('case-MaxIter'); break                            # case 1 [reach maxiter] final clipping should not be conducted
            elif FLAG_Sample == True: print('case-SampleLimit'); mask = mask_eq2; break    # case 2 [sample limit]
            elif np.array_equiv(mask, mask_eq2): print('case-Equiv'); break                # case 3 [ideal fit end] (good)
            else: mask = mask_eq2                                                           # case 4 [iteration] (good)

        self.mask_outlier_eq2 = np.logical_not(mask.copy())     # includes error clipped members !!!
        Nsample1 = sum(mask)

        ### fit eq 1 ###
        Niter = 0
        FLAG_Sample = False
        for i in range(maxiter[0]):
            Niter += 1
            ### fitting ###
            eq1_fit = leastsq(func_loss, eq1_fit[0],
                              args=(self.b_std[mask] - self.r_std[mask], self.m_std[mask] - self.m_ins[mask],
                                    self.err_color_std[mask], self.err_m_stdins[mask], losstype[0]), full_output=True)
            if eq1_fit[4] > 4 : assert False, 'Eq1 fit flag : %s' % eq1_fit[4]

            eq1_a_temp, eq1_b_temp = eq1_fit[0]
            eq1_aerr_temp, eq1_berr_temp = np.diag(eq1_fit[1]) ** 0.5
            eq1_boundary = sigma_boundary(self.b_std - self.r_std, eq1_a_temp, eq1_b_temp,
                                          eq1_aerr_temp, eq1_berr_temp, sigmaclip=sigmaclip[0])
            ### clipping ###
            if cliptype[0] == 'fit':
                mask_eq1 = (mask == True) * \
                           (self.m_std-self.m_ins >= eq1_boundary[0]) * (self.m_std-self.m_ins <= eq1_boundary[1])
                ### guarantee sample number ###
                sigma_add = 0
                while sum(mask_eq1) < Nlim :
                    if FLAG_Sample == False : FLAG_Sample = True
                    sigma_add += value_sig_add
                    eq1_boundary = sigma_boundary(self.b_std - self.r_std, eq1_a_temp, eq1_b_temp,
                                                  eq1_aerr_temp, eq1_berr_temp, sigmaclip=sigmaclip[0] + sigma_add)
                    mask_eq1 = (mask == True) * \
                               (self.m_std - self.m_ins >= eq1_boundary[0]) * (self.m_std - self.m_ins <= eq1_boundary[1])
            elif cliptype[0] == 'chi':
                chi_eq1 = func_loss(eq1_fit[0], self.b_std - self.r_std, self.m_std - self.m_ins,
                                      self.err_color_std, self.err_m_stdins, losstype[0])
                chi_eq1_nanmask = np.logical_not(np.isnan(chi_eq1))
                chi_eq1_mean = np.mean(chi_eq1[chi_eq1_nanmask])
                chi_eq1_std = np.std(chi_eq1[chi_eq1_nanmask])
                mask_eq1 = (mask == True) * (chi_eq1 >= chi_eq1_mean - chi_eq1_std * sigmaclip[0]) * \
                           (chi_eq1 <= chi_eq1_mean + chi_eq1_std * sigmaclip[0])
                ### guarantee sample number ###
                sigma_add = 0
                while sum(mask_eq1)< Nlim:
                    if FLAG_Sample == False: FLAG_Sample = True
                    sigma_add += value_sig_add
                    mask_eq1 = (mask == True) * (chi_eq1 >= chi_eq1_mean - chi_eq1_std * (sigmaclip[0] + sigma_add)) *\
                               (chi_eq1 <= chi_eq1_mean + chi_eq1_std * (sigmaclip[0] + sigma_add))
            elif cliptype[1] == 'none':  mask_eq1 = mask.copy()

            ### iteration decision ###
            if i == maxiter[0]-1 : print('case-MaxIter'); break                             # case 1 [reach maxiter] final clipping should not be conducted
            elif FLAG_Sample == True : print('case-SampleLimit'); mask = mask_eq1; break    # case 1 [sample limit]
            elif np.array_equiv(mask, mask_eq1) : print('case-Equiv'); break                # case 2 [ideal fit end] (good)
            else : mask = mask_eq1                                                          # case 4 [iteration] (good)

        self.mask_outlier_eq1 = np.logical_not(mask.copy())     # includes error clipped members !!!
        print('fitting N sample : %s -> %s -> %s' % (Nsample0, Nsample1, sum(mask)))

        self.sigmaclip = sigmaclip
        self.mask = mask
        self.mask_outlier = np.logical_not(self.mask)           # includes error clipped members !!!
        self.eq1_fit = eq1_fit
        self.eq2_fit = eq2_fit
        self.fititer = Niter
        self.C1 = eq1_fit[0][1]; self.C1err = eq1_fit[1][1, 1] ** 0.5
        self.C2 = eq1_fit[0][0]; self.C2err = eq1_fit[1][0, 0] ** 0.5
        self.C3 = eq2_fit[0][1]; self.C3err = eq2_fit[1][1, 1] ** 0.5
        self.C4 = eq2_fit[0][0]; self.C4err = eq2_fit[1][0, 0] ** 0.5
        self.boundary_eq2 = eq2_boundary; self.boundary_eq1 = eq1_boundary
        if cliptype[1] == 'chi':
            self.chi_eq2 = func_loss(eq2_fit[0], self.b_ins - self.r_ins, self.b_std - self.r_std,
                                     self.err_color_ins, self.err_color_std, losstype[1])
        if cliptype[0] == 'chi':
            self.chi_eq1 = func_loss(eq1_fit[0], self.b_std - self.r_std, self.m_std - self.m_ins,
                                     self.err_color_std, self.err_m_stdins, losstype[0])
        return self.mask, eq1_fit, eq2_fit, Niter

    def standardization(self, m_ins, e_ins, c_ins, ce_ins):
        """
        m_std - m_ins = C2 + C1 (b_std - r_std)  --- eq 1
        b_std - r_std = C4 + C3 (b_ins - r_ins)  --- eq 2
        """
        c_std = self.C4 + self.C3 * c_ins
        #try: print(len(m_ins))
        #except: print(c_ins, self.C3err, ce_ins, self.C1, self.C1err, self.C2err)
        ce_std = err_Add([np.ones_like(c_std)*self.C4err, err_Mul([np.ones_like(c_std)*self.C3, c_ins],
                                                                  [np.ones_like(c_std)*self.C3err, ce_ins])])
        
        if type(c_ins) is np.float64:
            if c_ins == 0.:
                ce_std = err_Add([np.ones_like(c_std)*self.C4err, ((self.C3err/self.C3)**2 + ce_ins**2)**0.5])
        

        m_std = m_ins + self.C2 + self.C1 * c_std
        e_std = err_Add([e_ins, np.ones_like(c_std)*self.C2err, err_Mul([np.ones_like(c_std)*self.C1, c_std],
                                                                        [np.ones_like(c_std)*self.C1err, ce_std])])
        return m_std, e_std, c_std, ce_std

    def plot_fitting(self, ax1, ax2, eq1_Xlabel='', eq1_Ylabel='', eq2_Xlabel='', eq2_Ylabel=''):
        self.mask_outlier_woerr_eq1 = self.mask_outlier_eq1 * np.logical_not(self.mask_outlier_eq2)
        if self.clip_err == True:
            self.mask_outlier_woerr_eq2 = self.mask_outlier_eq2 * np.logical_not(self.mask_err)
        else:
            self.mask_outlier_woerr_eq2 = self.mask_outlier_eq2

        """ eq1 """
        X_eq1 = self.b_std - self.r_std
        Y_eq1 = self.m_std - self.m_ins
        Xe_eq1 = self.err_color_std
        Ye_eq1 = self.err_m_stdins
        plotx = np.array([np.min(X_eq1), np.max(X_eq1)])
        plot_boundary_eq1 = sigma_boundary(plotx, self.C2, self.C1, self.C2err, self.C1err, self.sigmaclip[0])
        ax1.fill_between(plotx, plot_boundary_eq1[0], plot_boundary_eq1[1], alpha=0.15)
        if self.clip_err == True and sum(self.mask_err != 0):
            ax1.errorbar(X_eq1[self.mask_err], Y_eq1[self.mask_err],
                         xerr=Xe_eq1[self.mask_err],
                         yerr=Ye_eq1[self.mask_err],
                         fmt='o', markersize=0, c='grey', alpha=0.5,
                         ecolor='grey', elinewidth=0.4, capsize=1, capthick=0.5)
        if sum(self.mask_outlier_woerr_eq1) != 0:
            ax1.errorbar(X_eq1[self.mask_outlier_woerr_eq1], Y_eq1[self.mask_outlier_woerr_eq1],
                         xerr=Xe_eq1[self.mask_outlier_woerr_eq1],
                         yerr=Ye_eq1[self.mask_outlier_woerr_eq1],
                         fmt='o', markersize=2, c='blue', alpha=0.6,
                         ecolor='blue', elinewidth=0.4, capsize=1, capthick=0.5)
        if sum(self.mask_outlier_woerr_eq2) != 0 :
            ax1.errorbar(X_eq1[self.mask_outlier_woerr_eq2], Y_eq1[self.mask_outlier_woerr_eq2],
                         xerr=Xe_eq1[self.mask_outlier_woerr_eq2],
                         yerr=Ye_eq1[self.mask_outlier_woerr_eq2],
                         fmt='o', markersize=2, c='grey', alpha=0.3,
                         ecolor='grey', elinewidth=0.4, capsize=1, capthick=0.5)
        ax1.errorbar(X_eq1[self.mask], Y_eq1[self.mask],
                     xerr=Xe_eq1[self.mask], yerr=Ye_eq1[self.mask],
                     fmt='o', markersize=2, c='k', alpha=0.7,
                     ecolor='k', elinewidth=0.4, capsize=1, capthick=0.5)
        if self.eq1_fit[0][1] >= 0:
            ax1.plot(plotx, func_linear(plotx, self.C2, self.C1),
                     linestyle='--', c='k', linewidth=0.7, label='%.3f+%.3fx' % (self.C2, self.C1))
        else: ax1.plot(plotx, func_linear(plotx, self.C2, self.C1),
                       linestyle='--', c='k', linewidth=0.7, label='%.3f %.3fx' % (self.C2, self.C1))
        ax1.legend()
        ax1.set_xlabel('$%s$' %eq1_Xlabel, fontsize=13)
        ax1.set_ylabel('$%s$' %eq1_Ylabel, fontsize=13)

        """ eq2 """
        X_eq2 = self.b_ins - self.r_ins
        Y_eq2 = self.b_std - self.r_std
        Xe_eq2 = self.err_color_ins
        Ye_eq2 = self.err_color_std
        plotx = np.array([np.min(X_eq2), np.max(X_eq2)])
        plot_boundary_eq2 = sigma_boundary(plotx, self.C4, self.C3, self.C4err, self.C3err, self.sigmaclip[1])
        ax2.fill_between(plotx, plot_boundary_eq2[0], plot_boundary_eq2[1], alpha=0.15)
        if self.clip_err == True and sum(self.mask_err != 0):
            ax2.errorbar(X_eq2[self.mask_err], Y_eq2[self.mask_err],
                         xerr=Xe_eq2[self.mask_err],
                         yerr=Ye_eq2[self.mask_err],
                         fmt='o', markersize=0, c='grey', alpha=0.5,
                         ecolor='grey', elinewidth=0.4, capsize=1, capthick=0.5)
        if sum(self.mask_outlier_woerr_eq2 != 0):
            ax2.errorbar(X_eq2[self.mask_outlier_woerr_eq2],
                         Y_eq2[self.mask_outlier_woerr_eq2],
                         xerr=Xe_eq2[self.mask_outlier_woerr_eq2],
                         yerr=Ye_eq2[self.mask_outlier_woerr_eq2],
                         fmt='o', markersize=2, c='grey', alpha=0.6,
                         ecolor='grey', elinewidth=0.4, capsize=1, capthick=0.5)
        ax2.errorbar(X_eq2[np.logical_not(self.mask_outlier_eq2)], Y_eq2[np.logical_not(self.mask_outlier_eq2)],
                     xerr=Xe_eq2[np.logical_not(self.mask_outlier_eq2)],
                     yerr=Ye_eq2[np.logical_not(self.mask_outlier_eq2)],
                     fmt='o', markersize=2, c='k', alpha=0.7,
                     ecolor='k', elinewidth=0.4, capsize=1, capthick=0.5)
        if self.eq2_fit[0][1] >= 0:
            ax2.plot(plotx, func_linear(plotx, self.C4, self.C3),
                     linestyle='--', c='k', linewidth=0.7, label='%.3f+%.3fx' % (self.C4, self.C3))
        else:
            ax2.plot(plotx, func_linear(plotx, self.C4, self.C3),
                     linestyle='--', c='k', linewidth=0.7, label='%.3f %.3fx' % (self.C4, self.C3))
        ax2.legend()
        ax2.set_xlabel('$%s$' % eq2_Xlabel, fontsize=13)
        ax2.set_ylabel('$%s$' % eq2_Ylabel, fontsize=13)


















    # temp code: trace error sources
    def plot_fitting_error(self, ax1, ax2, eq1_Xlabel='', eq1_Ylabel='', eq2_Xlabel='', eq2_Ylabel=''):
        self.mask_outlier_woerr_eq1 = self.mask_outlier_eq1 * np.logical_not(self.mask_outlier_eq2)
        if self.clip_err == True:
            self.mask_outlier_woerr_eq2 = self.mask_outlier_eq2 * np.logical_not(self.mask_err)
        else:
            self.mask_outlier_woerr_eq2 = self.mask_outlier_eq2

        """ eq1 """
        X_eq1 = self.b_std - self.r_std
        Y_eq1 = self.m_std - self.m_ins
        Xe_eq1 = self.err_color_std
        Ye_eq1 = self.err_m_stdins
        plotx = np.array([np.min(X_eq1), np.max(X_eq1)])
        plot_boundary_eq1 = sigma_boundary(plotx, self.C2, self.C1, self.C2err, self.C1err, self.sigmaclip[0])
        ax1.fill_between(plotx, plot_boundary_eq1[0], plot_boundary_eq1[1], alpha=0.15)
        if self.clip_err == True and sum(self.mask_err != 0):
            ax1.errorbar(X_eq1[self.mask_err], Y_eq1[self.mask_err],
                         xerr=Xe_eq1[self.mask_err],
                         yerr=Ye_eq1[self.mask_err],
                         fmt='o', markersize=0, c='grey', alpha=0.5,
                         ecolor='grey', elinewidth=0.4, capsize=1, capthick=0.5)
        if sum(self.mask_outlier_woerr_eq1) != 0:
            ax1.errorbar(X_eq1[self.mask_outlier_woerr_eq1], Y_eq1[self.mask_outlier_woerr_eq1],
                         xerr=Xe_eq1[self.mask_outlier_woerr_eq1],
                         yerr=Ye_eq1[self.mask_outlier_woerr_eq1],
                         fmt='o', markersize=2, c='blue', alpha=0.6,
                         ecolor='blue', elinewidth=0.4, capsize=1, capthick=0.5)
        if sum(self.mask_outlier_woerr_eq2) != 0 :
            ax1.errorbar(X_eq1[self.mask_outlier_woerr_eq2], Y_eq1[self.mask_outlier_woerr_eq2],
                         xerr=Xe_eq1[self.mask_outlier_woerr_eq2],
                         yerr=Ye_eq1[self.mask_outlier_woerr_eq2],
                         fmt='o', markersize=2, c='grey', alpha=0.3,
                         ecolor='grey', elinewidth=0.4, capsize=1, capthick=0.5)
        ax1.errorbar(X_eq1[self.mask], Y_eq1[self.mask],
                     xerr=Xe_eq1[self.mask], yerr=Ye_eq1[self.mask],
                     fmt='o', markersize=2, c='k', alpha=0.7,
                     ecolor='k', elinewidth=0.4, capsize=1, capthick=0.5)
        if self.eq1_fit[0][1] >= 0:
            ax1.plot(plotx, func_linear(plotx, self.C2, self.C1),
                     linestyle='--', c='k', linewidth=0.7, label='%.3f+%.3fx' % (self.C2, self.C1))
        else: ax1.plot(plotx, func_linear(plotx, self.C2, self.C1),
                       linestyle='--', c='k', linewidth=0.7, label='%.3f %.3fx' % (self.C2, self.C1))
        ax1.legend()
        ax1.set_xlabel('$%s$' %eq1_Xlabel, fontsize=13)
        ax1.set_ylabel('$%s$' %eq1_Ylabel, fontsize=13)

        """ eq2 """
        X_eq2 = self.b_std - self.r_std
        Y_eq2 = self.m_std
        Xe_eq2 = self.e_ins
        Ye_eq2 = self.e_std
        plotx = np.array([np.min(X_eq2), np.max(X_eq2)])
        plot_boundary_eq2 = sigma_boundary(plotx, self.C4, self.C3, self.C4err, self.C3err, self.sigmaclip[1])
        #ax2.fill_between(plotx, plot_boundary_eq2[0], plot_boundary_eq2[1], alpha=0.15)
        if self.clip_err == True and sum(self.mask_err != 0):
            ax2.errorbar(X_eq2[self.mask_err], Y_eq2[self.mask_err],
                         xerr=Xe_eq2[self.mask_err],
                         # yerr=Ye_eq2[self.mask_err],
                         fmt='o', markersize=0, c='grey', alpha=0.5,
                         ecolor='grey', elinewidth=0.4, capsize=1, capthick=0.5)
            ax2.errorbar(X_eq2[self.mask_err], Y_eq2[self.mask_err]+0.001,
                         xerr=Ye_eq2[self.mask_err],
                         # yerr=Ye_eq2[self.mask_err],
                         fmt='o', markersize=0, c='cyan', alpha=0.5,
                         ecolor='cyan', elinewidth=0.4, capsize=1, capthick=0.5)
        if sum(self.mask_outlier_woerr_eq2 != 0):
            ax2.errorbar(X_eq2[self.mask_outlier_woerr_eq2],
                         Y_eq2[self.mask_outlier_woerr_eq2],
                         xerr=Xe_eq2[self.mask_outlier_woerr_eq2],
                         #yerr=Ye_eq2[self.mask_outlier_woerr_eq2],
                         fmt='o', markersize=0, c='grey', alpha=0.6,
                         ecolor='grey', elinewidth=0.4, capsize=1, capthick=0.5)
            ax2.errorbar(X_eq2[self.mask_outlier_woerr_eq2],
                         Y_eq2[self.mask_outlier_woerr_eq2],
                         xerr=Ye_eq2[self.mask_outlier_woerr_eq2],
                         # yerr=Ye_eq2[self.mask_outlier_woerr_eq2],
                         fmt='o', markersize=0, c='cyan', alpha=0.6,
                         ecolor='cyan', elinewidth=0.4, capsize=1, capthick=0.5)
        ax2.errorbar(X_eq2[np.logical_not(self.mask_outlier_eq2)], Y_eq2[np.logical_not(self.mask_outlier_eq2)],
                     xerr=Xe_eq2[np.logical_not(self.mask_outlier_eq2)],
                     #yerr=Ye_eq2[np.logical_not(self.mask_outlier_eq2)],
                     fmt='o', markersize=0, c='k', alpha=0.7,
                     ecolor='k', elinewidth=0.4, capsize=1, capthick=0.5)
        ax2.errorbar(X_eq2[np.logical_not(self.mask_outlier_eq2)], Y_eq2[np.logical_not(self.mask_outlier_eq2)],
                     xerr=Ye_eq2[np.logical_not(self.mask_outlier_eq2)],
                     #yerr=Ye_eq2[np.logical_not(self.mask_outlier_eq2)],
                     fmt='o', markersize=0, c='blue', alpha=0.7,
                     ecolor='blue', elinewidth=0.4, capsize=1, capthick=0.5)
        #if self.eq2_fit[0][1] >= 0:
        #    ax2.plot(plotx, func_linear(plotx, self.C4, self.C3),
        #             linestyle='--', c='k', linewidth=0.7, label='%.3f+%.3fx' % (self.C4, self.C3))
        #else:
        #    ax2.plot(plotx, func_linear(plotx, self.C4, self.C3),
        #             linestyle='--', c='k', linewidth=0.7, label='%.3f %.3fx' % (self.C4, self.C3))
        #ax2.legend()
        ax2.set_xlabel('$%s$ (ins:black / std:blue)' % eq2_Xlabel, fontsize=13)
        ax2.set_ylabel('Mag', fontsize=13)

"""           
            eq2_mask, eq2_fit, _ = \
                linfit_sigclip(self.m2_std[mask] - self.m1_std[mask], self.m2_ins[mask] - self.m1_ins[mask],
                               err_color_std[mask], err_color_ins[mask], sigmaclip=sigmaclip, maxiter=1)

            ### fit eq 1 ###
            eq1_mask, eq1_fit, _ = \
                linfit_sigclip(self.m1_std[mask] - self.m1_ins[mask], self.m2_std[mask] - self.m1_std[mask],
                               err_m1_stdins[mask], err_color_std[mask], sigmaclip=sigmaclip, maxiter=1)

            ### clipping ###
            print(Niter, 2, len(eq1_mask))
            mask_new = (mask == True) * eq1_mask * eq2_mask
            print(Niter, 3, len(mask_new))
            if np.array_equiv(mask, mask_new) : break
            else : mask = mask_new

        self.mask = mask
        self.eq1_fit = eq1_fit
        self.eq2_fit = eq2_fit
        self.fititer = Niter

        return self.mask, self.eq1_fit, self.eq2_fit, self.fititer
"""
