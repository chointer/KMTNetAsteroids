import pandas as pd
import numpy as np
import sys
sys.path.append("/home/astro/PycharmProjects/KMTNet_git/KMTNet/MyModule")
from ModuleCommon import err_Add, err_Mul, linfit_sigclip

class PCA:
    def __init__(self, table, FILTERS, constant=True, coe0=np.array([0, 1]), sigmaclip=10, maxiter=2):
        self.FILTERS = FILTERS
        self.HeaderName = ['F1', 'F1e', 'F2', 'F2e', 'F3', 'F3e', 'F4', 'F4e']
        self.AstTab = pd.DataFrame(table, columns=self.HeaderName, dtype=float)

        x = self.AstTab['F1'] - self.AstTab['F2']
        xerr = err_Add([self.AstTab['F1e'], self.AstTab['F2e']])
        y = self.AstTab['F2'] - self.AstTab['F3']
        yerr = err_Add([self.AstTab['F2e'], self.AstTab['F3e']])

        mask, fit, Niter = linfit_sigclip(x, y, xerr, yerr, coe0=coe0, sigmaclip=sigmaclip, maxiter=maxiter)
        theta = np.arctan(fit[0][1])
        coserr = (np.cos(np.arctan(fit[0][1] - fit[1][1, 1] ** 0.5)) - np.cos(np.arctan(fit[0][1])) +
                  np.cos(np.arctan(fit[0][1] + fit[1][1, 1] ** 0.5)) - np.cos(np.arctan(fit[0][1]))) / 2
        sinerr = (np.sin(np.arctan(fit[0][1] - fit[1][1, 1] ** 0.5)) - np.sin(np.arctan(fit[0][1])) +
                  np.sin(np.arctan(fit[0][1] + fit[1][1, 1] ** 0.5)) - np.sin(np.arctan(fit[0][1]))) / 2
        a_star = np.cos(theta) * x + np.sin(theta) * y
        a_star_err = err_Add([err_Mul([np.full(len(x), np.cos(np.arctan(fit[0][1])), dtype=np.float), x],
                              [np.full(len(x), coserr, dtype=np.float), xerr]),
                      err_Mul([np.full(len(y), np.sin(np.arctan(fit[0][1])), dtype=np.float), y],
                              [np.full(len(y), sinerr, dtype=np.float), yerr])])

        if constant is False:
            print('a_star = %.3f * (%s - %s) + %.3f * (%s - %s)'
                  % (np.cos(theta), self.FILTERS[0], self.FILTERS[1],
                     np.sin(theta), self.FILTERS[1], self.FILTERS[2]))

        elif constant is True:
            print('a_star = %.3f * (%s - %s) + %.3f * (%s - %s) - %.3f'
                  % (np.cos(theta), self.FILTERS[0], self.FILTERS[1],
                     np.sin(theta), self.FILTERS[1], self.FILTERS[2], np.mean(a_star)))
            a_star = a_star - np.mean(a_star)

        self.AstTab['a_star'] = a_star
        self.AstTab['a_star_err'] = a_star_err
        self.a_star = a_star
        self.a_star_err = a_star_err
        self.PCAfit = fit


def drawPCA_simple(ax, F1, F1e, F2, F2e, F3, F3e, F4, F4e, a_star, a_star_err,
                   fsize=17, col='k', alp=1., err=True):
    F1F2 = F1 - F2
    F1F2e = err_Add([F1e, F2e])
    F2F3 = F2 - F3
    F2F3e = err_Add([F2e, F3e])
    F3F4 = F3 - F4
    F3F4e = err_Add([F3e, F4e])
    plotx = np.array([np.mean(F1F2) - 2 * np.std(F1F2), np.mean(F1F2) + 2 * np.std(F1F2)])
    # ax.set_ylabel(FILTERS[2] + '-' + FILTERS[3], fontsize=fsize)
    # ax.set_xlabel('a_star', fontsize=fsize)
    if err is True:
        ax.errorbar(a_star, F3F4, xerr=a_star_err, yerr=F3F4e, fmt='o', markersize=2,
                    ecolor=col, elinewidth=0.5, capsize=0.5, capthick=0.5, c=col, alpha=alp)
    else:
        ax.scatter(a_star, F3F4, s=2, c=col, alpha=alp)
    ax.tick_params(axis='x', labelsize=fsize)
    ax.tick_params(axis='y', labelsize=fsize)



