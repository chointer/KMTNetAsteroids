import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Magnitude Decision Rule
# targeted asteroid: the latest observation
# serendipitously observed asteroid: average all observation


def getTarInfoList_Flat(table_TarInfo_name):
    # multiple target -> make one by one
    table_TarInfo = np.genfromtxt(table_TarInfo_name, dtype=np.str)
    for i in range(len(table_TarInfo)):
        separation = table_TarInfo[i, 1].split(sep=',')
        if len(separation) >= 2:
            table_TarInfo[i, 1] = separation[0]
            for seps in range(1, len(separation)):
                table_TarInfo = np.concatenate((table_TarInfo,
                                                [[table_TarInfo[i, 0], separation[seps], table_TarInfo[i, 2]]]))
    return table_TarInfo


class getAstPhotList:
    def __init__(self):
        self.table_id = None        # [[astID, date, mode]]
        self.table_ph = None        # [[mag set]]
        self.table_fl = None        # [[FLAGs_avg, FLAG_target]]
        self.FLAG_table = False

    def addast(self, info, table_Tar):
        # info:
        # [0] Date, [1]MODE, [2]astID, [3-10]mag&err,
        # [11], [12], [13], [14], [15], [16], [17], [18], [19], [20] obs_time_span

        # target_checking
        mask_target = (table_Tar[:, 0] == info[0]) & (table_Tar[:, 1] == info[2]) & (table_Tar[:, 2] == info[1])
        if sum(mask_target) == 0:
            FLAG_target = 1     # FLAG_target: Target=0, No-Target=Number of observations!
        elif sum(mask_target) == 1:
            FLAG_target = 0
        elif sum(mask_target) > 1:
            assert False, 'Error: getAstPhotList.addast.target_checking'

        # [case 0] generate lists
        if self.FLAG_table is False:
            self.table_id = np.array([[info[2], info[0], info[1]]], dtype=np.str)       # [[astID, date, mode]]
            self.table_ph = np.array([info[3:11]], dtype=np.float)
            self.table_fl = np.array([[info[11], info[12], info[13], info[14], info[15],
                                       info[16], info[17], info[18], info[19], info[20],
                                       FLAG_target]], dtype=np.float)
            self.FLAG_table = True
            return 0

        # duplication_checking
        idx_ast = np.where(self.table_id[:, 0] == info[2])[0]
        if len(idx_ast) > 1:
            assert False, 'Error: getAstPhotList.addast.duplication_checking'

        # [case 1] add new asteroid
        if len(idx_ast) == 0:
            self.table_id = np.concatenate((self.table_id, np.array([[info[2], info[0], info[1]]], dtype=np.str)))
            self.table_ph = np.concatenate((self.table_ph, np.array([info[3:11]], dtype=np.float)))
            self.table_fl = np.concatenate((
                self.table_fl, np.array([[info[11], info[12], info[13], info[14], info[15],
                                          info[16], info[17], info[18], info[19], info[20],
                                          FLAG_target]], dtype=np.float)))
            return 1

        # [case 2 & 3] update target asteroid
        elif FLAG_target == 0:  # 응, 타겟리스트에 있어
            # [case 2] if preceding data is not targeted observation, replace the data by new data.
            if self.table_fl[idx_ast[0], -1] != 0:   # 그런데 기존 관측은 타겟이 아니래 -> replace
                self.table_id[idx_ast[0]] = np.array([info[2], info[0], info[1]], dtype=np.str)
                self.table_ph[idx_ast[0]] = info[3:11].astype(np.float)
                self.table_fl[idx_ast[0]] = np.array([info[11], info[12], info[13], info[14], info[15],
                                                      info[16], info[17], info[18], info[19], info[20],
                                                      FLAG_target], dtype=np.float)
                return 2
            # [case 3] update to latest data.
            if int(info[0]) >= int(self.table_id[idx_ast[0], 1][0]):
                self.table_id[idx_ast[0]] = np.array([info[2], info[0], info[1]], dtype=np.str)
                self.table_ph[idx_ast[0]] = info[3:11].astype(np.float)
                self.table_fl[idx_ast[0]] = np.array([info[11], info[12], info[13], info[14], info[15],
                                                      info[16], info[17], info[18], info[19], info[20],
                                                      FLAG_target], dtype=np.float)
                return 3
            else:
                return 3

        # [case 4 & 5] update serend. asteroid
        elif FLAG_target == 1:  # 타겟리스트에 없어
            # [case 4] if preceding data is targeted observation, do not update.
            if self.table_fl[idx_ast[0], -1] == 0:   # 그런데 기존 관측은 타겟이래
                return 4
            # [case 5] update the data to averaged data
            elif self.table_fl[idx_ast[0], -1] != 0:
                self.table_id[idx_ast[0]] = np.array([info[2], info[0], info[1]], dtype=np.str)

                self.table_ph[idx_ast[0], ::2] = \
                    (self.table_ph[idx_ast[0], ::2] * self.table_fl[idx_ast[0], -1] +
                     info[3:11:2].astype(np.float)) / (self.table_fl[idx_ast[0], -1] + 1)
                self.table_ph[idx_ast[0], 1::2] = \
                    (((self.table_ph[idx_ast[0], 1::2] * self.table_fl[idx_ast[0], -1])**2 +
                      info[4:11:2].astype(np.float)**2)**0.5) / (self.table_fl[idx_ast[0], -1] + 1)

                self.table_fl[idx_ast[0], :-1] = \
                    ((self.table_fl[idx_ast[0], :-1] * self.table_fl[idx_ast[0], -1]) +
                     info[11:21].astype(np.float)) / (self.table_fl[idx_ast[0], -1] + 1)
                self.table_fl[idx_ast[0], -1] = self.table_fl[idx_ast[0], -1] + 1
                return 5
            else:
                assert False, 'Error: getAstPhotList.addast exception_situation'

        else:
            assert False, 'Error: getAstPhotList.addast exception_situation'

    def savetable(self, savename):
        table = np.concatenate((self.table_id, self.table_ph.astype(np.str), self.table_fl.astype(np.str)), axis=1)
        print(table[0])
        np.savetxt(savename, table, fmt='%s', delimiter='\t',
                   header='astID\tdate\t\tmode\tF1\t\t\tF1e\t\t\tF2\t\t\tF2e\t\t\t'
                          'F3\t\t\tF3e\t\t\tF4\t\t\tF4e\t\t\tN_obs')


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


griz = getAstPhotList()
N_obs_griz = 0
BVRI = getAstPhotList()
N_obs_BVRI = 0

for i in range(len(Datelist)):
    date = Datelist[i]
    type_filter = FilterFlag[i]
    #DirAst = '/data2/SHChoi/phot/obs' + date + '/KMTNet_2_stdzation/'
    #DirTar = '/data2/SHChoi/phot/targetlist/'
    DirAst = '/home/astro/PycharmProjects/KMTNet/FinalResults_new_BVRI_1111/'
    DirTar = '/home/astro/KMTNet/template/targetlist_final/'
    #table = np.genfromtxt(DirAst + 'AstPhot' + date + '.dat', dtype=np.str)
    if type_filter is True:    # griz
        continue
        table_tar = getTarInfoList_Flat(DirTar + 'asteroids_' + date + '_griz')
        N_obs_griz += len(table)
        for el in range(len(table)):
            griz.addast(table[el], table_tar)
    else:
        table = np.genfromtxt(DirAst + 'AstPhot' + date + '.dat', dtype=np.str)
        table_tar = getTarInfoList_Flat(DirTar + 'asteroids_' + date + '_BVRI')
        N_obs_BVRI += len(table)
        for el in range(len(table)):
            BVRI.addast(table[el], table_tar)
        
#griz.savetable('AstList_griz.cat')
BVRI.savetable('AstList_BVRI.cat')
print('griz')
#print('N_catalog\t%s' % len(griz.table_id))
#print('N_obs\t%s' % N_obs_griz)
#print('N_target\t%s' % sum(griz.table_fl[:, -1] == 0))
print('\n')
print('BVRI')
print('N_catalog\t%s' % len(BVRI.table_id))
print('N_obs\t%s' % N_obs_BVRI)
print('N_target\t%s' % sum(BVRI.table_fl[:, -1] == 0))

