# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import numpy as np
from scipy.stats import gaussian_kde, foldnorm, chi2
from joff._func._msg import _show_detailed_msg
from joff._test._thrd import _get_thrd
from joff._load._split import _v2l
from joff._plot._heatmap import _matrix2heatmap, _score2heatmap
from joff._plot._view_score import _view_score_kde
from joff._save._save_ts_result import _save_ts

''' 
    ['re&lv-Q&T2-kde&ineq', 'cust_mm_XXX-T2-ineq', 'cust_ts_XXX-kde']
    
    p['pr_id'] = 1、2
    p['pr_name'] = re-Q-kde、lv-T2-kde、cust_mm_XXX-T2-ineq、cust_ts_XXX-kde
    p['pr_dt']['mm'] = re、lv、cust_mm_XXX
    p['pr_dt']['ts'] = Q、T2、cust_ts_XXX
    p['pr_dt']['thrd_meth'] = kde、ineq
'''
def fault_evaluation(self, In, Out, Latent, Label, phase, p):
    if 'cust_ts' in p['pr_dt']['ts']:
        # >>> use 'self._cust_ts' to record in forward
        TS = eval('self.' + phase + '_' + p['pr_dt']['ts'])
        # print(np.sort(TS, reverse = True))
        p['thrd_meth'] = p['pr_dt']['thrd_meth']
    else:
        # >>> use 'self._cust_mm' to record in forward
        if 'cust_mm' in p['pr_dt']['mm']: MM = eval('self.' + phase + '_' + p['pr_dt']['mm'])
        else: MM = _get_mm(In, Out, Latent, p['pr_dt']['mm'])
        if phase == 'offline': self._MM_off = MM.copy()
        else: self._MM_on = MM.copy()
        TS = _get_ts(self, MM, phase, p)
        p['thrd_meth'] = p['pr_dt']['thrd_meth']

    if self.ts_post_op is not None: TS = _cust_ts_post(self, TS, phase, p)
    save_path = self._save_path + '/' + p['pr_name'].replace('\\', '')
    if phase == 'offline':
        self._TS_off.append(TS)
        self._thrd.append(_get_thrd(self, p))
        if self.if_save_ts:
            _save_ts(TS, self._thrd[-1], save_path, phase)
    else:
        self._TS_on.append(TS)
        _Dr =_eval_fd(self, Label, p)
        if self.if_save_ts:
            _save_ts(TS, self._thrd[-1], save_path, phase)

    # plot kde
    if_plot_mm_kde = p['if_plot_mm_kde'][p['pr_id']] if type(p['if_plot_mm_kde'])==list else p['if_plot_mm_kde']
    if phase == 'offline' and if_plot_mm_kde and not p['if_simple_msg']:
        _Score_off = _get_score(self, self._MM_off, p)
        _view_score_kde(_Score_off, self._save_path, p['pr_name'].replace('\\',''))

    # plot score heatmap
    if_plot_score_hm = p['if_plot_score_hm'][p['pr_id']] if type(p['if_plot_score_hm'])==list else p['if_plot_score_hm']
    if phase == 'online' and if_plot_score_hm and not p['if_simple_msg']:
        _Score_on = _get_score(self, self._MM_on, p)
        # print(np.sqrt(np.sum((np.sum(_Score_on**2,-1) - TS)**2)))
        _Dr_list, _S_list, TS_list = _v2l(_Dr, self.test_loader.seg_len), _v2l(_Score_on, self.test_loader.seg_len), _v2l(TS, self.test_loader.seg_len)
        for i in range(len(_S_list)):
            _score2heatmap(_Dr_list[i], _S_list[i], TS_list[i], self._thrd[-1], self._save_path+'/'+p['pr_name'].replace('\\',''), '[score_hp] Fault {:02d}'.format(i+1))

def _get_mm(In, Out, Latent, mm):
    if mm == 're': return In - Out
    if mm == 'lv': return Latent

def _get_ts(self, MM, phase, p):
    ts = p['pr_dt']['ts']
    if ts == 'T2': return _cal_T2(self, MM, phase, p)
    if ts == 'Q': return _cal_Q(self, MM, phase)
    return MM.reshape(-1,)

def _get_score(self, MM, p):
    Centered_MM = MM - self._MM_mean if p['pr_dt']['ts'] in ['T2','Q'] else MM
    score = Centered_MM.copy()
    '''
        _T2_cov_sqrt_inv 是残差缩放的标尺。
        误报：变量方差越小，乘以逆缩得越少，对发生在此变量上的故障越敏感
        漏报：变量方差越大，乘以逆缩得越猛，对发生在此变量上的故障越不敏感
    '''
    if p['pr_dt']['ts'] == 'T2':
        score = score @ self._T2_cov_sqrt_inv
    return score

def _cal_T2(self, MM, phase, p):
    if phase == 'offline':
        self._MM_mean = np.mean(MM, axis = 0).reshape(1,-1)
    MM -= self._MM_mean
    if phase == 'offline':
        self._T2_cov = np.cov(MM, rowvar=False)
        _cal_cov_halfinv(self._T2_cov, self)
        if not p['if_simple_msg'] and p['if_plot_cov_inv'] and self._T2_cov_sqrt_inv.shape[0] < 50:
            _matrix2heatmap(self._T2_cov_sqrt_inv, self._save_path, '[T2_cov_sqrt_inv] ' + p['pr_name'])

    if self.if_use_lstsq:
        cov_inv_MMt = np.linalg.lstsq(self._T2_cov, MM.T, rcond=None)[0].T
        T2 = (MM[:, np.newaxis, :] @ cov_inv_MMt[:, :, np.newaxis]).reshape(-1, )
    else:
        T2 = (MM[:,np.newaxis,:] @ self._T2_cov_inv @ MM[:,:,np.newaxis]).reshape(-1,)

    return T2

def _cal_cov_halfinv(cov, self = None):
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigenvalues_inv = np.diag(np.power(eigenvalues, -1.0))
    _T2_cov_inv = eigenvectors @ eigenvalues_inv @ eigenvectors.T
    eigenvalues_sqrt_inv = np.diag(np.power(eigenvalues, -0.5))
    _T2_cov_sqrt_inv = eigenvectors @ eigenvalues_sqrt_inv @ eigenvectors.T
    if self is not None:
        self._T2_cov_inv, self._T2_cov_sqrt_inv = _T2_cov_inv, _T2_cov_sqrt_inv
    return _T2_cov_sqrt_inv

def _cal_Q(self, MM, phase):
    if phase == 'offline':
        self._MM_mean = np.mean(MM, axis = 0).reshape(1,-1)
    MM -= self._MM_mean
    Q = (MM[:,np.newaxis,:] @ MM[:,:,np.newaxis]).reshape(-1,)
    return Q

def _cust_ts_post(self, TS, phase, p):
    post_op = self.ts_post_op[p['pr_id']] if type(self.ts_post_op)==list else self.ts_post_op
    if post_op == False: return TS
    # TS_h = np.sqrt(TS)
    # if phase == 'offline':
    #     self._TS_mean = np.mean(TS_h)
    #     self._TS_std = np.std(TS_h)
    # return ((TS_h - self._TS_mean)/self._TS_std)**2

    if post_op == '-log':
        if phase == 'offline':
            self._TS_std = np.std(TS)
        return -np.log(TS/self._TS_std)

    # if phase == 'offline':
    #     print('\nfoldnorm:', foldnorm.fit(np.sqrt(TS)))     # c, loc(mean), scale(std)
    #     print('chi2:', chi2.fit(TS))                        # df, loc(mean), scale(std)
    #
    #     E, D = np.mean(TS), np.var(TS)
    #     sigma2_1, sigma2_2 = (2 + 4*E + np.sqrt((2+4*E)**2-16*D) ) / 8, (2 + 4*E - np.sqrt((2+4*E)**2-16*D) ) / 8
    #     u2_1, u2_2 = E - sigma2_1, E - sigma2_2
    #     e_1, e_2 = 2*sigma2_1 + 4*sigma2_1*u2_1 -D, 2*sigma2_2 + 4*sigma2_2*u2_2 -D
    #
    #     if u2_1 > 0 and u2_2 < 0: sigma2, u2 = sigma2_1, u2_1
    #     elif u2_1 < 0 and u2_2 > 0: sigma2, u2 = sigma2_2, u2_2
    #     elif np.abs(e_1) < np.abs(e_2): sigma2, u2 = sigma2_1, u2_1
    #     else: sigma2, u2 = sigma2_2, u2_2
    #
    #     self._TS_mean, self._TS_std = np.sqrt(u2), np.sqrt(sigma2)
    #     print('calculated:', u2_1, u2_2, self._TS_mean, self._TS_std)

    # return ((np.sqrt(TS)-self._TS_mean) / self._TS_std) ** 2

# TS < thrd? -> seg_len -> P == L? -> AFAR, AMDR, FAR_c, MDR_c
def _eval_fd(self, Label, p):
    # label
    L_list = _v2l(np.argmax(Label,1), self.test_loader.seg_len)
    TS, thrd = self._TS_on[-1], self._thrd[-1]

    # pred
    P = (TS > thrd).astype(int)
    P_list = _v2l(P, self.test_loader.seg_len)

    # state assertion / decision results
    Dr = np.zeros_like(P)
    if not p['if_simple_msg']:
        Normal_loc, Faulty_loc = np.where(np.argmax(Label,1)==0)[0], np.where(np.argmax(Label,1)!=0)[0]
        RN, RF, FA, MD = Normal_loc[np.where(P[Normal_loc]==0)[0]], Faulty_loc[np.where(P[Faulty_loc]==1)[0]], Normal_loc[np.where(P[Normal_loc]==1)[0]], Faulty_loc[np.where(P[Faulty_loc]==0)[0]]
        Dr[RF] = 1; Dr[FA] = 2; Dr[MD] = 3

    # for 'eval' and 'last'
    self._perf.append(_get_fd_perf(P_list, L_list))
    self._single_perf.append(self._perf[-1][1]) \
        if self._perf[-1][0] < self.expt_FAR * 100. else self._single_perf.append(100.)

    if not p['if_simple_msg']:
        print(self._test_msg)
        _show_detailed_msg(self, p['pr_id'])

    return Dr

# P list int record if > thrd
# L list int record fault type
def _get_fd_perf(P_list, L_list, debug = False):
    N_c = len(P_list)
    FAR, MDR = np.zeros((N_c,)), np.zeros((N_c,))
    S_FA, S_MD, S_Normal, S_Faulty = 0., 0., 0., 0.
    for c in range(N_c):
        prd, real = P_list[c], L_list[c]
        normal_loc, faulty_loc = np.where(real == 0)[0], np.where(real != 0)[0]
        N_normal, N_faulty = normal_loc.size, faulty_loc.size
        FA = np.sum((prd[normal_loc] == 1).astype(int))
        MD = np.sum((prd[faulty_loc] == 0).astype(int))
        S_FA += FA
        S_MD += MD
        S_Normal += N_normal
        S_Faulty += N_faulty
        if debug:
            print('c =',c)
            print('p =', prd); print('l =', real)
            print('(l = normal) =',normal_loc); print('(l = faulty) =',faulty_loc)
            print('N_normal =', N_normal, ', N_faulty =', N_faulty)
            print('p[l = normal] =', prd[normal_loc]); print('p[l = faulty] =', prd[faulty_loc])
            print('N_FA =',FA, ', N_MD =', MD,'\n')
        FAR[c] = FA / N_normal * 100 if N_normal > 0 else np.nan
        MDR[c] = MD / N_faulty * 100 if N_faulty > 0 else np.nan
    AFAR, AMDR = np.round(S_FA / S_Normal * 100, 2), np.round(S_MD / S_Faulty * 100, 2)
    return AFAR, AMDR, np.round(FAR, 2), np.round(MDR, 2)

if __name__ == '__main__':
    P, L = [], []
    for i in range(3):
        P.append( np.random.randint(0,2,20) )
        L.append( np.random.randint(0,3,20) )
    print(_get_fd_perf(P, L, True))