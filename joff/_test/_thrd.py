# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import numpy as np
from scipy import interpolate
from scipy.stats import gaussian_kde
from joff._func._msg import _msg_code

def _get_thrd(self, p):
    self._test_msg = "\n{}{}) '{}':\033[0m".format(_msg_code('pi'), p['pr_id']+1, p['pr_name'])
    if p['thrd_meth'] == 'ineq':
        thrd = _get_ineq_thrd(self, p)
    if p['thrd_meth'] == 'kde':
        thrd = _get_kde_thrd(self, p)
    if p['thrd_meth'] == 'pdf':
        thrd = self._get_pdf_thrd(p)    # customize
    self._test_msg += '\nThreshold = {}{:.2f}\033[0m'.format(_msg_code('ly',False), np.round(thrd,2))
    return thrd

def _get_ineq_thrd(self, p):
    TS = self._TS_off[-1]
    sorted_TS = np.sort(TS)
    alld_error, delta = p['alld_error'], 1 - p['cl']
    
    # select thrd
    N = sorted_TS.size
    gama = N * (1 - p['expt_FAR'])
    k = int(np.floor(gama))
    
    if p['if_intp']: _thrd = sorted_TS[k-1] + (gama - k) * (sorted_TS[k] - sorted_TS[k-1])
    else: _thrd = sorted_TS[k-1]
    
    # S2 = Σ[TS - E[TS]]^2, LB = - beta = lower bound of 'TS - E[TS]' = - expt_FAR
    S2, beta = np.sum(((TS>_thrd).astype(int) - p['expt_FAR'])**2), p['expt_FAR']
    log_2dt = np.log(2/delta)
    self._test_msg += "\n\nSample size: N = {}".format(N)
    
    # Two-sided Chernoff bound
    N_min = log_2dt/(2*alld_error**2)
    self._test_msg += "\nChernoff's minimum requirements: N >= {}".format(int(np.ceil(N_min)))
    
    # Freedman inequalities with [ΔM], ΔM >= -lb, lb > 0
    # exp(-(Nε)^2/(2*(v2+Nεb))) <= δ
    N_min = (log_2dt*beta + np.sqrt((log_2dt*beta)**2 + 2*S2*log_2dt)) / alld_error
    self._test_msg += "\nFreedman's minimum requirements: N >= {}".format(int(np.ceil(N_min)))
    
    # De la Peña inequalities with [ΔM]
    # exp(-(Nε)^2/(2*v2)) <= δ
    N_min = np.sqrt(2*S2*log_2dt) / alld_error
    self._test_msg += "\nDe la Pena's minimum requirements: N >= {}".format(int(np.ceil(N_min)))
    
    # Bernstein inequalities with [ΔM], ΔM >= -1
    # (1+Nε/v2)^v2*exp(-Nε) <= δ
    N_min = fast_Bers_thrd(alld_error, S2, delta)
    self._test_msg += "\nBernstein's minimum requirements: N >= {}".format(N_min)
    
    return _thrd

# fast search Bernstein thrd
def fast_Bers_thrd(alld_error, S2, delta):
    def Bernstein(_N):
        return (1+_N*alld_error/S2)**S2 * np.exp(-_N*alld_error) - delta
    left_N, move_step = 0, 50
    while Bernstein(left_N) > 0: left_N += move_step
    for N in range(left_N - move_step, left_N):
        if Bernstein(N+1) < 0:
            return N+1

# def _get_kde_thrd(self, p):
#     TS = self._TS_off[-1]
#     print(TS.shape, TS.max(), TS.min(), TS.mean())
#
#     # create kde
#     kde = gaussian_kde(TS)
#
#     # 'np.unique' returns an array with no duplicate elements from smallest to largest
#     x = np.unique(TS)
#
#     # evaluate pdfs
#     kde_pdf = kde.evaluate(x)
#
#     # estimate cdf
#     cdf_func = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf, x))
#     cdf = cdf_func(x)
#
#     # estimate inv cdf
#     isf_curve = interpolate.InterpolatedUnivariateSpline(x, 1-cdf-p['expt_FAR'])
#     _thrd = isf_curve.roots()
#
#     # multiple solutions
#     if type(_thrd) == np.ndarray:
#         self._test_msg += "\n\nSolutions = " +  str(np.round(_thrd,2))
#         _thrd = np.min(_thrd)
#
#     return _thrd

def _get_kde_thrd(self, p):
    TS = self._TS_off[-1]

    # 使用原始TS的分位数作为备选方案
    sorted_ts = np.sort(TS)
    n = len(sorted_ts)

    # 寻找阈值：使得 P(X > thrd) = expt_FAR
    # 即 1 - CDF(thrd) = expt_FAR  => CDF(thrd) = 1 - expt_FAR
    target_cdf = 1 - p['expt_FAR']
    # 计算分位数位置
    idx = target_cdf * (n - 1)

    # 线性插值（类似于不等式解）
    lower_idx, upper_idx = int(np.floor(idx)), int(np.ceil(idx))
    lower_val, upper_val = sorted_ts[lower_idx], sorted_ts[upper_idx]
    weight = idx - lower_idx
    _thrd0 = lower_val * (1 - weight) + upper_val * weight

    # 构建x数组：包含两部分
    # 1. 等间距的基础部分
    n_points = np.min([TS.shape[0] // 2, 3000])
    x_base = np.linspace(sorted_ts[0] * 0.8, sorted_ts[-1] * 1.2, n_points)

    # 2. 取接近 target_cdf 正负 2% 范围内的点
    near_range = (sorted_ts[-1] - sorted_ts[0]) * 0.02

    # 确定窗口边界
    near_lower = max( _thrd0 - near_range, sorted_ts[0])
    near_upper = min( _thrd0 + near_range, sorted_ts[-1])

    # 密集区域点
    x_dense = sorted_ts[ np.where( (sorted_ts >= near_lower) & (sorted_ts <= near_upper) )[0] ]

    # 合并并去重排序
    x = np.unique(np.concatenate([x_base, x_dense]))

    # 创建KDE
    kde = gaussian_kde(TS)

    # 计算PDF
    pdf = kde.evaluate(x)

    # 使用梯形法进行数值积分（处理不等间距）
    cdf = np.zeros_like(x)
    for i in range(1, len(x)):
        # 计算当前小梯形的面积
        dx = x[i] - x[i - 1]
        avg_pdf = (pdf[i] + pdf[i - 1]) / 2
        cdf[i] = cdf[i - 1] + avg_pdf * dx

    # 归一化
    cdf = cdf / cdf[-1]

    # 插值求根
    cdf_func = interpolate.InterpolatedUnivariateSpline(x, cdf - target_cdf)

    # 寻找所有根
    roots = cdf_func.roots()

    # 没有根
    if len(roots) == 0:
        # 记录使用备选方案
        if hasattr(self, '_test_msg'):
            self._test_msg += f"\n\nNo KDE root found, using empirical quantile: {_thrd0:.2f}"
        return _thrd0

    # 多个根
    if len(roots) > 1:
        # 多个根时，选择最接近目标CDF的根
        # 计算每个根对应的实际CDF值
        root_cdfs = []
        for root in roots:
            # 在cdf中查找最接近的值
            idx_closest = np.argmin(np.abs(x - root))
            if idx_closest < len(cdf):
                root_cdfs.append(cdf[idx_closest])
            else:
                root_cdfs.append(1.0)  # 如果在末尾，设为1

        # 选择最接近目标CDF的根
        idx_best = np.argmin(np.abs(np.array(root_cdfs) - target_cdf))
        _thrd = roots[idx_best]

        self._test_msg += f"\n\nMultiple solutions = {np.round(roots, 2)}, selected = {_thrd:.2f}"
        return _thrd

    # 唯一根
    return roots[0]