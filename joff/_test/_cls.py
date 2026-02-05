# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

def _test_cls(self, Y, L, p):
    pass

def get_accuracy(self, output, target):
    # print(output.shape, target.shape)
    if np.isnan(target).astype(int).sum() > 0:
        # 标签中存在缺失值
        dim = target.shape[1]
        loc = np.where(target == target)
        target = target[loc]
        output, target = output.reshape(-1, dim), target.reshape(-1, dim)

    if len(target.shape) > 1:
        output_arg = np.argmax(output, 1)
        target_arg = np.argmax(target, 1)
    else:
        output_arg = np.array(output + 0.5, dtype=np.int)
        target_arg = np.array(target, dtype=np.int)

    return np.mean(np.equal(output_arg, target_arg).astype(np.float)) * 100

def get_FDR(self, output, target):
    '''
        正分率:
        FDR_i = pred_cnt[i][i] / n_sample_cnts[i]

        误分率:
        FPR_i = ∑_j(pred_cnt[i]),j ≠ i / ∑_j(n_sample_cnts),j ≠ i
    '''
    if hasattr(self, 'FDR') == False:
        self.statistics_number(target)
    if len(target.shape) > 1:
        output_arg = np.argmax(output, 1)
        target_arg = np.argmax(target, 1)

    pred_cnt = np.zeros((self.n_category, self.n_category))
    for i in range(self.n_sample):
        # 第 r 号分类 被 分到了 第 p 号分类
        p = output_arg[i]
        r = target_arg[i]
        pred_cnt[p][r] += 1
    pred_cnt_pro = pred_cnt / self.n_sample_cnts * 100
    # array是一个1维数组时，形成以array为对角线的对角阵；array是一个2维矩阵时，输出array对角线组成的向量
    FDR = np.diag(pred_cnt_pro)
    FPR = [(self.n_sample_cnts[i] - pred_cnt[i][i]) /
           (self.n_sample - self.n_sample_cnts[i]) * 100 for i in range(self.n_category)]

    self.pred_distrib = [pred_cnt, np.around(pred_cnt_pro, 2)]
    for i in range(self.n_category):
        self.FDR[i][0], self.FDR[i][1] = FDR[i], FPR[i]

    self.FDR[-1][0] = self.get_accuracy(output, target)
    self.FDR[-1][1] = 100 - self.FDR[-1][0]
    assert np.sum(np.diag(pred_cnt)) / np.sum(pred_cnt) * 100 == self.FDR[-1][0]
    self.FDR = np.around(self.FDR, 2)