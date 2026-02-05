# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import sys
import time
import torch
import torch.func as fc
import numpy as np
from joff._func._dvc import _2dvc
from joff._func._np import _2np


class Jacbi():
    def __init__(self, dynamics_fn, argnums):
        self.dynamics_fn = dynamics_fn
        self.argnums = argnums

    def batch_jac(self, *args):
        J = []
        for arg in zip(*args):
            J.append(self.jacobian(*arg))
        J = torch.stack(J)
        return J

    def jacobian(self, *args):
        J = torch.autograd.functional.jacobian(
            func=self.dynamics_fn,
            inputs=args,
            create_graph=True
        )[self.argnums]
        if J.dim() == 4:
            batch_size = args[self.argnums].size(0)
            J = J[torch.arange(batch_size), :, torch.arange(batch_size), :]
        return J

# 确保 dynamics_fn 是纯函数（所有变化的量都必须作为函数输入） -> 即输出完全由输入参数决定，不依赖外部状态，也没有副作用的函数。
# 只读的self的属性是允许的，但不允许在dynamics_fn中修改self的属性，可能造成赋值混乱
def vmap_jac(dynamics_fn, argnums, *args):
    """
    计算 dynamics_fn(*args) 对 *args 的偏导数（样本间互不干扰）
    Args:
        dynamics_fn: function
        argnums：指定要对dynamics_fn的第几个输入参数计算雅可比矩阵
        args: function 的批次输入，尺寸为 [batch_size, ？_dim]

    1. vmap（向量化映射）：将数据按批次维度自动拆开 -> 依次（并行）输入函数得到结果 -> 自动拼接
    将函数中的操作批量执行，从而避免显式循环，以提高计算效率。

    2. jacrev（反向模式雅可比计算）：计算函数的雅可比矩阵。使用反向自动微分来计算。
    对于函数 f: R^n -> R^m，jacrev(f) 返回一个函数，用于计算 f(x) 在 x 处的雅可比矩阵 [m x n]。

    3. vmap(jacrev(f))：获取雅可比矩阵计算函数 -> 用vmap将原函数变为批次并行函数 -> 获取样本间互不干扰的雅克比矩阵
    对于批次输入 x_batch: [batch, n]，vmap(jacrev(f))为每个样本计算一个雅可比矩阵，并自动拼接得到 [batch, m, n] 的雅可比矩阵。
    """

    # 单样本雅可比函数
    single_jac_fn = fc.jacrev(dynamics_fn, argnums=argnums)

    # vmap 批量处理
    # 明确指定所有输入的batch维度（in_dims）
    in_dims = tuple([0] * len(args))
    batch_jac_fn = fc.vmap(single_jac_fn, in_dims=in_dims, randomness='same')

    # 计算雅可比
    # 注意：args应该是批量输入 [batch_size, ...]
    with torch.no_grad():
        return batch_jac_fn(*args)


def _get_jcb_hess(self):
    _2dvc(self, self.dvc)
    if self.training: self.eval()

    loader = self.unshuf_train_loader if hasattr(self,'unshuf_train_loader') else self.train_loader
    I = _2dvc(torch.eye(loader.X.shape[1]), self.dvc)
    M, J, H = {}, {}, {}
    self.Frobenius_M, self.Frobenius_J, self.Frobenius_H = {}, {}, {}
    for vmap_mm in ['lv', 're']:
        M[vmap_mm], J[vmap_mm], H[vmap_mm] = [], [], []
    
    print()
    start = time.perf_counter()
    with torch.no_grad():
        for b_id, (x, l) in enumerate(loader):
            x, l = x.to(self.dvc), l.to(self.dvc)
            self._label = l

            if hasattr(self, 'mv_normal'):
                self._rd_test = self.mv_normal.sample(torch.Size([self.sample_times, x.size(0)])).to(self.dvc)

            for vmap_mm in ['lv', 're']:
                self._vmap_mm = vmap_mm
                mm = self.forward_for_vmap(x)
                jacobian = torch.vmap(torch.func.jacrev(self.forward_for_vmap, argnums=0))(x)
                hess = torch.vmap(torch.func.hessian(self.forward_for_vmap, argnums=0))(x)
                if vmap_mm == 're':
                    if len(jacobian.size()) == 4: jacobian, hess = jacobian.diagonal().permute(-1,0,1), hess.diagonal().permute(-1,0,1,2)
                    mm, jacobian = x - mm, I - jacobian

                if b_id == 0:
                    # and (not hasattr(self, 'if_simple_msg') or not self.if_simple_msg):
                    print('Size({}) = {}, Size(jacobian({})) = {}, Size(hess({})) = {}'.format(vmap_mm, mm.size(), vmap_mm, jacobian.size(), vmap_mm, hess.size()) )

                M[vmap_mm].append(_2np(mm))
                J[vmap_mm].append(_2np(jacobian))
                H[vmap_mm].append(_2np(hess))
            
            if (b_id+1) % 10 == 0 or (b_id+1) == len(loader):
                msg_str = 'Calculating batch gradients: {}/{}'.format(b_id+1, len(loader))
                sys.stdout.write('\r'+ msg_str + '                                    ')
                sys.stdout.flush()

    end = time.perf_counter()
    # if not hasattr(self, 'if_simple_msg') or not self.if_simple_msg:
    print('\nFinish batch gradient calculation, cost {} seconds'.format(int(end - start)))

    for vmap_mm in ['lv', 're']:
        M[vmap_mm], J[vmap_mm], H[vmap_mm] = np.concatenate(M[vmap_mm], 0), np.concatenate(J[vmap_mm], 0), np.concatenate(H[vmap_mm], 0)

        # self.Frobenius_M[vmap_mm] = np.abs(M[vmap_mm]).sum(axis=-1).mean()
        # self.Frobenius_J[vmap_mm] = np.abs(J[vmap_mm]).sum(axis=(-1,-2)).mean()
        # self.Frobenius_H[vmap_mm] = np.abs(H[vmap_mm]).sum(axis=(-1,-2,-3)).mean()
        self.Frobenius_M[vmap_mm] = np.abs(M[vmap_mm]).mean()
        self.Frobenius_J[vmap_mm] = np.abs(J[vmap_mm]).mean()
        self.Frobenius_H[vmap_mm] = np.abs(H[vmap_mm]).mean()

    # if not hasattr(self, 'if_simple_msg') or not self.if_simple_msg:
    print('\nAverage of F-norm(M(lv)) = {:.4f}, Average of F-norm(M(re)) = {:.4f}'.format(self.Frobenius_M['lv'], self.Frobenius_M['re']))
    print('Average of F-norm(J(lv)) = {:.4f}, Average of F-norm(J(re)) = {:.4f}'.format(self.Frobenius_J['lv'], self.Frobenius_J['re']))
    print('Average of F-norm(H(lv)) = {:.4f}, Average of F-norm(H(re)) = {:.4f}'.format(self.Frobenius_H['lv'], self.Frobenius_H['re']))