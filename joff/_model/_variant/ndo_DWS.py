# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from joff._model.arx import ARX, arx_defaults

# A negative-diagonal dominant observer
class NDO(ARX):
    def __init__(self, **kwargs):
        kwargs = dict(arx_defaults, **kwargs)
        ARX.__init__(self, **kwargs)

    def init_Sys_Matri(self):
        m_y = self.y_dim
        if not hasattr(self, 'C0_inv'):
            C0 = self.C[:,:m_y]
            # self.C0_inv = torch.inverse(C0)
            self.C0_inv = torch.diag(1./torch.diag(C0))

            # 分块矩阵
        if self.if_adp_P:
            if self.p_diag_log.device != self.dvc:
                self.p_diag_log = self.p_diag_log.to(self.dvc)
                self.P_21 = self.P_21.to(self.dvc)
            p_diag = torch.exp(self.p_diag_log)
            P_11_addi, P_22 = torch.diag(p_diag[:m_y]), torch.diag(p_diag[m_y:])
            P_12, P_21= self.P_21.T, self.P_21
            P22_inv = torch.diag(1. / p_diag[m_y:])
            # print(P_12.size(), P22_inv.size(), P_21.size())
            P_11 = P_12 @ P22_inv @ P_21 + P_11_addi
            S = P_11_addi
            self.S_inv = torch.diag(1. / p_diag[:m_y])
            self.P_L2 = - P22_inv @ P_21
            P_top = torch.cat((P_11, P_12), dim=-1)
            P_bottom = torch.cat((P_21, P_22), dim=-1)
            self.P = torch.cat((P_top, P_bottom), dim=0)
        else:
            P11 = self.P[:m_y, :m_y]
            P12 = self.P[:m_y, m_y:]
            P21 = self.P[m_y:, :m_y]
            P22 = self.P[m_y:, m_y:]
            P22_inv = torch.diag(1./torch.diag(P22))
            self.S_inv = torch.diag(1./torch.diag(P11))
            self.P_L2 = - P21

    # J [batch_size, x_dim, x_dim]
    def get_L(self, J):
        """
        根据稳定性条件计算观测器增益 L
        """
        # 获取雅克比矩阵 A: [batch_size, x_dim, x_dim]
        A = J.to(self.dvc)
        # print(f'A.size() = {A.size()}')
        # 计算 M0 = P A + A^T P
        if self.P.device != self.dvc:
            self.P, self.S_inv, self.C0_inv, self.P_L2, self.C = (self.P.to(self.dvc), self.S_inv.to(self.dvc),
                                                                  self.C0_inv.to(self.dvc), self.P_L2.to(self.dvc),
                                                                  self.C.to(self.dvc))
        m_y = self.y_dim
        # M0: [batch_size, x_dim, x_dim]
        M0 = self.P @ A + A.transpose(-2, -1) @ self.P
        # print(f'M0.size() = {M0.size()}')
        # 计算 T
        '''
            一维张量(1D) -> 二维对角阵(2D)     torch.diag(vec)         输入 1D 张量（N），返回 2D 对角矩阵（N,N）
            二维矩阵(2D) -> 一维对角线(1D)     torch.diag(mat)         输入 2D 矩阵（N,N），返回 1D 对角线张量（N）
            批次张量(2D) -> 批次对角阵(3D)     torch.diag_embed(vec)   输入 2D 张量（B,N），返回 3D 对角矩阵（B,N,N）
            批次矩阵(3D) -> 批次对角线(2D)     torch.diagonal(mat)     输入 3D 张量（B,N,N），返回 2D 张量（B,N）
        '''
        # M0_obs: [batch_size, y_dim, x_dim]
        M0_obs = M0[:, :m_y]
        # m0_11: [batch_size, y_dim]
        m0_11 = torch.diagonal(M0_obs, dim1=1, dim2=2)
        t = (m0_11 + torch.sum(M0_obs.abs(), dim=-1) - m0_11.abs()) / 2.0
        t_epsilon = 1e-3 # 最小正值
        t = torch.where(t > 0, t, t_epsilon) + self.neg_diag_thrd
        # T: [batch_size, y_dim, y_dim]
        T = torch.diag_embed(t)
        # print(f'T.size() = {T.size()}')

        # 计算 L1 和 L2
        L1 = self.S_inv @ T @ self.C0_inv
        L2 = self.P_L2 @ L1
        # print(f'L1.size() = {L1.size()}')
        # print(f'L2.size() = {L2.size()}')

        # 组合 L: [batch_size, x_dim, y_dim]
        L = torch.cat([L1, L2], dim=1)
        # print(f'L.size() = {L.size()}')
        N = self.P @ L @ self.C
        M = M0 - N - N.transpose(1, 2)

        self.M_stb_loss += self.neg_diag_loss(M)
        # self.M_unnobs = self.neg_diag_loss(M[y_dim:])

        return L

    def fix_with_L(self, y_error, J):
        L = self.get_L(J)
        # [batch_size, x_dim] = [batch_size, x_dim, y_dim] @ [batch_size, y_dim]
        L_ey = L @ y_error.unsqueeze(-1)
        # print(f'L_ey.size() = {L_ey.size()}')
        return L_ey.squeeze(dim=-1)


if __name__ == "__main__":
    import time
    from joff._run._runner import Runner

    stack = 6
    p = {  # FD设置
        'load_datas': [{'special': 'CSTR/fd', 'stack': stack, 'n_delay': 0},
                       {'special': 'TTS/fd', 'stack': stack, 'n_delay': 0},
                       {'special': 'HY/fd', 'stack': stack, 'n_delay': 0}],
        'fd_prs': [['cust_mm_ey-T2&Q-kde'],
                   ['cust_mm_ey-T2&Q-kde']
                   ],
        # 模型设置
        'models': ['NDO', 'DAE'],
        'structs': [[-1, 240, 120, 240, -1], [-1, '/3', '/1.5'], [-1, '/5', '*2']],
        'acts': [(1, ['s', 's', 's', 'a'], ['g', 'a', 'a']),
                 (2, ['g', 's', 'a'], ['a']),
                 (1, ['g', 't', 'a'], ['a']),
                 (2, ['g', 't', 'a'], ['a']),
                 (1, ['a', 'g', 'a'], ['s', 'a']),
                 (2, ['a', 'g', 'a'], ['s', 'a'])
                 ],
        # 特殊设置
        'expt_FAR': 0.5 / 100.,
        'plot_whole_fd': True,
        'drop_thrd': 100,
        'auto_drop': True,
        'opt': 'Adam',
        'lr': 1e-4,
        # NDO paprameters
        'x_dim': 50,
        'u_dim': 3,
        'y_dim': 7,
        'n_net': 1,
        'neg_diag_thrd': 0.1,
        'if_adp_P': True,
        'if_calc_jac': True,
        'if_allow_drop': False,
        'alf': (1., 0.1),
        'unobs_stb': 'f_norm'
    }

    R = Runner(**p)

    model = R._get_model(1, 1, 1)
    model.run(e=24, b=3, run_times=1)
