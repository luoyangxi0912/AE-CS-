# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from joff._nn._fcnn_module import FCNN
from joff._model.nice import NICE
from joff._load._load_para import _load_module_para

class PFBO(FCNN):
    def __init__(self, **kwargs):
        """
        nn:
            H_E: (U_{k−p:k−1}, Y_{k−p+1:k}) -> (^x_k, ^U_{k−p:k−1})
            H_P: (^x_k, U_{k:k+q-1}) -> (^Y_{k+1:k+q}, ^U_{k:k+q-1})
            H_P^{-1}: (U_{k:k+q-1}, Y_{k+1:k+q}) -> (^x_k, ^U_{k:k+q-1})
            H_E^{-1}: (^x_k, U_{k−p:k−1}) -> (^Y_{k−p+1:k}, ^U_{k−p:k−1})
            input sample: (U_{k−p:k+q-1}, Y_{k−p+1:k+q})
        Args:
            p: 历史窗口大小
            q: 预测窗口大小
            u_dim: 输入维度
            y_dim: 输出维度
            x_dim: 状态维度
        """
        # 设置默认参数
        defaults = {
            'p': 5,
            'q': 5,
            'u_dim': 9,
            'y_dim': 52-9,
            'x_dim': 100,
            'if_extend_u': False,           # 是否在HE和HP中扩展U
            'struct': [-1, '/4', '*1'],     # HE 中 ACL 的结构
            'act': ['t','a'],               # HE 中 ACL 的激活函数
            'hp_struct': [-1, '/4', '*1'],  # HP 中 ACL 的结构
            'hp_act': ['t','a'],            # HP 中 ACL 的激活函数
            'num_ACL': 3,
            'addi_learn': 'x',
            'addi_mm': 'x',
            'if_nice_loss': False           # 是否计算完全的NICE loss
        }
        kwargs = dict(defaults, **kwargs)

        # H_E 模块的输入输出维度
        self.he_input_dim = kwargs['p'] * (kwargs['u_dim'] + kwargs['y_dim'])  # U_{k−p:k−1}, Y_{k−p+1:k}
        self.he_output_dim = kwargs['x_dim']  # \hat x_k
        if kwargs['if_extend_u']: self.he_output_dim += kwargs['p'] * kwargs['u_dim']   # \hat x_k, \hat U_{k−p:k−1}
        kwargs['struct'] = [self.he_input_dim] + kwargs['struct'][1:-1] + [self.he_input_dim]

        # H_P 模块的输入输出维度（逆向模型）
        self.hp_input_dim = kwargs['q'] * kwargs['y_dim']  # \hat Y_{k+1:k+q}
        if kwargs['if_extend_u']: self.hp_input_dim += kwargs['q'] * kwargs['u_dim']  # \hat Y_{k+1:k+q}, \hat U_{k:k+q-1}
        self.hp_output_dim = kwargs['x_dim'] + kwargs['q'] * kwargs['u_dim']    # \hat x_k, U_{k:k+q-1}

        kwargs['hp_struct'] = [self.hp_input_dim] + kwargs['hp_struct'][1:-1] + [self.hp_input_dim]

        if kwargs['if_nice_loss']:
            kwargs['view_addi_info'] = ['forward_loss', 'reverse_loss', 'he_loss', 'hp_loss', 'loss']
        else:
            kwargs['view_addi_info'] = ['forward_loss', 'reverse_loss', 'he_w_inv_loss', 'hp_w_inv_loss', 'loss']
            if self.he_input_dim == self.he_output_dim: kwargs['view_addi_info'].remove('he_w_inv_loss')
            if self.hp_input_dim == self.hp_output_dim: kwargs['view_addi_info'].remove('hp_w_inv_loss')
        if 'x' in kwargs['addi_learn']: kwargs['view_addi_info'].insert(-1, 'x_recon_loss')
        if 'u' in kwargs['addi_learn']: kwargs['view_addi_info'].insert(-1, 'u_recon_loss')

        # 初始化FCNN
        FCNN.__init__(self, **kwargs)

        # 创建可逆模块（状态估计器）
        kwargs['struct'], kwargs['act'] = self.struct, self.act
        self.H_E = NICE(
            output_dim = self.he_output_dim,
            **kwargs
        )

        # 创建可逆模块（观测预测器）
        kwargs['struct'], kwargs['act'] = self.hp_struct, self.hp_act
        self.H_P = NICE(
            output_dim = self.hp_output_dim,
            **kwargs
        )

        # 优化器
        self.opti()

    def forward(self, x):
        """
        前向传播
        Args:
            H_E: (U_{k−p:k−1}, Y_{k−p+1:k}) -> (^ x_k, ^ U_{k−p:k−1})
            H_P: (^ x_k, U_{k:k+q-1}) -> (^ Y_{k+1:k+q}, ^ U_{k:k+q-1})
            H_P^{-1}: (U_{k:k+q-1}, Y_{k+1:k+q}) -> (^ x_k, ^ U_{k:k+q-1})
            H_E^{-1}: (^ x_k, U_{k−p:k−1}) -> (^ Y_{k−p+1:k}, ^ U_{k−p:k−1})
            input sample: (U_{k−p:k+q-1}, Y_{k−p+1:k+q})
            label: fault type
        """
        batch_size = x.shape[0]
        X = x.reshape(batch_size, self.p + self.q, self.u_dim + self.y_dim)

        he_U_Y = X[:, :self.p, :].reshape(batch_size, -1)
        he_U = X[:, :self.p, :self.u_dim].reshape(batch_size, -1)
        he_Y = X[:, :self.p, self.u_dim:].reshape(batch_size, -1)

        hp_U_Y = X[:, self.p:, :].reshape(batch_size, -1)
        hp_U = X[:, self.p:, :self.u_dim].reshape(batch_size, -1)
        hp_Y = X[:, self.p:, self.u_dim:].reshape(batch_size, -1)

        # H_E 前向（正向）
        he_forward = self.H_E(he_U_Y)
        if self.if_extend_u: he_x_hat, he_fU_hat = he_forward[:, :self.x_dim], he_forward[:, self.x_dim:]
        else: he_x_hat = he_forward

        # H_P 前向（逆向）
        hp_forward = self.H_P(torch.cat([he_x_hat, hp_U], dim=1), reverse=True)
        if self.if_extend_u: hp_fU_hat, hp_Y_hat = hp_forward[:, :int(self.u_dim * self.q)], hp_forward[:, int(self.u_dim * self.q):]
        else: hp_Y_hat = hp_forward

        # H_P 反向（正向）
        if self.if_extend_u: hp_reverse = self.H_P(hp_U_Y)
        else: hp_reverse = self.H_P(hp_Y)
        hp_x_hat, hp_rU_hat = hp_reverse[:, :self.x_dim], hp_reverse[:, self.x_dim:]

        # H_E 反向（逆向）
        if self.if_extend_u: he_reverse = self.H_E(torch.cat([hp_x_hat, he_U], dim=1), reverse=True)
        else: he_reverse = self.H_E(hp_x_hat, reverse=True)
        he_rU_hat, he_Y_hat = he_reverse[:, :int(self.u_dim * self.p)], he_reverse[:, int(self.u_dim * self.p):]
        # print(he_fU_hat.shape, he_U.shape, hp_fU_hat.shape, hp_U.shape, hp_Y_hat.shape, hp_Y.shape)

        # 计算前向与反向Y的误差损失
        self.forward_loss = torch.sum((hp_Y_hat - hp_Y) ** 2) * 1.0 / batch_size
        self.reverse_loss = torch.sum((he_Y_hat - he_Y) ** 2) * 1.0 / batch_size

        # 计算前向与反向U的重构损失
        if self.if_extend_u and 'u' in self.addi_learn:
            fU_recon = (torch.sum((he_fU_hat - he_U) ** 2) + torch.sum((hp_fU_hat - hp_U) ** 2) ) * 0.1 / batch_size
            rU_recon = (torch.sum((hp_rU_hat - hp_U) ** 2) + torch.sum((he_rU_hat - he_U) ** 2) ) * 0.1 / batch_size
            self.u_recon_loss = fU_recon + rU_recon
            self.forward_loss += fU_recon
            self.reverse_loss += rU_recon
        self.loss = self.forward_loss + self.reverse_loss

        # 获取x的重构损失
        if 'x' in self.addi_learn:
            self.x_recon_loss = torch.sum((he_x_hat - hp_x_hat) ** 2) * 1.0 / batch_size
            self.loss += self.x_recon_loss

        # 获取W的可逆损失/NICE损失
        if not self.if_nice_loss:
            if self.he_input_dim != self.he_output_dim: self.he_w_inv_loss = self.H_E.w_inv_loss
            if self.hp_input_dim != self.hp_output_dim: self.hp_w_inv_loss = self.H_P.w_inv_loss
            if hasattr(self, 'he_w_inv_loss'): self.loss += 10 * self.he_w_inv_loss
            if hasattr(self, 'hp_w_inv_loss'): self.loss += 10 * self.hp_w_inv_loss
        else:
            self.he_loss, self.hp_loss = self.H_E.loss, self.H_P.loss
            self.loss += self.H_E.loss + self.H_P.loss

        # 保存统计量
        if self.if_extend_u and 'u' in self.addi_learn and 'u' in self.addi_mm:
            self._cust_mm = {'cust_mm_hp': torch.cat([hp_fU_hat - hp_U, hp_Y_hat - hp_Y], dim=1),
                             'cust_mm_he': torch.cat([he_rU_hat - he_U, he_Y_hat - he_Y], dim=1),
                             'cust_mm_cat': torch.cat([hp_fU_hat - hp_U, he_rU_hat - he_U,
                                                       hp_Y_hat - hp_Y, he_Y_hat - he_Y], dim=1)}
        else:
            self._cust_mm = {'cust_mm_hp': hp_Y_hat - hp_Y,
                             'cust_mm_he': he_Y_hat - he_Y,
                             'cust_mm_cat': torch.cat([hp_Y_hat - hp_Y, he_Y_hat - he_Y], dim=1)}
        if 'x' in self.addi_learn and 'x' in self.addi_mm:
            self._cust_mm['cust_mm_x'] = he_x_hat - hp_x_hat
            self._cust_mm['cust_mm_catx'] = torch.cat([self._cust_mm['cust_mm_cat'],
                                                       self._cust_mm['cust_mm_x']], dim=1)
        # return hp_forward

class PFBO_NR(FCNN):
    def __init__(self, **kwargs):
        # 设置默认参数
        defaults = {
            'if_extend_u': False,
            'p': 5,
            'q': 5,
            'u_dim': 9,
            'y_dim': 52 - 9,
            'struct_nr': [-1, '/4', '*1'],     # HE 中 ACL 的结构
            'act_nr': ['s','a'],               # HE 中 ACL 的激活函数
            'num_ACL_nr': 3,
            'view_addi_info': ['nr1_loss', 'nr2_loss', 'loss']
        }
        # 更新defaults
        _kwargs = dict(defaults, **kwargs)
        # 初始化FCNN
        FCNN.__init__(self, **_kwargs)

        self.pfbo = PFBO(**kwargs)
        # 加载最后保存的参数
        _load_module_para(self.pfbo, 'last', replace = ('PFBO_NR', 'PFBO'))

        # 创建可逆模块（状态估计器）
        kwargs['num_ACL'] = slef.num_ACL_nr
        if self.if_extend_u: self.struct_nr[0] = int(self.u_dim * (self.p + self.q))
        else: self.struct_nr[0] = int(self.y_dim * self.q)
        kwargs['struct'], kwargs['act'] = self.struct_nr, self.act_nr
        self.nr_dim = self.struct_nr[0]
        self.nr1 = NICE(**kwargs)

        if self.if_extend_u: self.struct_nr[0] = int(self.y_dim * (self.p + self.q))
        else: self.struct_nr[0] = int(self.y_dim * self.p)
        kwargs['struct'], kwargs['act'] = self.struct_nr, self.act_nr
        self.nr2 = NICE(**kwargs)

    def forward(self, x):
        self.pfbo(x)
        mm_cat = self.pfbo._cust_mm['cust_mm_cat'].detach()
        nr1_input, nr2_input = mm_cat[:, :self.nr_dim], mm_cat[:, self.nr_dim:]
        nr1_output, nr2_output = self.nr1(nr1_input), self.nr2(nr2_output)
        self._cust_mm = {'cust_mm_hp': nr1_output,
                         'cust_mm_he': nr2_output,
                         'cust_mm_cat': torch.cat([nr1_output, nr2_output], dim=1)}
        self.nr1_loss, self.nr2_loss = self.nr1.loss, self.nr2.loss
        self.loss = self.nr1_loss + self.nr2_loss

if __name__ == '__main__':
    import time
    from joff._run._runner import Runner
    p = {# FD设置
         'load_datas': [{'special': 'CSTR/fd', 'stack': 10},
                        {'special': 'HY/fd', 'stack': 10}],
         'fd_prs': [['cust_mm_hp&cust_mm_x&cust_mm_cat&cust_mm_catx-T2&Q-kde'],
                    ['re-T2&Q-kde'],
                    ['re-T2&Q-kde']
                    ],
         # 模型设置
         'models': ['PFBO', 'DAE', 'VAE'],
         'structs': [[-1, '/2', '/2'], [-1, '/3', '/1.5'], [-1, '/5', '*2']],
         'acts': [(1, ['g', 's', 'g'], ['g', 'a', 'a']),
                  (2, ['g', 's'], ['a']),
                  (1, ['g', 't'], ['a']),
                  (2, ['g', 't'], ['a']),
                  (1, ['a', 'g'], ['s', 'a']),
                  (2, ['a', 'g'], ['s', 'a'])
                  ],
         # 特殊设置
         'expt_FAR': 2 /100.,
         'plot_whole_fd': True,
         'drop_thrd': 150,
         'auto_drop': True,
         'opt': 'Adam',
         'lr': 5e-4,
         'p': 5,
         'q': 5,
         'x_dim': (52 - 9) * 5,
         'hp_struct': [-1, '/2', '*1'],
         'hp_act': ['g', 'a', 'a'],
         'num_ACL': 3,
         # 'addi_learn': ['x', 'u'],
         'addi_learn': 'x',
         'addi_mm': 'x',
         'if_extend_u': True,
         'if_nice_loss': False
        }

    R = Runner(**p)
    run_act = 0
    if run_act:
        for act_id in range(len(p['acts'])):
            model = R._get_model(2,2, act_id + 1)
            model.run(e = 24, run_times = 1)
            del model
            time.sleep(1.0)
    else:
        model = R._get_model(2, 1, 1)
        model.run(e=1, run_times = 1)

    # 'HY/selected1_fd', 'stack': 10
    # <- PFBO ->
    # gsg, gaa (120): 0.69, 6.0
    # <- DAE ->
    # gsg, gaa (120): 0.83, 28.52

    # 'HY/selected3_fd', 'stack': 10
    # <- PFBO ->
    # gsg, gaa: 0.23, 8.72
    # gsg, gaa (120): 0.62, 2.3
