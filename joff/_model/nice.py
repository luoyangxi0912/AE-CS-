# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from joff._nn._fcnn_module import FCNN

class AdditiveCouplingLayer(FCNN):
    """加性耦合层"""

    def __init__(self, **kwargs):
        defaults = {
            'up': 0,
            'struct': None,
            'act': None
        }
        kwargs = dict(defaults, **kwargs)
        FCNN.__init__(self, **kwargs)

        # 输入维度
        self.input_dim = self.struct[0]
        self.split_dim = self.struct[0] // 2
        if self.up == 0:
            self.struct[0] = self.split_dim
            self.struct[-1] = self.input_dim - self.split_dim
        else:
            self.struct[0] = self.input_dim - self.split_dim
            self.struct[-1] = self.split_dim
        self.act[-1] = 'a'

        # 耦合网络
        self.net = self.Seq()

    def forward(self, x, reverse=False):
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        if not reverse:
            # 前向传播
            if self.up == 0:
                shift = self.net(x1)
                y2 = x2 + shift
                return torch.cat([x1, y2], dim=1)
            else:
                shift = self.net(x2)
                y1 = x1 + shift
                return torch.cat([y1, x2], dim=1)
        else:
            # 逆向传播
            if self.up == 0:
                shift = self.net(x1)
                y2 = x2 - shift
                return torch.cat([x1, y2], dim=1)
            else:
                shift = self.net(x2)
                y1 = x1 - shift
                return torch.cat([y1, x2], dim=1)

def init_weights_SVD(W1, W2):
    """
    W1: (d, k) 从Z(d维)到X(k维)
    W2: (k, d) 从X(k维)到Z(d维)
    根据k<d或k>d分别初始化，逼近伪逆关系
    """
    d, k = W1.size(0), W1.size(1)
    if k < d:
        # 情况1: X维度(k) < Z维度(d)
        # 我们希望 W1^T W1 ≈ I_k (左伪逆结构)
        # 初始化为随机正交基

        # 1. 生成随机正交矩阵Q (d, k)，列正交
        Q, _ = torch.linalg.qr(torch.randn(d, k))

        # 2. 设置W1 = Q (d, k)，满足W1^T W1 = I_k
        W1.data.copy_(Q)

        # 3. 设置W2 = Q^T (k, d)，作为W1的左逆近似
        W2.data.copy_(Q.T)

        # 此时：W2 W1 = Q^T Q = I_k，但W1 W2 = Q Q^T (秩k，非单位阵)

    else:  # k > d
        # 情况2: X维度(k) > Z维度(d)
        # 我们希望 W2 W2^T ≈ I_d (右伪逆结构)

        # 1. 生成随机正交矩阵P (k, d)，列正交
        P, _ = torch.linalg.qr(torch.randn(k, d))

        # 2. 设置W2 = P (k, d)，满足W2^T W2 = I_d
        W2.data.copy_(P)

        # 3. 设置W1 = P^T (d, k)，作为W2的右逆近似
        W1.data.copy_(P.T)

        # 此时：W1 W2 = P^T P = I_d，但W2 W1 = P P^T (秩d，非单位阵)

    # 添加小噪声打破对称性，让网络有学习空间
    W1.data += 0.01 * torch.randn_like(W1)
    W2.data += 0.01 * torch.randn_like(W2)

    return W1, W2

class NICE(FCNN):
    """可逆模块，由多个加性耦合层组成"""

    def __init__(self, **kwargs):
        defaults = {
            'struct': None,
            'act': None,
            'output_dim': None,
            'num_ACL': 3,
            'view_addi_info': ['nlog_loss', 'w_inv_loss', 'loss'],
            'if_svd_init': True
        }
        kwargs = dict(defaults, **kwargs)
        FCNN.__init__(self, **kwargs)

        # 创建 ACL list
        self.layers = nn.ModuleList()

        # 创建多个耦合层，交替分割维度
        _cnt = 0
        _act = self.act.copy()
        for i in range(self.num_ACL):
            kwargs['struct'] = self.struct.copy()
            kwargs['act'] = [ _act[ (_cnt + k)%len(_act) ] for k in range(len(self.struct)-2) ] + ['a']
            _cnt += len(self.struct)-2
            self.layers.append(AdditiveCouplingLayer(up = i % 2, **kwargs))
        # 添加尺度缩放向量
        self.scaling_factor = nn.Parameter(torch.ones(1, self.struct[0]))
        if self.output_dim != self.struct[0]:
            # 添加伪逆矩阵
            # X = Z @ W1, Z = X @ W2
            self.W1 = nn.Parameter(torch.randn(self.struct[0], self.output_dim) * 0.1)
            self.W2 = nn.Parameter(torch.randn(self.output_dim, self.struct[0]) * 0.1)

            if self.if_svd_init:
                self.W1, self.W2 = init_weights_SVD(self.W1, self.W2)
            self.I = torch.eye(min(self.struct[0],self.output_dim))

        self.opti()

    def forward(self, x, reverse=False):
        if not reverse:
            # 按奇偶重排
            x = torch.cat([x[:, 0::2], x[:, 1::2]], dim=1)

            # 前向传播
            for layer in self.layers:
                x = layer(x)
            x *= torch.exp(self.scaling_factor)
            self.nlog_loss = torch.sum(x ** 2) / 2.0 - torch.sum(self.scaling_factor)
            self.loss = self.nlog_loss
            if self.output_dim != self.struct[0]:
                _z = x.detach()
                _z_hat = _z @ self.W1 @ self.W2
                recon_loss = torch.mean(torch.sum((_z - _z_hat) ** 2, dim=-1), dim=0)
                # 降维
                if self.struct[0] > self.output_dim:
                    eye_loss = torch.norm(self.W2 @ self.W1 - self.I.to(self.dvc), p='fro') ** 2 / self.I.shape[0]
                # 升维
                else:
                    # _z = x.detach()
                    # _x = (_z @ self.W1).detach()
                    # _x_hat = _x @ self.W2 @ self.W1
                    # recon_loss = torch.mean(torch.sum((_x - _x_hat) ** 2, dim=-1), dim=0)
                    eye_loss = torch.norm(self.W1 @ self.W2 - self.I.to(self.dvc), p='fro') ** 2 / self.I.shape[0]

                self.w_inv_loss = recon_loss + eye_loss
                self.loss += self.w_inv_loss
                x = x @ self.W1
        else:
            # 逆向传播
            if self.output_dim != self.struct[0]:
                x = x @ self.W2
            x *= torch.exp(-self.scaling_factor)

            for layer in reversed(self.layers):
                x = layer(x, reverse=True)

            # 按奇偶逆向重排
            x_dim = x.size(-1)
            half_x_dim = (x_dim + 1) // 2

            # 创建输出张量
            restored = torch.zeros_like(x, device=x.device, dtype=x.dtype)

            # 交错填充
            restored[:, 0::2] = x[:, : half_x_dim]
            restored[:, 1::2] = x[:, half_x_dim :]
            x = restored
        return x
