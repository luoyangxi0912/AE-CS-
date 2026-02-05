import torch
import torch.nn as nn

# FCCell 后可接激活函数
class FCCell(nn.Module):
    def __init__(self, input_size, hidden_size, extend_size = 0):
        super().__init__()
        if extend_size is None: extend_size = 0
        self.in_features, self.out_features = input_size + extend_size, hidden_size
        self.linear = nn.Linear(input_size + extend_size, hidden_size)
    def __repr__(self):
        return f'Linear(in_features={self.in_features}, out_features={self.out_features}, bias=True)'
    def forward(self, *args):
        x = torch.cat([args[0], args[1]], dim=-1) \
            if len(args)>1 and args[1] is not None else args[0]
        return self.linear(x)

# RNNCell 和 FCCell 一样可接激活函数，区别是 FCCell 定义时只有两个参数且forward 时只有一个输入
class RNNCell(FCCell):
    def __init__(self, input_size, hidden_size, extend_size = None):
        if extend_size is None: extend_size = hidden_size
        super().__init__(input_size, hidden_size, extend_size)
    def __repr__(self):
        return f'RNNCell(in_features={self.in_features}, out_features={self.out_features}, bias=True)'

# LSTMCell 后不接激活函数
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, extend_size = None):
        super().__init__()
        if extend_size is None: extend_size = hidden_size
        elif extend_size != hidden_size: self.trans_h = nn.Linear(extend_size, hidden_size)
        self.in_features, self.out_features = input_size + hidden_size, hidden_size
        self.sigmas = nn.ModuleList()
        for i in range(3):
            sigma = nn.Sequential(
                nn.Linear(input_size + hidden_size, hidden_size),
                nn.Sigmoid()
            )
            self.sigmas.append(sigma)
        self.c_candi = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )
    def __repr__(self):
        return f'LSTMCell(in_features={self.in_features}, out_features={self.out_features}, bias=True)'
    def forward(self, *args):
        x, h0, c0 = args[0], args[1], args[2]
        if hasattr(self, 'trans_h'): h0 = self.trans_h(h0)
        z = torch.cat([x, h0], dim=-1)
        sigma = []
        for i in range(3): sigma.append(self.sigmas[i](z))
        c_candi = self.c_candi(z)
        c = sigma[0] * c0 + sigma[1] * c_candi
        h = sigma[2] * torch.tanh(c)
        self.c = c
        return h

# GRUCell 后不接激活函数
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, extend_size = None):
        super().__init__()
        if extend_size is None: extend_size = hidden_size
        elif extend_size != hidden_size: self.trans_h = nn.Linear(extend_size, hidden_size)
        self.in_features, self.out_features = input_size + hidden_size, hidden_size
        self.sigmas = nn.ModuleList()
        for i in range(2):
            sigma = nn.Sequential(
                nn.Linear(input_size + hidden_size, hidden_size),
                nn.Sigmoid()
            )
            self.sigmas.append(sigma)
        self.h_candi = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )
    def __repr__(self):
        return f'GRUCell(in_features={self.in_features}, out_features={self.out_features}, bias=True)'
    def forward(self, *args):
        x, h0 = args[0], args[1]
        if hasattr(self, 'trans_h'): h0 = self.trans_h(h0)
        z = torch.cat([x, h0], dim=-1)
        sigma = []
        for i in range(2): sigma.append(self.sigmas[i](z))
        z2 = torch.cat([x, sigma[0]* h0], dim=-1)
        h_candi = self.h_candi(z2)
        h = (1-sigma[1]) * h0 + sigma[1] * h_candi
        return h
