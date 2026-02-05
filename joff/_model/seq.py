import torch
import torch.nn as nn
from joff._nn._attr import _update_dict
from joff._nn._seq import _get_act, _check_struct, _expand_act
from joff._model.cell import FCCell, RNNCell, LSTMCell, GRUCell

class Layer(nn.Sequential):
    def forward(self, *args):
        """
        重写 forward：仅调用第一个子模块的 forward，传递所有接收的参数
        *args：接收任意数量的位置参数（解决"参数数量不匹配"问题）
        **kwargs：接收任意数量的关键字参数（增强通用性）
        """
        # 确保 Sequential 中至少有一个子模块，避免索引错误
        if len(self) == 0:
            raise ValueError("FirstModuleSequential 不能为空，请至少添加一个子模块")

        x = args[0]
        h0 = args[1] if len(args) >1 else None
        c0 = args[2] if len(args) >2 else None

        for cell in self:
            # print(f'cell = {cell} >< input_size = {x.size()}')
            # 有且只有一个Dropout
            if isinstance(cell, nn.Dropout):
                x = cell(x)
                h0 = cell(h0) if h0 is not None else None
            # 有且只有一个此类Cell
            elif isinstance(cell, (FCCell, RNNCell, LSTMCell, GRUCell)):
                x = cell(x, h0, c0)
                self.c = cell.c if hasattr(cell, 'c') else None
            else:
                x = cell(x)
        return x

class Sequential(nn.Module):
    def __init__(self, _super = None, **kwargs):
        super().__init__()
        default = {'struct': None, 'act': None, 'extend_dim': None, 'cell_type': None,
                   'drop_thrd': None, 'auto_drop': None, 'drop_rate': None, 'use_bn': None}
        # 检查 kwargs 是否合规
        default = _update_dict(default, _super.kwargs)
        p = _update_dict(default, dict(**kwargs))

        # 检查 struct 和 act 是否合规
        p['struct'] = _check_struct(p['struct'])  # 检查是不是有字符运算
        p['act'] = _expand_act(p['act'], len(p['struct']) - 2)  # 自动扩展 act 至与 struct 所需激活数匹配，取 act[:-1] 列表循环（最后一个作为输出层激活）
        # 检查 cell_type 和 extend_dim 是否合规
        if p['cell_type'] is None: p['cell_type'] = 'fc'
        if p['extend_dim'] is not None:
            if not isinstance(p['extend_dim'], list): p['extend_dim'] = list(p['extend_dim'])
            # 将后续不需扩维的层补0
            if len(p['extend_dim']) < len(p['struct']) - 1: p['extend_dim'] += [0] * (len(p['struct']) - 1 - len(p['extend_dim']))
        else:
            # 将不需扩维的层设置为0
            p['extend_dim'] = [0] * (len(p['struct']) - 1)

        # 设置类的属性
        for key in p.keys(): setattr(self, key, p[key])
        # print(p)
        # 创建 module
        for i in range(len(self.struct) - 1):
            layer = Layer()
            input_dim, output_dim = self.struct[i], self.struct[i + 1]
            exd_dim = self.extend_dim[i]

            # Dropout (激活之后，线性之前)
            p_drop = _super.get_drop_rate(input_dim + exd_dim, **p)
            if p_drop > 0.: layer.append(nn.Dropout(p=p_drop))

            # Linear
            cell = eval(f'{self.cell_type.upper()}Cell({input_dim},{output_dim},{exd_dim})')
            layer.append(cell)

            # BatchNorm1d (激活之前，线性之后)
            if self.use_bn and i < len(self.struct) - 2:
                # and p['act'][i] in ['s', 't', 'g']:
                layer.append(nn.BatchNorm1d(output_dim))

            # Act
            if self.cell_type not in ['lstm', 'gru']:
                layer.append(_get_act(self.act[i]))
            exec(f'self.layer_{i} = layer')

    def __len__(self):
        return len(self.struct) - 1

    # 核心：实现 __iter__ 方法，返回子模块的迭代器（支持 for 循环）
    def __iter__(self):
        # 迭代 _modules 的值（即所有子模块）
        return iter(self._modules.values())

    # 可选：实现 __getitem__ 方法，支持索引访问（如 seq[0]）
    def __getitem__(self, idx):
        # 将有序字典的值转为列表，通过索引取值
        modules = list(self._modules.values())
        return modules[idx]

    # state = (h0s *[list], c0s *[list]): h0s 是扩展到各层输入的附加状态, c0s 是各层记忆细胞状态
    def forward(self, x, *states):
        if len(states) == 0: return self.seq(x)

        if self.cell_type != 'lstm': h0s = states
        else: h0s, c0s = states[:len(states) // 2], states[len(states) // 2:]

        a = x
        self.hs = []
        if self.cell_type == 'lstm': self.cs = []
        for i, layer in enumerate(self):
            h = h0s[i] if i < len(h0s) and h0s[i] is not None else None
            c = c0s[i] if self.cell_type == 'lstm' else None
            # 前向传播
            a = layer(a, h, c)
            self.hs.append(a)
            if self.cell_type == 'lstm': self.cs.append(layer.c)
        return a
