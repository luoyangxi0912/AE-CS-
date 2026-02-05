import torch
import torch.nn as nn
from joff._nn._fcnn_module import FCNN

# 深层RNN：可以通过 struct 改变各隐层神经元数目
# 但是不能改变各层的激活函数（受到rnn中参数 nonlinearity 的限制只能选择'tanh'和'relu'）
# 支持批次训练 (btach, seq_len, in_dim)
class DL_RNN(FCNN):
    def __init__(self, **kwargs):
        defaults = {
            'struct': [-1, 50, 100, 50, 1],  # 各层神经元数（隐藏层可变）
            'basic_module': 'rnn',  # 模型类型：'rnn'/'lstm'/'gru'（默认rnn）
            'seq_lens': [3, 1],     # [n, m]：仅n_to_m时生效
            'batch_size': None,     # 用于设置h0和c0
            'y_matri': None,        # 最后一层 h 到 y 的映射，可以用自定义矩阵代替
            'if_bidirt': False      # 是否双向
        }
        # 合并默认参数和用户传入的kwargs（用户参数覆盖默认值）
        kwargs = dict(defaults, **kwargs)
        # RNN 的 data_loader 需满足 (shuffle = False, drop_last = True) 以确保
        kwargs['Loader'] = 'RNNLoader'
        FCNN.__init__(self, **kwargs)
        self.in_dim, self.out_dim = self.struct[0], self.struct[-1]
        self.num_directions = 2 if self.if_bidirt else 1
        # 'seq_task' 任务类型：n_to_1/n_to_n/n_to_m
        if self.seq_lens[1] == 1: self.seq_task = 'n_to_1'
        elif self.seq_lens[0] == self.seq_lens[1]: self.seq_task = 'n_to_n'
        else: self.seq_task = 'n_to_m'

        # 校验参数合法性
        if self.basic_module not in ['rnn', 'lstm', 'gru']:
            raise ValueError(f"basic_module必须是'rnn'/'lstm'/'gru'，当前：{self.basic_module}")

        # 初始化h0、c0
        self.init_state()

        # 分层搭建rnn
        self.seq_blocks = nn.ModuleList()
        seq_layer = eval(f'nn.{self.basic_module.upper()}')
        for i in range(len(self.struct)-2):
            block = nn.ModuleList()
            input_size = self.struct[i] if i == 0 else self.struct[i] * self.num_directions
            p_drop = self.get_drop_rate(input_size, **self.kwargs)
            if p_drop>0.: block.append(nn.Dropout(p = p_drop))
            block.append(
                seq_layer(
                    input_size=input_size,
                    hidden_size=self.struct[i+1],
                    num_layers=1,
                    bidirectional=self.if_bidirt,
                    batch_first=True # 批次在前 [batch_size, seq_len, feature]
                )
            )
            self.seq_blocks.append(block)

        # 基础映射层：将隐藏维度映射到每个时间步的输出维度out_dim
        if self.y_matri is None: self.output_layer = nn.Linear(self.struct[-2] * self.num_directions, self.struct[-1])
        else: self.output_layer = lambda last_h: self.y_matri @ last_h

        # n→m 专用：时序长度变换层（用Conv1d实现灵活的时序维度变换）
        if self.seq_task == 'n_to_m':
            self.seq_transform = nn.Conv1d(
                in_channels=self.seq_lens[0],  # 输入通道数=每个时间步的输出维度
                out_channels=self.seq_lens[1],  # 输出通道数不变
                kernel_size=1  # 卷积核大小（控制时序感受野）
            )

        # 设定优化器
        self.opti()

    def init_state(self):
        if self.batch_size is not None: batch_size = self.batch_size
        else: batch_size = self.b

        self.c0s, self.h0s = [], []
        for i in range(len(self.struct) - 2):
            self.h0s.append(torch.zeros(self.num_directions, batch_size, self.struct[i + 1]))
            if self.basic_module == 'lstm': self.c0s.append(torch.zeros(self.num_directions, batch_size, self.struct[i + 1]))

    def forward(self, x):
        """
        前向传播：适配n→1/n→n/n→m三种任务
        :param x: 输入张量，形状 (batch_size, n, in_dim)，n = x_seq_len = seq_lens[0]
        :return: 输出张量，形状 (batch_size, m, out_dim)，n = y_seq_len = seq_lens[1]
        """
        # 重新初始化

        if hasattr(self, 'cur_loader') and self.cur_loader.reset_state: self.init_state()

        layer_input = x
        # 逐层前向（和之前一致）
        for i, block in enumerate(self.seq_blocks):
            # 初始化状态
            h0 = self.h0s[i]
            if self.basic_module == 'lstm':
                c0 = self.c0s[i]
                initial_state = (h0.detach().to(self.dvc), c0.detach().to(self.dvc))
            else:
                initial_state = h0.detach().to(self.dvc)

            # dropout
            if len(block) > 1:
                layer_input = block[0](layer_input)
                seq_layer = block[1]
            else:
                seq_layer = block[0]

            # layer_h 包含 seq_len 个时间步的 h [batch_size,seq_len, feature]
            # final_state 是最后一个时刻的特征 (h0, c0) [num_directions, batch_size, feature] *2
            layer_h, final_state = seq_layer(layer_input, initial_state)
            layer_input = layer_h

            if self.basic_module == 'lstm':
                self.h0s[i], self.c0s[i] = final_state
            else:
                self.h0s[i] = final_state

        # 将所有时间步的隐藏特征映射到out_dim：(batch_size, n, out_dim)
        layer_input = self.output_layer(layer_input)

        if self.seq_task == 'n_to_1':
            # n→1：取最后一个时间步（也可改用torch.mean(layer_input, dim=1)）
            y = layer_input[:, -1, :].unsqueeze(1)  # 形状：(batch_size, 1, out_dim)

        elif self.seq_task == 'n_to_n':
            # n→n：直接返回所有时间步输出
            y = layer_input  # 形状：(batch_size, n, out_dim)

        elif self.seq_task == 'n_to_m':
            # n→m：变换时序长度，用卷积（Conv1d）提取时序特征
            y = self.seq_transform(layer_input)  # (batch_size, out_dim, N)

        return y

if __name__ == "__main__":
    device = torch.device("cuda")
    batch_size, seq_len, in_dim = 3, 5, 10
    y_seq_len, out_dim = 2, 1
    input_data = torch.randn(batch_size, seq_len, in_dim).to(device)  # (3, 5, 10)

    # 1. 测试 n→1 任务（5→1）
    print("===== n→1 任务 =====")
    n_to_1_model = DL_RNN(
        basic_module='lstm',
        seq_lens=[seq_len, 1],
        batch_size = batch_size,
        struct = [in_dim, 150, 50, 150, out_dim],  # 各层神经元数（隐藏层可变）
        auto_drop = True,
        if_bidirt = True   # 是否双向
    ).to(device)
    print(n_to_1_model)
    print(f"输入形状 (batch_size, seq_len,  in_dim): {input_data.shape}")
    print(f"输出形状 (batch_size, y_seq_len, out_dim): {n_to_1_model(input_data).shape}")  # (3, 1)

    # 2. 测试 n→n 任务（5→5）
    print("\n===== n→n 任务 =====")
    n_to_n_model = DL_RNN(
        basic_module='gru',
        batch_size=batch_size,
        seq_lens=[seq_len, seq_len],
        struct= [in_dim, 100, 50, out_dim],
        if_bidirt=False  # 是否双向
    ).to(device)
    print(n_to_n_model)
    print(f"输入形状 (batch_size, seq_len,  in_dim): {input_data.shape}")
    print(f"输出形状 (batch_size, y_seq_len, out_dim): {n_to_n_model(input_data).shape}")  # (5, 3, 1)

    # 3. 测试 n→m 任务（5→3）
    print("\n===== n→m 任务 =====")
    n_to_m_model = DL_RNN(
        basic_module='rnn',
        batch_size=batch_size,
        seq_lens = [seq_len, y_seq_len],  # [n, m]：仅n_to_m时生效
        struct= [in_dim, 50, 100, out_dim],
        if_bidirt=False   # 是否双向
    ).to(device)
    print(n_to_m_model)
    print(f"输入形状 (batch_size, seq_len, in_dim): {input_data.shape}")
    print(f"输出形状 (batch_size, y_seq_len, out_dim): {n_to_m_model(input_data).shape}")  # (3, 3, 1)