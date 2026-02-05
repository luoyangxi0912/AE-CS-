import torch
import torch.nn as nn
from torch.autograd import Variable
from joff._nn._fcnn_module import FCNN
from joff._nn._vmap import Jacbi, vmap_jac

# detach() 避免梯度计算：会断开与计算图的连接，不保留梯度，但共享内存（所以修改 detached 张量会影响原张量，除非原张量不再使用）。
def detach_(list_b):
    if list_b is None: return None
    if not isinstance(list_b, list): return list_b.detach()
    return [b.detach() if b is not None else None for b in list_b] if len(list_b)!=0 else None
# clone() 避免原地操作：克隆一个新的张量，保留梯度，但不共享内存。然而，克隆会保留原张量的梯度信息，即如果原张量需要梯度，那么克隆的张量也会在反向传播中计算梯度。
def clone_(list_b, if_detach = False, if_grad = True):
    if list_b is None: return None
    if if_detach: list_b = detach_(list_b)
    if not isinstance(list_b, list): return list_b.clone().requires_grad_(if_grad)
    return [b.clone().requires_grad_(if_grad) if b is not None else None for b in list_b] if len(list_b)!=0 else None
def todvc_(list_b, dvc):
    if list_b is None: return None
    if not isinstance(list_b, list): return list_b.to(dvc).requires_grad_(list_b.requires_grad)
    return [b.to(dvc).requires_grad_(b.requires_grad) if b is not None else None for b in list_b] if len(list_b)!=0 else None

'''
    有源自回归 Auto-Regressive with Extra Inputs
    （系统可观测 = 可观测部分可通过设计L稳定 + 不可观测部分自稳定）
'''
arx_defaults = {
    'x_dim': 0,
    'u_dim': 0,
    'y_dim': 0,
    'struct': [-1, 50, 100, 50, -1],  # 各层神经元数（隐藏层可变）
    'view_addi_info': ['obs_loss', 'loss'],
    'n_net': 1,                 # 模型数量：1（合并模型）/2（分开模型）
    'neg_diag_thrd': 0.5,       # 负对角占优阈值
    'unobs_stb': None,          # 不可观测部分稳定性损失计算方式：‘f_norm’/‘min_norm’/'jacbi'/'l2_norm'
    'alf': (1.,0.1),
    'if_allow_drop': True,      # 是否允许dropout（仅 if_calc_jac = True 时有效）
    'if_adp_P': True,           # 是否自适应学习P
    'if_calc_jac': False        # 是否计算雅克比
}
class ARX(FCNN):
    def __init__(self, **kwargs):
        # 合并默认参数和用户传入的kwargs（用户参数覆盖默认值）
        kwargs = dict(arx_defaults, **kwargs)

        # RNN 的 data_loader 需满足(shuffle = False, drop_last = True)
        kwargs['Loader'] = 'RNNLoader'
        kwargs['act'][-1] = 'a'
        kwargs['v_dim'] = kwargs['x_dim'] - kwargs['y_dim']
        if kwargs['n_net'] == 1:
            # obs + unobs 网络 (u, x) -> x
            kwargs['struct'] = [kwargs['u_dim']] + kwargs['struct'][1:-1] + [kwargs['x_dim']]
            kwargs['extend_dim'] = [kwargs['x_dim']]
        else:
            # unobs 网络 (u, x) = (u, y, v) -> v
            kwargs['struct'] = [kwargs['u_dim']] + kwargs['struct'][1:-1] + [kwargs['v_dim']]
            kwargs['extend_dim'] = [kwargs['x_dim']]
            # obs 网络 (u, y) -> y
            if 'act_2' not in kwargs.keys(): kwargs['act_2'] = kwargs['act'].copy()
            if 'struct_2' not in kwargs.keys(): kwargs['struct_2'] = kwargs['struct'].copy()
            kwargs['struct_2'] = [kwargs['u_dim']] + kwargs['struct_2'][1:-1] + [kwargs['y_dim']]
            kwargs['extend_dim_2'] = [kwargs['y_dim']]

        if kwargs['unobs_stb'] not in [None, 'l2-norm']: kwargs['view_addi_info'].insert(1, 'A22_stb_loss')
        if kwargs['if_adp_P']: kwargs['view_addi_info'].insert(-1, 'P_norm')
        if kwargs['unobs_stb'] == 'l2_norm': kwargs['l2_norm'] = kwargs['lr'] / 10.
        if kwargs['unobs_stb'] == 'jacbi': kwargs['if_calc_jac'] = True

        if kwargs['if_calc_jac']: kwargs['view_addi_info'][1:1] = ['M_stb_loss']
        if kwargs['if_calc_jac'] and not kwargs['if_allow_drop']: kwargs['auto_drop'] = False
        

        FCNN.__init__(self, **kwargs)
        # print(self.struct)

        if self.unobs_stb is not None:
            self.act_factor = 1.0
            for act in self.act:
                if act == 's': self.act_factor *= 0.25
                elif act == 'g': self.act_factor *= 0.858

        # 构建网络结构
        self.seq = self.Seq(extend_dim = self.extend_dim)
        if self.n_net == 2:
            self.extend_dim_2 += self.struct[1:-1]
            self.seq_2 = self.Seq(struct = self.struct_2, act = self.act_2, extend_dim = self.extend_dim_2)

        self.curr_batch_size = -1
        # 输出矩阵 C
        self.C = torch.eye(self.x_dim)[:self.y_dim]
        # 正定矩阵 P
        if not self.if_adp_P:
            self.P = torch.eye(self.x_dim)
        else:
            self.p_diag_log = nn.Parameter(torch.zeros(self.x_dim))
            self.P_21 = nn.Parameter(torch.zeros([self.v_dim, self.y_dim]))

        # 优化器
        self.opti()

    # 负对角占优损失
    def neg_diag_loss(self, A):
        if A.dim() == 2:
            a_diag = torch.diag(A)
            # 负对角 + 对角占优
            return torch.sum(torch.relu(a_diag + self.neg_diag_thrd) +
                             torch.relu(torch.sum(A.abs(), dim=-1) - a_diag.abs() * 2 + self.neg_diag_thrd))
        else:
            a_diag = torch.diagonal(A, dim1=1, dim2=2)
            # 负对角 + 对角占优
            return torch.sum(torch.relu(a_diag + self.neg_diag_thrd) +
                             torch.relu(torch.sum(A.abs(), dim=-1) - a_diag.abs() * 2 + self.neg_diag_thrd))/a_diag.size(0)

    # 不可观测部分的稳定性损失
    def unobs_w_loss(self):
        if not hasattr(self, 'W_set'):
            # 这里是约束不可观测部分 A_22 = J_22，让 A_22 负定，从而好设计P 让 M_22 负定
            self.W_set = []
            for name, param in self.seq.named_parameters():
                if "weight" in name:  # 匹配参数名中的"weight"（精准匹配可用name.endswith("weight")）
                    self.W_set.append(param)
                    # print(f'{name}, {param.shape}')
            self.W_set[0] = self.W_set[0][:, self.u_dim + self.y_dim:]
            if self.n_net == 1:
                self.W_set[-1] = self.W_set[-1][self.y_dim:,:]

        # 不可观测稳定性约束：加法（所有||W||<1）比乘法严格（||W||乘积*act_factor<1），F范数比1-/inf-范数严格
        A22_stb_loss = 0.
        for W in self.W_set:
            if self.unobs_stb == 'f_norm':
                A22_stb_loss += torch.relu(torch.linalg.norm(W) -1) # 默认是 F 范数
            elif self.unobs_stb == 'min_norm':
                A22_stb_loss += torch.relu(min(torch.linalg.norm(W, ord=1), torch.linalg.norm(W, ord=float('inf'))) - 1)
        # A22_stb_loss = torch.relu(A22_stb_loss * self.act_factor - 1)
        return A22_stb_loss

    # 计算雅克比矩阵
    def get_jacbi(self, *args):
        if self.if_allow_drop:
            cur_training = self.training
            self.eval()
        if self.unobs_stb == 'jacbi':
            J = Jacbi(self.dynamics_fn, 1).batch_jac(*args)
            A_2 = J[self.y_dim:]
            self.A22_stb_loss += self.neg_diag_loss(A_2)
        else:
            J = vmap_jac(self.dynamics_fn, 1,*args)
        if self.if_allow_drop and cur_training:
            self.train()
        # if self.unobs_stb == 'jacbi':
        #     return J
        # else:
        return clone_(J, True, False)

    # 初始化上一个时刻的 x 和 y
    def init_state(self, batch_size):
        if (self.curr_batch_size == batch_size and
                (not hasattr(self, 'cur_loader') or not self.cur_loader.reset_state)): return
        self.curr_batch_size = batch_size
        self.x_k = torch.zeros([batch_size, self.x_dim], device=self.dvc, requires_grad=True)

    # 单时刻前向 dynamics_fn: (x_{k-1}, u_{k-1}, h_{k-1}, c_{k-1}) -> x_k
    def dynamics_fn(self, u, x):
        if self.n_net == 1: return self.seq(u, x)
        v = self.seq(u, x)
        y_pre = x[:, :self.y_dim] if x.dim() > 1 else x[:self.y_dim]
        extd_y = [y_pre] + self.seq.hs[:-1]
        y = self.seq_2(u, *extd_y)
        x = torch.cat([y, v], dim=-1)
        return x

    # X [batch_size, stack * (u_dim + y_dim)]
    def forward(self, X):
        # 数据解包
        batch_size = X.size(0)
        stack = X.size(1) // (self.u_dim + self.y_dim)
        X_3d = X.reshape(batch_size, stack, self.u_dim + self.y_dim)

        # U: [stack, batch, u_dim], Y: [stack, batch, y_dim]
        U, Y = X_3d[:,:,:self.u_dim].transpose(0,1) , X_3d[:,:,self.u_dim:].transpose(0,1)
        # print(f'U.size() = {U.size()}, Y.size() = {Y.size()}, training = {self.training}')

        # 检查是否要初始化状态
        self.init_state(batch_size)

        if self.C.device != self.dvc: self.C = self.C.to(self.dvc)

        Y_next_prd = []
        x = self.x_k
        self.M_stb_loss, self.A22_stb_loss = 0., 0.
        for k in range(stack):
            u = U[k]  # [batch_size, u_dim]
            # 前向运算
            x_next = self.dynamics_fn(u, x)
            # 用L修正x
            if self.if_calc_jac and hasattr(self,'fix_with_L'):
                if k == 0 and hasattr(self, 'init_Sys_Matri'): self.init_Sys_Matri()
                # 计算雅克比
                args = clone_([u, x], True, True)
                J = self.get_jacbi(*args)
                y_pre_error = Y[k] - x @ (self.C).T
                # print(f'x.size() = {x.size()}')
                x_fix = self.fix_with_L(y_pre_error, J)
                x_next = x_next + x_fix

            # 下一时刻的预测值
            if k < stack - 1:
                y_next_prd = x_next @ (self.C).T
                Y_next_prd.append(y_next_prd)
            # 用下一时刻的 x 更新当前 x
            # x = clone_(x_next)
            x = x_next

        self.x_k = clone_(x_next, True, True)
        Y_next_prd = torch.stack(Y_next_prd)
        Y_next = Y[1:]

        if self.if_adp_P: self.P_norm = torch.linalg.norm(self.P)
        self.obs_loss = torch.mean((Y_next_prd - Y_next) ** 2)
        self.loss = self.obs_loss * self.alf[0]
        if self.unobs_stb is not None and self.unobs_stb != 'l2_norm':
            if self.unobs_stb != 'jacbi': self.A22_stb_loss = self.unobs_w_loss()
            else: self.A22_stb_loss = self.A22_stb_loss / stack
            self.loss = self.loss + self.A22_stb_loss * self.alf[1]
        if hasattr(self, 'M_stb_loss'):
            self.M_stb_loss = self.M_stb_loss / stack
            self.loss = self.loss + self.M_stb_loss * self.alf[1]
        
        Y_prd_flat = Y_next_prd.transpose(0, 1).reshape(batch_size, -1)
        Y_flat = Y_next.transpose(0, 1).reshape(batch_size, -1)
        self._cust_mm = {'cust_mm_ey': Y_prd_flat - Y_flat}

        return Y_prd_flat






