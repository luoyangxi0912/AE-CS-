
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from joff._nn._fcnn_module import FCNN

MaskedAttention_default = {
        # temporal: 单变量时序间的注意力      q: (bsz, tgt_len* embed_dim, 1), k: (bsz, src_len* embed_dim, 1), mask: <tgt_len* embed_dim, src_len* embed_dim>;
        # spatial:  单时刻变量间的注意力      q: (bsz, embed_dim* tgt_len, 1), k: (bsz, kdim* tgt_len, 1),      mask: <embed_dim* tgt_len, kdim* tgt_len>;
        # time_lagged: 多变量时序间的注意力   q: (bsz, tgt_len, embed_dim), k: (bsz, src_len, kdim),            mask: <tgt_len, src_len>;
        # topological: 多时刻变量间的注意力   q: (bsz, embed_dim, tgt_len), k: (bsz, kdim, src_len),            mask: <embed_dim, kdim>.
        'mask_type': 'temporal',
        'embed_dim': 10,
        'num_heads': 1,
        'tgt_len': 8,
        'src_len': None,
        'kdim': None,
        'vdim': None,
        'if_mask_diagnal': False
        }

class MaskedAttention(FCNN):
    def __init__(self, **kwargs):
        kwargs = dict(MaskedAttention_default, **kwargs)
        FCNN.__init__(self, **kwargs)
        if self.src_len is None: self.src_len = self.tgt_len
        if self.kdim is None: self.kdim = self.embed_dim
        if self.vdim is None: self.vdim = self.embed_dim
        ''' 
            Define:
                embed_dim = head_dim * num_heads = q_dim
                kdim, vdim for srouce input_dim
                batch_first=True -> bsz in the 1st place, otherwise in the 2nd place
                
            Input: 
                query: (bsz, tgt_len, embed_dim), key: (bsz, src_len, kdim), value: (bsz, src_len, vdim)
                attn_mask: <tgt_len, src_len> or <bsz * num_heads, tgt_len, src_len>
                    - bool: True value indicates that the corresponding position is not allowed to attend.
                    - float: mask values will be added to the attention weight.
                average_attn_weights: True
                
            Forward:
                -> weighted: query * W_Q^T: (embed_dim, embed_dim), key * W_K^T: (kdim, embed_dim), value * W_V^T: (vdim, embed_dim)
                -> reshape: Q: (bsz * num_heads, tgt_len, head_dim), K: (bsz * num_heads, src_len, head_dim), V: (bsz * num_heads, src_len, head_dim)
                -> product: inner_product = QK^T/sqrt(head_dim)): (bsz, tgt_len, src_len) if average_attn_weights=True
                -> masked_attention: attn_output_weights = softmax(inner_product + attn_mask): <tgt_len, src_len>
                -> weighted: attn_output: (bsz, tgt_len, embed_dim) = (attn_output_weights * V) * W_O^T
            
            Output:
                attn_output: (bsz, tgt_len, embed_dim), attn_output_weights: (bsz, tgt_len, src_len)
        '''
        self.attention = nn.MultiheadAttention(self.embed_dim, self.num_heads,
                                               kdim = self.kdim, vdim = self.vdim, batch_first = True)
        if self.mask_type is not None: self._mask = self.get_mask()
        self.opti()

    def get_mask(self):
        if self.mask_type == 'temporal':
            block = torch.ones(self.embed_dim, self.tgt_len, self.src_len, dtype=float)
        elif self.mask_type == 'spatial':
            block = torch.ones(self.embed_dim, self.tgt_len, self.kdim, dtype=float)
        elif self.mask_type == 'time_lagged':
            _mask = Parameter(torch.ones(self.tgt_len, self.src_len, dtype=float))
            self._requires_grad(_mask)
        elif self.mask_type == 'topological':
            _mask = Parameter(torch.ones(self.embed_dim, self.kdim, dtype=float))
            self._requires_grad(_mask)
        if self.mask_type in ['temporal', 'spatial']:
            _mask = (1 - torch.block_diag(*block)) * -1e9
        else:
            self._diagonal_mask = torch.diag_embed(torch.ones(_mask.size(0))).to(self.dvc) * -1e9
            xavier_uniform_(_mask)
        return _mask.to(self.dvc)

    def _get_mask(self):
        if self.mask_type is None: return None
        _mask = torch.relu( torch.sign(self._mask) ) * -1e9
        if self.if_mask_diagnal: _mask += self._diagonal_mask
        return _mask

    def _pre_view(self, x, _len = None):
        if _len is None: _len = self.tgt_len
        if x.ndim <3: x = x.reshape(x.shape[0], _len, x.shape[1]//_len)
        bsz, _len, _dim = x.shape
        if self.mask_type == 'temporal': return x.permute(0, 2, 1).view(bsz, _len* _dim, 1)
        if self.mask_type == 'spatial':  return x.view(bsz, _len * _dim, 1)
        if self.mask_type == 'time_lagged':  return x
        if self.mask_type == 'topological':  return x.permute(0, 2, 1)

    def _post_view(self, x):
        bsz, _Len, _ = x.shape
        if self.mask_type == 'temporal': return x.view(bsz, self.embed_dim, _Len// self.embed_dim).permute(0, 2, 1)
        if self.mask_type == 'spatial':  return x.view(bsz, self.tgt_len, _Len// self.tgt_len)
        if self.mask_type == 'time_lagged':  return x
        if self.mask_type == 'topological':  return x.permute(0, 2, 1)

    # [target]: q; [source]: k,v
    # q: (bsz, tgt_len, embed_dim); k: (bsz, src_len, kdim); v: (bsz, src_len, vdim)
    def forward(self, q, k = None, v = None):
        if k is None: k = q
        if v is None: v = k

        residual = q
        q, k, v = self._pre_view(q), self._pre_view(k, self.src_len), self._pre_view(v, self.src_len)
        _mask = self._mask if self.mask_type in ['temporal', 'spatial'] else self._get_mask()
        q, attn = self.attention(q, k, v, attn_mask=_mask)
        q = self._post_view(q) + residual
        return q, attn


if __name__ == '__main__':
    module = MaskedAttention(task = 'prd')
    print(module.kwargs)
    print(type(module))
    print(next(module.parameters()).device == torch.device('cpu'))
    # print(dir(module))