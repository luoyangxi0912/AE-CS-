
import torch
import torch.nn as nn
from joff._nn._fcnn_module import FCNN
from joff._model.attention import MaskedAttention, MaskedAttention_default

class ST_Attn(FCNN):
    def __init__(self, **kwargs):
        kwargs = dict(MaskedAttention_default, **kwargs)
        FCNN.__init__(self, **kwargs)
        kwargs['mask_type'] = 'temporal'
        # linear:
        self.Spat_Attn = MaskedAttention(**kwargs).attention
        # nonlinear: act

        kwargs['mask_type'] = 'topological'
        kwargs['if_mask_diagnal'] = True
        # linear:
        self.Temp_Attn = MaskedAttention(**kwargs).attention
        # nonlinear: act

        self.opti()

    # [target]: q; [source]: k,v
    # q: (bsz, tgt_len, embed_dim); k: (bsz, src_len, kdim); v: (bsz, src_len, vdim)
    def forward(self, x):
        if x.ndim <3: x = x.reshape(x.shape[0], self.tgt_len, self.embed_dim)
        spat_x, spat_attn = self.Spat_Attn(x)
        return


if __name__ == '__main__':
    module = ST_Attn(task = 'prd')
    print(module.kwargs)
    print(type(module))
    print(next(module.parameters()).device == torch.device('cpu'))
    # print(dir(module))