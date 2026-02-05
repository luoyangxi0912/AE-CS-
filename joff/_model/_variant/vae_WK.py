
import torch
from joff._model.vae import VAE, VAE_default

KW_VAE_default = {
            'if_generate_var': False
            }

# kai wang
class KW_VAE(VAE):
    def __init__(self, **kwargs):
        kwargs_default = dict(VAE_default, **KW_VAE_default)
        kwargs = dict(kwargs_default, **kwargs)
        VAE.__init__(self, **kwargs)

        self.pi = torch.acos(torch.zeros(1).to(self.dvc))
        if self.if_generate_var:
            self.decoder = self.Seq(struct=self.de_struct[:-1], act=self.de_act[:-1])
            self.x_u = self.Seq(struct=self.de_struct[-2:], act=['a'])
            self.x_logv2 = self.Seq(struct=self.de_struct[-2:], act=['a'])

    def forward(self, x):
        h = self.encoder(x)
        u, logv2 = self.u(h), self.logv2(h)
        v, v2 = torch.exp(logv2 / 2.), torch.exp(logv2)
        self._latent = torch.cat([u, logv2], dim=-1)

        rd = self.mv_normal.sample(torch.Size([self.sample_times, u.size(0)])).to(self.dvc)

        _recon_tensor = torch.zeros((self.sample_times, x.size(0), x.size(1))).to(self.dvc)
        if self.if_generate_var: gene_cov = torch.zeros_like(_recon_tensor).to(self.dvc)
        for k in range(self.sample_times):
            z = u + v * rd[k]
            if self.if_generate_var:
                x_h = self.decoder(z)
                recon, gene_cov[k] = self.x_u(x_h), torch.exp(self.x_logv2(x_h))
            else:
                recon = self.decoder(z)
            _recon_tensor[k] = recon

        _recon_loss = torch.zeros((x.size(0),)).to(self.dvc)
        for i in range(x.size(0)):
            _recon_i = _recon_tensor[:, i, :]
            recon_cov_i = torch.mean( gene_cov[:, i, :], 0) if self.if_generate_var else torch.var(_recon_i, 0)
            _recon_loss[i] = torch.mean( torch.abs( torch.sum( (x[i]-_recon_i) ** 2/recon_cov_i, 0)  + \
                             torch.log(torch.cumprod(recon_cov_i, 0)[-1]) + x.size(1)*torch.log(2*self.pi) )) /2

        _kl_loss = torch.sum(u ** 2 / self.priori_v2 + v2 / self.priori_v2 - torch.log(v2 / self.priori_v2) - 1, 1) / 2
        self._cust_ts = {'cust_ts_recon': _recon_loss, 'cust_ts_kl': _kl_loss.view(-1,)}

        self.recon_loss = torch.sum(_recon_loss)
        self.kl_loss = torch.sum(_kl_loss)
        if self.loss_mean: self.recon_loss, self.kl_loss = self.recon_loss / x.size(0), self.kl_loss / x.size(0)
        self.loss = self.weighted_loss([self.recon_loss, self.kl_loss])

        if self.if_output_mean: return torch.mean(_recon_tensor, 0)
        return recon