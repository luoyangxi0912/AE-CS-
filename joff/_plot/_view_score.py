import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy, gaussian_kde, chi2
from joff._plot._basic_plot_func import _get_subplot_layout
from joff._func._msg import _msg_code

def _get_pdf_y(x, m):
    if m == 1: return chi2.pdf(x, 1)
    else: return norm.pdf(x)

def _view_score_kde(MM, path, pr_name):
    # MM 是乘以 Cov^{-1/2} 之后的
    N, m, bound = MM.shape[0], MM.shape[1], 4

    # 计算每个变量的核密度估计
    kde_list = []
    for i in range(m):
        kde = gaussian_kde(MM[:, i])
        kde_list.append(kde)

    # 计算每个变量的概率密度函数与标准正态分布之间的 KL 散度
    kl_div_list = []
    for i in range(m):
        raw_data = kde_list[i].dataset.reshape(-1,)
        kl_div_val = entropy(kde_list[i](raw_data), _get_pdf_y(raw_data, m))
        kl_div_list.append(kl_div_val)

    # 使用 Matplotlib 画出子图，每个子图包含变量的核密度估计和标准正态分布
    if m > 1:
        fig, axes = _get_subplot_layout(m)
    else:
        fig = plt.figure(figsize= (27.7, 9))
        ax = fig.add_subplot(111)
    print()
    for i in range(m):
        if m > 1:
            row, col = int(i / axes.shape[1]), int(np.mod(i, axes.shape[1]))
            ax = axes[row, col]
        x = np.linspace(-bound, bound, N)
        ax.plot(x, kde_list[i](x), label='KDE of $s_{'+str(i+1)+'}$')
        label = 'Chi2' if m == 1 else 'Standard Normal'
        title = '\mathcal{N}^2(0, 1)' if m == 1 else '\mathcal{N}(0, 1)'
        ax.plot(x, _get_pdf_y(x, m), label=label)
        ax.legend()
        ax.set_title('KL Divergence $p(s_{' + str(i+1) + '}||'+ title +')$ = ' + '{:.4f}'.format(kl_div_list[i]))

        _info = "Plot: {}{}\033[0m/{}{}\033[0m subplot of 'score_kde'...".format(_msg_code('pp', False),i+1,_msg_code('pp', False),m)
        sys.stdout.write('\r' + _info + '                ')
        sys.stdout.flush()
    print('\nSave \033[4m[mm_kde] '+ pr_name + '.pdf' + '\033[0m in ' + path)
    plt.tight_layout()
    plt.savefig(path + '/[mm_kde] '+ pr_name + '.pdf', bbox_inches='tight')
    plt.close(fig)
    # 输出 kl 最大的两个维度
    return np.argsort(-np.array(kl_div_list))[:2]


