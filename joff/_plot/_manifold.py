
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from joff._save._save_excel import _save_excel, _read_excel
from joff._plot._basic_plot_func import _generate_colors, _get_paired_colors, _gradient_lines, _multi_colored_line
from joff._load._split import _split_xy, SegData

def _get_labels(n_class, label = None):
    if label is None: Lables = [str(i + 1) for i in range(n_class)]
    elif type(label) != list: Lables = [label + r':' + str(i + 1) for i in range(n_class)]
    else: Lables = label
    return Lables

class Manifold():
    # X: array, Y: array
    def __init__(self,
                 X = None,          # X = {(x(t),y(t))}_{t=1}^N
                 Y = None,          # 这里的Y是作为点的类别的（影响点的颜色）
                 method = 'tsne',
                 save_path = '../Dataset/HY/fd/Manifold',
                 perplexity = 30.0,
                 n_colors = None,
                 if_load_data=False,
                 if_save_data=False
                 ):
        assert  X is not None
        if not os.path.exists(save_path): os.makedirs(save_path)
        if type(X) == list: X = np.concatenate(X, 0)
        if type(Y) == list: Y = np.concatenate(Y, 0)
        if Y is not None and len(Y.shape) > 1 and Y.shape[1] > 1:
            Y = np.array(np.argmax(Y, axis=1).reshape(-1, 1), dtype=np.float32)

        if if_load_data:
            data = _read_excel(save_path + method + '.xlsx')
            if type(data) == list: X, Y = data
            else: X, Y = data, None
        elif method == 'tsne':
            # t-SNE
            X = manifold.TSNE(
                n_components=2, perplexity=perplexity, init='pca', n_iter=500, random_state=0).fit_transform(X)
        elif method == 'mds':
            # MDS
            X = manifold.MDS(
                n_components=2, n_init=1, max_iter=150, normalized_stress="auto").fit_transform(X)
        elif method == 'hlle':
            # Hessian LLE
            X = manifold.LocallyLinearEmbedding(
                n_neighbors=5, n_components=2, max_iter=150, method='hessian', eigen_solver='dense').fit_transform(X)

        if (not if_load_data) and if_save_data:
            _save_excel(save_path, method + '.xlsx', [X,Y], ['X','Y'])
        self.file_path = save_path + '/' + method
        self.method = method
        self.n_colors = int(n_colors)

        self.X, self.Y = X, Y

    # X: array, Y: array
    def plot(self, X = None, Y = None, splits = None, lgd_label = None, if_use_Y = False, if_show_lgd = True):
        # data & color
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y
        n_colors = self.n_colors
        if self.n_colors is None:
            if if_use_Y: n_colors = np.max(Y)
            else: n_colors = len(splits)-1
        if splits is not None: X, Y = _split_xy(X, Y, splits)
        else: X, Y = [X], [Y]

        Lables = _get_labels(len(X), lgd_label)
        colors = _generate_colors(n_colors)
        lighten, darken = _get_paired_colors(colors)

        # plt
        plt.style.use('default')
        fig = plt.figure(figsize=[32, 18])
        ax = fig.add_subplot(111)
        kwargs = {}
        for i in range(len(X)):
            data_X = X[i]
            if if_show_lgd: kwargs['label'] = Lables[i]
            if not if_use_Y: _gradient_lines(data_X, [lighten[i], darken[i]], ax, **kwargs)
            else: _multi_colored_line(data_X[:,0], data_X[:,1], Y[i], colors, False, ax, **kwargs)

        # legend
        if if_show_lgd: plt.legend(fontsize=28, loc=1)
        _str = ' (plot)' if not if_use_Y else ' (plot with Y)'
        print('Plot {} in {}'.format(self.method + _str, self.file_path))
        plt.savefig(self.file_path + _str + '.pdf', bbox_inches='tight', format='pdf')
        plt.close(fig)

    # X: array, Y: array
    def scatter(self, X = None, Y = None, lgd_label = None, if_text = False):
        # data & color
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y
        if Y is None:
            n_colors = 1 if self.n_colors is None else self.n_colors
            Datas = [X]
        else:
            n_colors = int(np.max(Y)) if self.n_colors is None else self.n_colors
            Datas = []
            for i in range(n_colors):
                Loc = np.where(Y==i)[0]
                Datas.append(X[Loc])

        Lables = _get_labels(n_colors, lgd_label)
        colors = _generate_colors(n_colors)

        # plt
        plt.style.use('default')
        fig = plt.figure(figsize=[32, 18])
        ax = fig.add_subplot(111)

        # (x1, x2) with color 'y'
        for i in range(n_colors):
            data_X = Datas[i]
            plt.scatter(data_X[:, 0], data_X[:, 1], label=Lables[i],
                        # cmap=plt.cm.Spectral,
                        color=colors[i]
                        )
        if if_text:
            # text
            for i in range(n_colors):
                mean_x = np.mean(Datas[i], axis=0)
                plt.text(mean_x[0], mean_x[1], str(i + 1),
                         ha='center', va='bottom',
                         fontdict={'family': 'serif',
                                   'style': 'italic',
                                   'weight': 'normal',
                                   'color': [0.21, 0.21, 0.21],
                                   'size': 26}
                         )
            # legend
            plt.legend(fontsize=28, loc=1)
            # axis
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')
        else:
            plt.axis('off')

        print('Plot {} (scatter) in {}'.format(self.method, self.file_path))
        plt.savefig(self.file_path + ' (scatter).pdf', bbox_inches='tight', format='pdf')
        plt.close(fig)

if __name__ == '__main__':
    from joff._load._read_hy_data import _read_hy_dataset

    methods = ['tsne', 'mds']

    X_list, Y_list = _read_hy_dataset()
    Seg = SegData(X_list, Y_list)
    n_class = Seg.n_class

    X, Y = Seg.dynamic()
    MF = Manifold(X, Y,
                  method=methods[0],
                  save_path='../Dataset/HY/fd/Manifold',
                  perplexity = 1000.,
                  # if_load_data=True,
                  # if_save_data=True
                  n_colors=n_class
                  )
    MF.plot(if_show_lgd=False)
    MF.scatter()

    # for i in range(len(X_list)):
    #     X, Y = Seg.dynamic(index = i)
    #     cate = Seg.labels[i]
    #     __str = ' (Normal)' if cate == 0 else ' (Fault {:02d})'.format(int(cate))
    #     for method in methods:
    #         MF = Manifold(X, Y,
    #                       method=method,
    #                       save_path='../Dataset/HY/fd/Manifold/seg '+ str(i+1) + __str,
    #                       n_colors=n_class
    #                       # perplexity = 10000.
    #                       # if_load_data=True,
    #                       # if_save_data=True
    #                       )
    #         # MF.plot(if_show_lgd = False)
    #         MF.scatter()

