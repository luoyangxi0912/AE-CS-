# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

def _get_fe_perf(pred, real, loc, scaler, var_name, special):
    if special in ['CSTR_fe']:
        var_name = ['$f_{C_i}$', '$f_{T_i}$', '$f_{T_{ci}}$', '$f_{C_i^{(s)}}$', \
                  '$f_{T_i^{(s)}}$', '$f_{T_{ci}^{(s)}}$', '$f_{C^{(s)}}$', \
                  '$f_{T^{(s)}}$', '$f_{T_c^{(s)}}$', '$f_{Q_c^{(s)}}$']

    v_index = [0, 1, 2, 3, 4, 8, 5, 6, 9, 7]
    file_path = '../save/VAE_FIdN_CSTR/[VAE_FIdN] Monitoring_Indicators.xlsx'
    est_fs = []
    scaler = pd.read_csv('../data/FD_CSTR/scaler/[st] scaler_x.csv', header=None).values
    scaler = scaler[v_index]
    n_f = 10
    for i in range(n_f):
        est_fs.append(pd.read_excel(file_path, sheet_name=i).values)

    n, m = est_fs[0].shape[0], est_fs[0].shape[1]


    legend_plus = ['', '', '', '$f_{C_i^{(s)}}$', \
                   '$f_{T_i^{(s)}}$', '$f_{T_{ci}^{(s)}}$', '$f_{C^{(s)}}$', \
                   '$f_{T^{(s)}}$', '$f_{T_c^{(s)}}$', '$f_{Q_c^{(s)}}$']

    f_mag = [0, 0, 0, 0.001, 0.05, 0.05, 0.001, 0.05, 0.05, -0.1]

    c4 = ['lime', 'darkorange', 'fuchsia', 'aqua']

    RMSE, RMSE_F = np.zeros(8), np.zeros(8)
    for c in range(n_f):
        print('Plot Recon_{}.pdf'.format(c + 1))
        x = np.arange(1, 1202)
        color = plt.get_cmap('Blues')(np.linspace(0.05, 0.95, m))
        # 预测的故障信号
        data = est_fs[c]
        # 调整变量为对应的顺序
        data = data[:, v_index]

        # data = est_fs[c]
        fig = plt.figure(figsize=[26, 15])
        ax = fig.add_subplot(111)
        for j in range(m):
            if c > 2 and c == j: continue
            if c <= 2 and j > m - 5:
                ax.plot(x, data[:, j], linewidth=3, c=c4[m - j - 1], label=legend[j])
            else:
                ax.plot(x, data[:, j], linewidth=2, c=color[j], label=legend[j])
        # 加性故障
        if c > 2:
            # pred fault signal
            ax.plot(x, data[:, c], linewidth=3, c='b', label=legend[c])
            # real fault signal
            f = np.zeros((n,))
            f_signal = np.linspace(f_mag[c], f_mag[c] * 1000, 1000)
            # f[201:] = f_signal
            # 生成真实故障信号
            f[201:] = f_signal / np.sqrt(scaler[c, 1])

            RMSE_F[c - 3] = np.sum((data[:, c] - f) ** 2)
            RMSE_F[-1] += RMSE_F[c - 3]
            RMSE_F[c - 3] = np.sqrt(RMSE_F[c - 3] / 1201)

            RMSE[c - 3] = np.sum(data[:, :c] ** 2) + np.sum(data[:, c + 1:] ** 2)
            RMSE[-1] += RMSE[c - 3]
            RMSE[c - 3] = np.sqrt(RMSE[c - 3] / 1201)

            ax.plot(x, f, linewidth=3, c='r', label=legend_plus[c])

        if c > 2:
            lgd = ax.legend(loc='upper right', fontsize=43)
        else:
            lgd = ax.legend(loc='upper right', fontsize=46)
        lgd.get_frame().set_alpha(0.5)
        ax.tick_params('x', labelsize=48)
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.set_xlabel('Samples', fontsize=58)  # 设置x坐标轴
        ax.tick_params('y', labelsize=48)
        ax.set_ylabel('Fault signal', fontsize=58)  # 设置y坐标轴
        plt.tight_layout()
        plt.savefig('../save/VAE_FIdN_CSTR/Recon_{}.pdf'.format(c + 1), bbox_inches='tight')
        plt.savefig('../save/VAE_FIdN_CSTR/Recon_{}.svg'.format(c + 1), bbox_inches='tight')
        plt.show()
        plt.close(fig)

    RMSE_F[-1] = np.sqrt(RMSE_F[-1] / (1201 * 7))
    RMSE[-1] = np.sqrt(RMSE[-1] / (1201 * 7))
    print('RMSE_F:', np.round(RMSE_F, 4))
    print('RMSE:  ', np.round(RMSE, 4))
