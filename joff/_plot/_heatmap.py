import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def _matrix2heatmap(matrix, path, file):
    # Create the heatmap plot
    fig, ax = plt.subplots()

    # coolwarm; RdBu.reversed(); RdYlGn.reversed()
    cmap = plt.cm.coolwarm
    # cmap = plt.cm.RdBu.reversed()
    bound = np.abs(matrix.max())
    norm = mcolors.Normalize(vmin=-bound, vmax=bound)
    # matrix = np.flip(matrix, axis=0)
    heatmap = ax.pcolor(matrix, cmap=cmap, norm=norm)
    for i in range (matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j + 0.5, i + 0.5, np.round(matrix[i,j]/bound, 2), ha='center', va='center')

    # Set the x-axis and y-axis labels and ticks
    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels( ['$r_{'+ str(i) +'}$' for i in range(1, matrix.shape[1] + 1) ], minor=False)
    ax.set_yticklabels( ['$r_{'+ str(i) +'}$' for i in range(1, matrix.shape[0] + 1) ], minor=False)

    # Add colorbar
    cbar = plt.colorbar(heatmap)

    # Show the plot
    file_name = path + '/' + file
    plt.savefig(file_name + '.pdf', bbox_inches='tight')
    # plt.savefig(file_name + '.png', bbox_inches='tight')
    plt.close(fig)
    print("Plot \033[4m{}\033[0m in '{}'".format(file + '.pdf', path))

def _score2heatmap(dr, score, TS, thrd, path, file, zoom = None):
    zoom = 0.5 if zoom is None else zoom
    col1 = [plt.cm.Paired(0), plt.cm.Paired(2), plt.cm.Paired(6), plt.cm.Paired(4)]
    col2 = plt.cm.binary

    labels = ['RN', 'RF', 'FA', 'MD']
    score = np.abs(score)
    loc0 = np.where(dr == 0)[0]
    loc0 = loc0[np.argsort(-np.sum(score[loc0]**2,-1))][:5]
    loc1 = np.where(dr == 1)[0]
    loc1 = loc1[np.argsort(np.sum(score[loc1]**2,-1))][:5]
    loc = sorted(loc0.tolist() + loc1.tolist() + np.where(dr >= 2)[0].tolist())

    score, dr = score[loc], dr[loc]
    rate = TS[loc] /thrd
    score_max = np.sqrt(thrd/score.shape[1])
    score = score / score_max
    matrix = np.concatenate([dr.reshape(-1, 1), score], -1)

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(7 *zoom, matrix.shape[0] * 0.6 *zoom))

    ax.set_xlim(0.5, matrix.shape[1] + 0.5)
    label_x = ['DR'] + ['$\\bar s_{'+str(i+1)+'}$' for i in range(score.shape[1])]
    ax.set_xticks(np.arange(matrix.shape[1]) + 1, labels = label_x, fontsize = 16 *zoom, minor=False)

    ax.set_ylim(matrix.shape[0] + 0.5, 0.5)
    label_t = ['$t('+str(i+1)+')$' for i in loc]
    ax.set_yticks(np.arange(len(loc)) + 1, labels = label_t, fontsize = 14 *zoom, minor=False)

    ax2 = ax.twinx()
    ax2.set_ylim(matrix.shape[0] + 0.5, 0.5)
    label_r = [str(np.round(rate[i],2)) for i in range(rate.shape[0])]
    ax2.set_yticks(np.arange(len(loc)) + 1, labels = label_r, fontsize = 14 *zoom, minor=False)

    # Define a function to draw each cell of the heatmap
    def draw_cell(i, j, val):
        color = col1[int(val)] if j == 0 else col2(val)
        rect = plt.Rectangle((j + 0.5, i + 0.5), 1, 1, facecolor=color)  # , edgecolor='black'
        plt.gca().add_patch(rect)
        # text
        if j == 0:
            plt.text(j+1, i+1, labels[dr[i]], ha='center', va='center', fontsize=16 * zoom)
        else:
            color = 'white' if val > 0.7 else 'black'
            plt.text(j+1, i+1, np.round(val, 2), ha='center', va='center', fontsize=14 * zoom, color=color)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            draw_cell(i, j, matrix[i, j])

    # Set a title for the plot
    ax.set_title('$|s|$ should greater than {:.2f} (i.e., $\\bar s > 1$)'.format(score_max), fontsize = 18.5 *zoom)

    # Show the plot
    file_name = path + '/' + file
    if not os.path.exists(path): os.makedirs(path)
    plt.savefig(file_name + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot \033[4m{}\033[0m in '{}'".format(file + '.pdf', path))

if __name__ == '__main__':
    score = np.diag(np.arange(1,11))
    dr = np.random.randint(0,4, (10,) )
    _score2heatmap(dr, score, np.sum(score**2,-1), 120.0, '../Result/_plot', 'random_score')
    _matrix2heatmap(score, '../Result/_plot', 'random_matrix2heatmap')
