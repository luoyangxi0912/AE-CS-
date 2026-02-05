import math
import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from joff.customize import _plot_pr_dt
from joff._nn._attr import _update_dict

# 生成子视图 layout
def _get_subplot_layout(N, zoom = None):
    # 计算最接近的正方形子图数量
    m = math.ceil(math.sqrt(N))
    n = math.ceil(N / m)
    zoom = N if zoom is None else zoom

    # 计算画布的大小
    W = 16 / 9 * zoom
    H = 1 * zoom

    # 创建画布和子图
    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(W, H), sharex=True)
    return fig, axes

# 生成多种不同的颜色（colors 默认为(0,1)之间）
def _generate_colors(num_colors, if_times_255 = False):
    colors = []
    for i in range(num_colors):
        hue = (0.6 + i / float(num_colors)) % 1.0
        lightness = (50 + i) % 100
        saturation = (90 + i) % 60 + 20
        if if_times_255: rgb = tuple(int(255 * x) for x in colorsys.hls_to_rgb(hue, lightness / 100.0, saturation / 100.0))
        else: rgb = tuple(x for x in colorsys.hls_to_rgb(hue, lightness/100.0, saturation/100.0))
        colors.append(rgb)
    return colors

# 生成多种不同的颜色（colors 默认为(0,1)之间）
def _generate_fd_colors(num_colors, if_times_255=False):
    """
    生成n个高区分度RGB颜色，前两个固定为蓝色、红色，可直接用于Matplotlib plot
    :param num_colors: 需要的颜色数量（正整数）
    :param if_times_255: 是否返回0~255整数格式（False返回0~1浮点数，可直接用在plot）
    :return: RGB颜色列表
    """
    colors = []
    # 固定前两个颜色：蓝色、红色（标准RGB值）
    base_colors = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]  # 蓝、红（0~1浮点数）

    if num_colors <= 2:
        colors = base_colors[:num_colors]
    else:
        # 先加入固定的蓝、红
        colors.extend(base_colors)
        # 后续颜色从HLS生成，避开蓝/红相近色调，保证区分度
        for i in range(2, num_colors):
            # 调整hue起始值，避开蓝红（0=红，0.67=蓝），从0.16（橙）开始
            hue = (0.16 + (i - 2) / (num_colors - 2)) % 1.0
            # 亮度固定在0.5（避免过亮/过暗），饱和度固定在0.9（高饱和，区分度高）
            lightness = 0.5
            saturation = 0.9
            # HLS转RGB（colorsys.hls_to_rgb 输入范围：hue(0~1), lightness(0~1), saturation(0~1)）
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append(rgb)

    # 按需转换为0~255整数格式（仅用于非Matplotlib场景，如PIL等）
    if if_times_255:
        colors = [tuple(int(255 * x) for x in rgb) for rgb in colors]

    return colors

# 得到更亮和更暗的颜色
def _get_paired_colors(colors, factor = 0.2):
    lighten, darken = [], []
    for color in colors:
        r, g, b = [min(color[i] * (1.0 + factor), 1.0) for i in range(3)]
        lighten.append((r, g, b))
        r, g, b = [max(color[i] * (1.0 - factor), 0.0) for i in range(3)]
        darken.append((r, g, b))
    return lighten, darken

# 生成渐变色线段（两点/多点）
# points = [(x1, y1), (x2, y2)] or [x(1~N), y(1~N)], colors = [(r1, g1, b1), (r2, g2, b2)]
def _gradient_lines(points, colors, ax = None, n_intp = 10, **kwargs):
    p = _update_dict(_plot_pr_dt, kwargs)
    if 'color' in p.keys(): p.pop('color')
    # 创建数据
    if type(points) == list:
        p1, p2 = points
        x = np.linspace(p1[0], p2[0], n_intp)
        y = np.linspace(p1[1], p2[1], n_intp)
    else:
        x = points[:,0]
        y = points[:,1]

    # 转rgb颜色
    for i, c in enumerate(colors):
        if type(c) != tuple:
            colors[i] = mcolors.to_rgb(c)

    # 定义颜色映射
    cmap = LinearSegmentedColormap.from_list('_gradient_color[cmap]', colors)

    if ax is None: ax = plt
    # 绘制图形
    for i in range(len(x)-1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color=cmap(i/len(x)), **p)
        if i == 0 and 'label' in p.keys(): p.pop('label')
    return ax

# 按color_id生成带颜色的线段
def _multi_colored_line(x, y, color_id, colors, if_gradient = False, ax = None, **kwargs):
    p = _update_dict(_plot_pr_dt, kwargs)
    if 'color' in p.keys(): p.pop('color')
    if ax is None: ax = plt
    _colors = [colors[int(i)-1] for i in color_id]
    for i in range(len(x)-1):
        if if_gradient and _colors[i] != _colors[i+1]:
            _gradient_lines([(x[i], y[i]), ([x[i+1], y[i+1]])], [_colors[i], _colors[i+1]], ax, **p)
        else:
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color= mcolors.rgb2hex(_colors[i]), **p)
        if i == 0 and 'label' in p.keys(): p.pop('label')
    return ax

if __name__ == '__main__':
    ax = _gradient_lines([(1, 2), (3, 4)], ["#FF0000", 'g'], **{'label': 'gradient line'})
    ax.legend(frameon = False)
    ax.show()
