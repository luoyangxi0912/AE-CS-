# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""
import os
import urllib.parse

def _clickable_file_link(file_path):
    """
    格式化文件路径为 PyCharm 可点击的链接，处理空格
    """
    abs_path = os.path.abspath(file_path)
    formatted_path = abs_path.replace('\\', '/')
    encoded_path = urllib.parse.quote(formatted_path, safe='/:')
    return f"file:///{encoded_path}"

RESET = '\033[0m'

def _word_style(_str, _style = None):
    if _style is not None:
        _style_str = ''
        for c in _style:
            _style_str += f'{styles[c]}'
        return f'{_style_str}{_str}{RESET}'
    return _str

def fg_color(_int):
    """前景色（文字颜色）"""
    return f'\033[38;5;{_int}m'

def bg_color(_int):
    """背景色"""
    return f'\033[48;5;{_int}m'

def rgb_color(rgb):
    return f'\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m'

def rgb_to_int(target_rgb):
    """在256色表中找到最接近目标RGB的颜色"""

    # 256色表中的RGB值（近似）
    # 前16色是系统色，16-231是6x6x6 RGB立方体，232-255是灰度

    # 6级RGB值（0-5对应的实际RGB值）
    rgb_levels = [0, 95, 135, 175, 215, 255]

    # 计算最接近的索引
    r_idx = min(range(len(rgb_levels)), key=lambda i: abs(rgb_levels[i] - target_rgb[0]))
    g_idx = min(range(len(rgb_levels)), key=lambda i: abs(rgb_levels[i] - target_rgb[1]))
    b_idx = min(range(len(rgb_levels)), key=lambda i: abs(rgb_levels[i] - target_rgb[2]))

    # 计算颜色代码
    _int = 16 + 36 * r_idx + 6 * g_idx + b_idx

    return _int

def hex_to_rgb(hex_color):
    """将十六进制颜色转换为RGB"""
    hex_color = hex_color.lstrip('#')
    hex_color = hex_color.lstrip('~')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return target_rgb

# 常用颜色代码
int_colors = {
    "橙色": 208,
    "深橙色": 202,
    "深红橙": 160,  # 比202更暗，偏红
    "暗橙色": 166,  # 暗橙色
    "粉色": 205,
    "紫色": 129,
    "棕色": 130,
    "灰色": 245,
    "浅灰色": 250,
    "红色": 196,
    "绿色": 46,
    "蓝色": 21,
    "黄色": 226,
    "青色": (117, 194, 179),
    "洋红色": 201,
    "亮草绿": 82,
    "鲜草绿": 46,
    "标草绿": 40,
    "黄草绿": 118
}
hex_colors = {
    "淡蓝色": '#49B0CE',
    "亮蓝色": '#65DFFF',
    "亮紫色": '#DBB8FF',
    "亮粉色": '#FF78B2',
    "灰青色": '#75C2B3',
    "亮绿色": '#599E5E',
    "草绿色": '#5EA963',
    "灰绿色": '#629755',
    "亮橙色": '#FFA245',
    "珊瑚橙": '#FF806C',
    "淡黄色": '#DACA77',
    "亮黄色": '#FFBF67',
    "灰黄色": '#D7C781',
    "1鲜红": '#e63946',
    "1浅红": '~e63946',
    "1大红": '#c1121f',
    "1深红": '#780000',
    "1淡蓝": '#a8dadc',
    "1天蓝": '#669bbc',
    "1灰蓝": '#457b9d',
    "1深青": '#1d3557',
    "1赭石": '#fdf0d5',
    "1墨青": '#003049',
    "2朱红": '#bc4749',
    "2赭石": '#f2e8cf',
    "2翠绿": '#a7c957',
    "2柳绿": '~a7c957',
    "2森绿": '#6a994e',
    "2墨绿": '#386641',
    "3淡紫": '#e4c1f9',
    "3粉色": '#ff99c8',
    "3浅绿": '#d0f4de',
    "3浅黄": '#fcf6bd',
    "4浅肉": '#ebd4cb',
    "4肉色": '#da9f93',
    "4玫瑰": '#b6465f',
    "4深红": '#890620',
    "3深褐": '#2c0703'
}
def _to_code(_color):
    if _color is None: return None
    if _color in hex_colors.keys(): _color = hex_colors[_color]
    if _color in int_colors.keys(): _color = int_colors[_color]
    if type(_color) == int: return fg_color(_color)
    elif type(_color) == tuple: return rgb_color(_color)
    elif _color[0] == '#': return rgb_color(hex_to_rgb(_color))
    elif _color[0] == '~': return fg_color(rgb_to_int(hex_to_rgb(_color)))

styles = {
    "b": '\033[1m',  # 加粗
    "d": '\033[2m',  # 暗淡
    "/": '\033[3m',  # 斜体
    "_": '\033[4m',  # 下划线
    "s": '\033[5m',  # 闪烁
    "i": '\033[7m',  # 反色
    "h": '\033[8m',  # 隐藏
}

def _cstr(_str, _color = None, _style = None):
    code = _to_code(_color) if _color is not None else ''
    return _word_style(f"{code}{_str}{RESET}", _style)

if __name__ == '__main__':
    for key in int_colors.keys():
        print(_cstr(f'{key:10}', int_colors[key]) + "- int colors")
    dark_oranges = {
        "柿子橙": 130,  # 暗柿子橙
        "南瓜橙": 94,  # 暗南瓜橙
        "琥珀橙": 136,  # 暗琥珀橙
        "暗砖红": 124,  # 暗砖红色
        "暗棕橙": 130  # 暗棕橙色
    }
    for key in dark_oranges.keys():
        print(_cstr(f'{key:10}', dark_oranges[key]) + "- dark oranges")

    for key in hex_colors.keys():
        print(_cstr(f'{key:10}', hex_colors[key]) + "- hex colors")