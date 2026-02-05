# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import time
import numpy as np

from joff._nn._attr import _update_dict, _set_func_pr, _update_module_attr
from joff._load._make_dataset import _make_dataset
from joff._load._make_dataset import Data
from joff._func._task import _get_task_dt
from joff._save._save_df import _fd_perf2v
from joff._save._save_result import save_runner_result
from joff.customize import Runner_dt, _run_dt

import os
import importlib.util
import sys

def import_all_classes_from_directory(directory_path):
    """
    动态导入指定目录及其子目录下所有.py文件中定义的所有类。

    参数:
        directory_path (str): 目标目录的路径

    返回:
        dict: 一个字典，键为类名，值为对应的类对象
    """
    classes_dict = {}

    # 遍历目录树[citation:1][citation:8]
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                file_path = os.path.join(root, file)

                # 动态加载模块[citation:8]
                module_name = os.path.splitext(file)[0]
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module  # 缓存模块[citation:8]
                try:
                    spec.loader.exec_module(module)
                except Exception as e:
                    print(f"Error loading module {module_name} from {file_path}: {e}")
                    continue

                # 提取模块中定义的类[citation:8]
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    # 判断是否为类，并且是在当前模块中定义的[citation:8]
                    if isinstance(attr, type) and attr.__module__ == module_name:
                        classes_dict[attr_name] = attr

    return classes_dict

from_main = import_all_classes_from_directory('../_model')
from_sub = import_all_classes_from_directory('../../_model')
classes = dict(from_main, **from_sub)

# 现在你可以通过类名来使用这些类
# for class_name, class_obj in classes.items():
#     print(f"Found class: {class_name}")
    # 如果需要，你可以在这里实例化类
    # instance = class_obj()

class Runner():
    def __init__(self, **kwargs):
        p = _update_dict(Runner_dt, kwargs)
        for key in p.keys():
            setattr(self, key, p[key])
            if key in Runner_dt.keys() and key in kwargs.keys(): del kwargs[key]
        self.basic_addi_name = kwargs['_addi_name'] if '_addi_name' in kwargs.keys() else ''
        self.model_kwargs = kwargs

    # get model from dt
    def _get_model(self, dataset_id, model_id, act_id, **kwargs):
        # load dataset
        Dataset_name = 'Data' + str(dataset_id)
        if not hasattr(self, 'D'):
            if type(self.load_datas[dataset_id - 1]) == Data: self.D = self.load_datas[dataset_id - 1]
            else:
                self.D = _make_dataset(**self.load_datas[dataset_id - 1])
        if hasattr(self.D, 'name'): Dataset_name = self.D.name

        model_class = self.models[model_id - 1]
        # record attrs
        new_kwargs = self.model_kwargs.copy()
        new_kwargs = dict(new_kwargs, **kwargs)
        # 默认：de_struct = struct.inverse()、 di_struct = struct
        for i, attr_str in enumerate(['struct', 'act', 'de_act', 'di_act']):
            if i == len(list(self.acts[act_id - 1])): break
            attr = self.acts[act_id - 1][i]
            if i == 0: attr = self.structs[attr - 1]
            new_kwargs[attr_str] = attr
        if self.if_add_run_info: new_kwargs['_addi_name'] = self.basic_addi_name + '_' + Dataset_name + '_Act' + str(act_id)
        if self.fd_prs is not None:  new_kwargs['fd_pr'] = self.fd_prs[int(np.mod(model_id - 1, len(self.fd_prs)))]
        # model = eval(model_class + '(**new_kwargs)')
        new_kwargs['D'] = self.D
        model = classes[model_class](**new_kwargs)
        model.load(dataset = self.D)
        model.D.name = Dataset_name
        return model

    # run Runner (all the models)
    def run(self, dataset_id, **kwargs):
        model = None
        cnt = 1
        # loop struct(act)
        for act_id in range(1, len(self.acts) + 1):
            act_str = ''
            act_row = list(self.acts[act_id - 1])
            for i in range(1, len(act_row)):
                if i == 0: continue
                for act in act_row[i]: act_str += act
                act_str += ' '

            # loop model(model_id)
            for model_id in range(1, len(self.models) + 1):
                del model
                time.sleep(1.0)
                model = self._get_model(model_id, dataset_id, act_id)
                save_path = '../Result/Runner{} ({} Models{} Acts{})'.format(self.basic_addi_name, model.D.name,
                                                                    len(self.models), len(self.acts))
                p = {}
                p = _set_func_pr(model, _get_task_dt(model.kwargs['task'], _run_dt), **kwargs)
                p['if_save_eval_df'] = False
                p['if_simple_msg'] = True
                p['if_save_plot'] = False
                p['save_path'] = save_path + '/' + model.name + '_Act' + str(act_id)
                p['_save_path'] = p['save_path']
                _update_module_attr(model, p)

                print("\n\033[31m★ Running {} - run count <{}/{}>\033[0m".format(model._name, cnt, len(self.acts)*len(self.models)))

                model.run(**p)
                model.act_str = act_str[:-1]
                model.runner_cnt = cnt
                save_runner_result(model, save_path, 'Runner.xlsx')

                time.sleep(1.0)
            cnt += 1


if __name__ == '__main__':
    p = {'models': ['DAE','VAE'],
         'structs': [[-1,'/2','/2'], [-1, '/3', '/1.5']],
         'acts': [(1, ['g','s'],['a']), 
                  (1, ['s','t'],['s']), 
                  (2, ['a', 'g'],['s','a'])],
         'load_datas': [{'special': 'CSTR/fd'},
                        {'special': 'TE', 'stack': 40},
                        {'special': 'HY/fd', 'stack': 10}],
         'fd_prs': [['re-T2&Q-kde'], ['re-T2&Q-kde']
                    ]
        }
    p['plot_whole_fd'] = True
    # p['use_bn'] = True
    p['auto_drop'] = True
    p['opt'] = 'Adam'
    R = Runner(**p)
    dae = R._get_model(3,2,1)
    dae.run(e = 50)