# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import os
import torch
from joff._save._print_style import _clickable_file_link, _cstr

def _make_path(path, _type):
    if not os.path.exists(path): os.makedirs(path)
    return path + '/' + _type.capitalize()
    
def _save_module_para(self, _type = 'last', sub = None):
    # 'sub' only works for 'fd' + 'best'
    if self.task != 'fd':
        file_path = _make_path(self._save_path, _type)
    
    elif _type == 'best':
        assert sub is not None
        _save_path = self._save_path + '/' + self.fd_pr_name[sub]
        file_path = _make_path(_save_path, _type)
        
    else:
        if not os.path.exists(self._save_path): os.makedirs(self._save_path)
        file_path = self._save_path + '/' + _type.capitalize()
    
    if self.save_mode == 'para':
        torch.save(self.state_dict(), file_path)
    else:
        file_path.insert(0, 'Module-')
        torch.save(self, file_path)
        
    if _type == 'last':
        clickable_text = _clickable_file_link(self._save_path + '/')
        print(f"\nSave {_cstr(f'model paras [{_type.capitalize()}]','珊瑚橙','/')} in '{clickable_text}'")

def _save_module(self, file):
    torch.save(self, file)

if __name__ == '__main__':
    def A():
        return B()
    def B():
        print('hhh')
    print(A())
    print('Joff_Package'[:-2])