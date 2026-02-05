import os
import numpy as np

def _save_ts(TS, thrd, save_path, phase):
    _TS = np.insert(TS,0, thrd)
    if not os.path.exists(save_path): os.makedirs(save_path)
    np.savetxt(save_path + '/' + phase + '_ts.csv',
               _TS,
               fmt='%f', delimiter=',')