# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:33:33 2022

@author: Joff Pan
"""

import os
import re
import sys
import time
import traceback
from IPython.core.ultratb import ColorTB, VerboseTB

class Logger(object):
    def __init__(self, file_name = ''):
        self._console = sys.stdout          # save console's address
        sys.stdout = self                   # sys connect to Logger
        self.file_name = file_name
        self.newline = True                 # if print will gene a newline

    def write(self, message):
        self.to_console(message)            # send msg to console
        self.to_file(message)               # send msg to Logger

    def to_console(self, message):
        self._console.write(message)        # write msg in console

    def to_file(self, message):
        if_write_file = True
        if message.find('\r') == 0:
            if self.newline: self.newline = False
            self.line_message = message     # save line msg
            if_write_file = False
        elif not self.newline:              # combine last line msg with newline msg
            message = self.line_message + message
            self.newline = True
        if if_write_file:
            with open(self.file_name, 'a') as logfile:
                logfile.write(re.sub(r'\033\[.*?m', '', message))      # write msg in Logger

    def flush(self):
        pass
    
    def reset(self):
        sys.stdout = self._console




if __name__ == '__main__':
    # select a path to save log
    log_path = '../Log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # set a log name acorrding to the system time
    log_file_name = log_path + 'log-' + time.strftime("%Y.%m.%d-%Hh %Mm %Ss", time.localtime()) + '.log'
    
    
    print('before')
    
    logger = Logger(log_file_name)
    
    def overwrite_console(p_str):
        logger.to_console("{}\r".format(p_str))
    
    try:
        logger.to_file("log only!\n")
        logger.to_console("console only!\n")
        print("both log and console")
        print(log_file_name)
        
        s = 'abcdefghijkl'
        
        for x in range(0,5):
            overwrite_console(s[x]*(5-x))
    
        print(2/0)
    except:
        # color = ColorTB()
        ver = VerboseTB()
        exc = sys.exc_info()

        logger.to_file(traceback.format_exc())
        logger.reset()
        
        for i, _str in enumerate(ver.structured_traceback(*exc)):
            if i == 1: _str = _str[_str.find('Traceback'):]
            print(_str)
    finally:
        logger.reset()
