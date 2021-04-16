#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
import cv2
from PIL import Image
try:
    import torch
except:
    print('CANNOT import torch !')

_MUTE = False
class Singleton(object):
    _instance = None

    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls)  
        return cls._instance
    
class GlobalDict(Singleton):
    def __init__(self):
        self.dict = {}
        
    def __getitem__(self, key):
        if key in self.dict:
            return self.dict[key]
        else:
            debug(info='No key called '+ str(key), info_type='error')
            return None
    
    def __setitem__(self, key, value):
        self.dict[key] = value   
    
def debug(info, info_type='debug'):
	if info_type == 'error':
		print('\033[1;31m ERROR:', info, '\033[0m')
	elif info_type == 'success':
		print('\033[1;32m SUCCESS:', info, '\033[0m')
	elif info_type == 'warning':
		print('\033[1;34m WARNING:', info, '\033[0m')
	elif info_type == 'debug':
		print('\033[1;35m DEBUG:', info, '\033[0m')
	else:
		print('\033[1;36m MESSAGE:', info, '\033[0m')

def write_params(log_path, parser, description=None):
    opt = parser.parse_args()
    options = parser._optionals._actions
    with open(log_path+'params.md', 'w+') as file:
        file.write('# Params\n')
        file.write('********************************\n')
        file.write('Time: '+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n')
        if description is not None:
            file.write('**Description**: '+description+'\n')
        
        file.write('| Param | Value | Description |\n')
        file.write('| ----- | ----- | ----------- |\n')
        for i in range(len(parser._optionals._actions)):
            option = options[i]
            if option.dest != 'help':
                file.write('|**'+ option.dest+'**|'+str(opt.__dict__[option.dest])+'|'+option.help+'|\n')
        file.write('********************************\n\n')


def check_shape(batch, msg=None):
    if _MUTE: return
    if isinstance(batch, dict):
        for key in batch.keys():
            print('\033[1;36m', key, ':', batch[key].shape, '\033[0m')
    elif isinstance(batch, torch.Tensor):
        if msg is not None:
            print('\033[1;36m', msg, ':', batch.shape, '\033[0m')
        else:
            print('\033[1;36m', batch.shape, '\033[0m')
    elif isinstance(batch, (tuple, list)):
        print('\033[1;36m', len(batch), '\033[0m')
        if len(batch) < 10:
            for item in batch:
                if isinstance(item, torch.Tensor):
                    print('\033[1;36m', item.shape, '\033[0m')
                else:
                    print('\033[1;36m', type(item), '\033[0m')
    else:
        if msg is not None:
            print('\033[1;36m', msg, ':', batch, '\033[0m')
        else:
            print('\033[1;36m', batch, '\033[0m')

def to_device(batch, device):
    if isinstance(batch, dict):
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

def set_mute(state):
    global _MUTE
    assert isinstance(state, bool)
    _MUTE = state