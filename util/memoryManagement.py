# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:02:06 2022

@author: Harikala
"""

import torch
import gc
from GPUtil import *
from numba import cuda

import os
import psutil


def free_gpu_cache(MY_DEVICE):
    #print("Initial GPU Usage")
    #gpu_usage()                             

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    #print("GPU Usage after Empty Cache")
    #gpu_usage()
    
    
    
    gc.collect()
    torch.cuda.empty_cache()
    #print("GPU Usage after gc collect")
    #gpu_usage()

    
    
    

                        
                        
def display_gpu_info(MY_DEVICE):
    print("Current GPU Usage ::")
    #gpu_usage()   
    t = torch.cuda.get_device_properties(MY_DEVICE).total_memory
    print('Current Device :',MY_DEVICE)
    print('Total Memroy ',t)
    r = torch.cuda.memory_reserved(MY_DEVICE)
    print('Memroy Reserved ',r)
    a = torch.cuda.memory_allocated(MY_DEVICE)
    print('Memroy Allocated ',a)
    f = r-a  # free inside reserved
    print('Memory Free ',f)
    
    

def gpu_info():
    
    gpus = GPUtil.getGPUs()
  
    for g in gpus:
        print('GPU Name     :',g.name,'     Drive Name   :',g.driver,'     MemoryFree  :',g.memoryFree/g.memoryTotal)
    
    
    
    


def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def format_bytes(bytes):
    if abs(bytes) < 1000:
        return str(bytes)+"B"
    elif abs(bytes) < 1e6:
        return str(round(bytes/1e3,2)) + "kB"
    elif abs(bytes) < 1e9:
        return str(round(bytes / 1e6, 2)) + "MB"
    else:
        return str(round(bytes / 1e9, 2)) + "GB"



                      

