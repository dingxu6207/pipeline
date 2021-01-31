# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:21:40 2021

@author: dingxu
"""

import pandas as pd
import numpy as np

def smooth(csv_path,weight=0.85):
    data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
 
 
    save = pd.DataFrame({'Step':data['Step'].values,'Value':smoothed})
    save.to_csv('smooth_'+csv_path)
 
#file = 'run-train-tag-epoch_loss.csv'
file = 'run-validation-tag-epoch_loss.csv'
if __name__=='__main__':
    smooth(file)
    