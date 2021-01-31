# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:23:29 2021

@author: dingxu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file1 = 'smooth_run-train-tag-epoch_loss.csv'
df1 =  pd.read_csv(file1, sep=',')

file2 = 'smooth_run-validation-tag-epoch_loss.csv'
df2 =  pd.read_csv(file2, sep=',')

npdf1 = np.array(df1)
npdf2 = np.array(df2)

plt.figure(0)
A, = plt.plot(npdf1[:,1], npdf1[:,2])
B, = plt.plot(npdf2[:,1], npdf2[:,2])
plt.xlabel('epoch',fontsize=14)
plt.ylabel('epoch_loss',fontsize=14)

legend=plt.legend(handles=[A,B],labels=['train','validation'])

axins = plt.inset_axes((0.2, 0.2, 0.4, 0.3))