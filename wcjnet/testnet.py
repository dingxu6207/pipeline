# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 00:09:05 2020

@author: dingxu
"""

import os 
import time

cmd0 = 'D:\\iperf-3.1.3-win64\\iperf3.exe -c 98.234.59.155'
cmd1 = 'D:\\iperf-3.1.3-win64\\iperf3.exe -c 98.234.59.155 -R'

r_v = os.system(cmd0) 
f = os.popen(cmd0)  
data = f.readlines()  
f.close()  


rv1 = os.system(cmd1) 
f1 = os.popen(cmd1)  
data1 = f1.readlines()  
f1.close()  


str_time = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())

local_time = str_time+'\n'

file = str_time[-8:-6]+'-'+str_time[-5:-3]+'-'+str_time[-2:]+'.txt'
webf = open(file,"w")
webf.writelines(local_time)
webf.writelines(data)
webf.writelines(data1)

webf.close()

