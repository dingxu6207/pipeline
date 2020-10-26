# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 01:26:54 2020

@author: dingxu
"""

import os

cmd = 'python'+' '+'testnet.py'
#os.system(cmd)


import time
def sleeptime(hour,min,sec):
    return hour*3600 + min*60 + sec

second = sleeptime(0,10,0)
while 1==1:
    time.sleep(second)
    os.system(cmd)
    
    print('it is ok!')
    