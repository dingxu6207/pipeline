# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 22:03:08 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

phraseflux = np.loadtxt('mag.txt')

phrase = phraseflux[:,0]

flux = phraseflux[:,1]


fft_size = 345
 

fy = np.fft.fft(flux) / fft_size

plt.figure(0)
plt.plot(np.clip(20*np.log10(np.abs(fy[:20])), -120, 120), "o")
plt.xlabel("frequency bin")
plt.ylabel("power(dB)")


# 取FFT计算的结果freqs中的前n项进行合成，返回合成结果
def fft_combine(freqs, n, loops=1):
    length = len(freqs) * loops
    data = np.zeros(length)
    index = loops * np.arange(0, length, 1.0) / length * (2 * np.pi)
    for k, p in enumerate(freqs[:n]):
        if k != 0: p *= 2 # 除去直流成分之外，其余的系数都*2
        data += np.real(p) * np.cos(k*index) # 余弦成分的系数为实数部
        data -= np.imag(p) * np.sin(k*index) # 正弦成分的系数为负的虚数部
        #print(data)
    #print(np.real(p), -np.imag(p))
    
    return index, data


# 绘制原始的三角波和用正弦波逐级合成的结果，使用取样点为x轴坐标
temp = []
plt.figure(1)
plt.plot(flux,'.')
for i in range(5):
    index, data = fft_combine(fy, i+1, 1)  # 计算1个周期的合成波形
    plt.plot(data,label = "N=%s" % i)
    temp.append(data)

print(fy[:5])  

#plt.plot(nihedata, '.')
plt.legend()
plt.title("partial Fourier series of triangle wave")
plt.show()



dataflux =temp[2]
phrasefluxdata = np.vstack((phrase, dataflux))

np.savetxt('N2data.txt', phrasefluxdata.T)

N4data = np.loadtxt('N2data.txt')

phrase = N4data[:,0]
Nflux = N4data[:,1]

plt.figure(5)
plt.plot(phrase, Nflux)
plt.plot(phrase, flux,'.')
