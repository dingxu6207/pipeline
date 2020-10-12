# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:25:35 2020

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

phraseflux = np.loadtxt('mag.txt')

phrase = phraseflux[:,0]

flux = phraseflux[:,1]

'''
plt.plot(phrase, flux, '.')

ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
'''

# 产生size点取样的三角波，其周期为1
def triangle_wave(size):
    x = np.arange(0, 1, 1.0/size)
    y = np.where(x<0.5, x, 0)
    y = np.where(x>=0.5, 1-x, y)
    return x, y
 
def square_wave(size):
    x = np.arange(0, 1, 1.0/size)
    y = np.where(x<0.5, 1.0, 0)
    return x, y


fft_size = 256
 
# 计算三角波和其FFT
x, y = triangle_wave(fft_size)
fy = np.fft.fft(y) / fft_size


# 绘制三角波的FFT的前20项的振幅，由于不含下标为偶数的值均为0， 因此取
# log之后无穷小，无法绘图，用np.clip函数设置数组值的上下限，保证绘图正确
plt.figure(0)
plt.plot(np.clip(20*np.log10(np.abs(fy[:20])), -120, 120), "o")
plt.xlabel("frequency bin")
plt.ylabel("power(dB)")
plt.title("FFT result of triangle wave")

# 取FFT计算的结果freqs中的前n项进行合成，返回合成结果，计算loops个周期的波形
def fft_combine(freqs, n, loops=1):
    length = len(freqs) * loops
    data = np.zeros(length)
    index = loops * np.arange(0, length, 1.0) / length * (2 * np.pi)
    for k, p in enumerate(freqs[:n]):
        if k != 0: p *= 2 # 除去直流成分之外，其余的系数都*2
        data += np.real(p) * np.cos(k*index) # 余弦成分的系数为实数部
        data -= np.imag(p) * np.sin(k*index) # 正弦成分的系数为负的虚数部
    return index, data


# 绘制原始的三角波和用正弦波逐级合成的结果，使用取样点为x轴坐标
plt.figure(1)
plt.plot(y, label="original triangle", linewidth=2)
for i in [0,1,2,3,4,5,6,7,8,9,10]:
    index, data = fft_combine(fy, i+1, 2)  # 计算两个周期的合成波形
    plt.plot(data, label = "N=%s" % i)
plt.legend()
plt.title("partial Fourier series of triangle wave")
plt.show()


plt.figure(2)
plt.plot(x,y,'.')

