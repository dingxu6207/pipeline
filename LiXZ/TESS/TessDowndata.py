# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 06:20:45 2021

@author: dingxu
"""

#coding=utf-8
#from ctypes import *
'''
利用python的ctypes模块可以在python中调用c/c++写的代码。
但是c/c++写的代码要编译成dll文件，在dll中导出你想在python中调用的函数或变量。
'''
import os
import sys
import ftplib  # 定义了FTP类,实现ftp上传和下载
 
class myFtp:
    # 定义一个ftp对象
    ftp = ftplib.FTP()
 
    def __init__(self, host, port=21): # 类似c++构造函数
        self.ftp.connect(host, port)
 
    # 登陆
    def Login(self, user, passwd):
        self.ftp.login(user, passwd)
        print(self.ftp.welcome)  # 打印出欢迎信息
 
    # 下载单个文件
    # LocalFile为本地文件路径（带文件名）,RemoteFile为ftp文件路径(不带文件名)
    def DownLoadFile(self,LocalFile,RemoteFile):
        if(os.path.exists(LocalFile)):
            os.remove(LocalFile)
        file_handler = open(LocalFile, 'wb')
        print(file_handler)
        # 下载ftp文件
        self.ftp.retrbinary('RETR ' + RemoteFile, file_handler.write)
        file_handler.close()
        return True
 
    # 下载整个目录下的文件
    # LocalDir为本地目录（不带文件名）,RemoteDir为远程目录(不带文件名)
    def DownLoadFileTree(self, LocalDir, RemoteDir):
        print("RemoteDir:", RemoteDir)
        if not os.path.exists(LocalDir):
            os.makedirs(LocalDir)
        # 打开该远程目录
        self.ftp.cwd(RemoteDir)
        # 获取该目录下所有文件名，列表形式
        RemoteNames = self.ftp.nlst()
        for file in RemoteNames:
            Local = os.path.join(LocalDir, file)  # 下载到当地的全路径
            print(self.ftp.nlst(file))  # [如test.txt]
            if file.find(".") == -1:  #是否子目录 如test.txt就非子目录
                if not os.path.exists(Local):
                    os.makedirs(Local)
                self.DownLoadFileTree(Local, file)  # 下载子目录路径
            else:
                self.DownLoadFile(Local, file)
        self.ftp.cwd("..")  # 返回路径最外侧
        return
 
    # 关闭ftp连接
    def close(self):
        self.ftp.close()
 
 
filehtp = 'https://archive.stsci.edu/missions/tess/tid/s0033/0000/0000/0461/8473/'
if __name__ == "__main__":
    # 建立一个ftp连接
    ftp = myFtp('filehtp')
    # 登陆
    #ftp.Login('user', 'password')
    # 下载单个文件
    #ftp.DownLoadFile('D:/data/test.txt', '/home/test.txt')
    # 下载整个目录下的文件
    ftp.DownLoadFileTree('/missions/tess/tid/s0033/0000/0000/0461/', '/8473/')
    # 关闭ftp连接
    ftp.close()
    print("over")
    