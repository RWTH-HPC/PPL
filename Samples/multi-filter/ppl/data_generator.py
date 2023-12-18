# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 08:41:14 2021

@author: Adrian
""" 

f = open("image.txt", 'w')
for i in range(8194):
    for j in range(8194):
        f.write("1\n")
    
f.close()