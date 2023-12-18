# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 08:41:14 2021

@author: Adrian
""" 

f = open("batch.txt", 'w')
for i in range(64):
    for j in range(262144):
        f.write("1\n")
    
f.close()