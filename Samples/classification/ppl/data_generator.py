# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 08:41:14 2021

@author: Adrian
""" 

f = open("data.csv", 'w')
for i in range(4096):
    for j in range(524288):
        f.write("1; ")
    f.write("\n")
    
f.close()