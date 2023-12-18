# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 08:41:14 2021

@author: Adrian
""" 

f = open("A.txt", 'w')
b1 = open("b1.txt", 'w')
b2 = open("b2.txt", 'w')
b3 = open("b3.txt", 'w')
for i in range(8192):
	b1.write("1\n")
	b2.write("1\n")
	b3.write("1\n")
	for j in range(8192):
		f.write("1\n")
    
f.close()
b1.close()
b2.close()
b3.close()