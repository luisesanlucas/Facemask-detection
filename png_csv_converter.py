#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:55:29 2021

@author: estefania
"""
from skimage import io
import pandas as pd
import os.path
import os


my_df=pd.DataFrame() #makes df

#insert directory of pics
for file in os.listdir('/Users/estefania/Documents/Machine_Learning/Group_Project/pic_folder'):
    if file.endswith('.png'):
        img = io.imread("{}".format(file),as_gray=True)/255
        my_df["{}".format(file)] = img.flatten() 
        print("The image pixels dimensions are ")
        print(my_df.shape)
   
my_df= my_df.T #makes each png a row

#type where you want the csv stored
my_df.to_csv('/Users/estefania/Documents/Machine_Learning/Group_Project/pd.csv', na_rep="NAN!",sep=',')   
    

