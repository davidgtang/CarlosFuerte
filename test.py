#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 19:29:02 2016

@author: dgt377
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pandas import set_option
set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None

#%%
# Loading Data
filename = './2016-ml-contest/training_data.csv'
training_data = pd.read_csv(filename)
training_data

# Converts to category
training_data['Well Name'] = training_data['Well Name'].astype('category')
training_data['Formation'] = training_data['Formation'].astype('category')
training_data['Well Name'].unique()
training_data.describe() # Gives you stats on things

blind = training_data[training_data['Well Name'] == 'SHANKLE'] # Finds all Shankle points and puts it into blind
training_data = training_data[training_data['Well Name'] != 'SHANKLE'] # All other wells

#%%
# 1=sandstone  2=c_siltstone   3=f_siltstone 
# 4=marine_silt_shale 5=mudstone 6=wackestone 7=dolomite
# 8=packstone 9=ba[]fflestone

# Hex color codes
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
       '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']
#facies_color_map is a dictionary that maps facies labels
#to their respective colors
facies_color_map = {}   # Dictionary # enumerate puts out ind=0, label=SS, and loops through the whole thing
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

#def label_facies(row, labels):
#    #return labels[ row['Facies'] -1]
#    
#    print(row)
#    return
    
def label_facies(row): 
    print(row)
    return np.sum(row)
    
training_data.loc[:,'FaciesLabels'] = training_data.apply(lambda row: label_facies(row, facies_labels), axis=1)
facies_color_map