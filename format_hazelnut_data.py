#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:28:02 2019

Author: David O'Callaghan
"""

import pandas as pd

def format_hazelnut_data(hazelnuts_file_path):
    """
    This function converts the raw data from hazelnuts.txt to a
    Pandas DataFrame with the target variable (variety) as the 
    last column and attribute in every other column. The sample_id
    is set to the index. Each row is a sample in the data set.
    """
    raw_data = pd.read_csv(hazelnuts_file_path, delimiter='\t', header=None)
    
    #transpose
    data_transpose = raw_data.transpose()
    
    #add column names
    column_names = ['sample_id', 'length', 'width', 'thickness', 
                    'surface_area', 'mass', 'compactness', 
                    'hardness', 'shell_top_radius', 'water_content', 
                    'carbohydrate_content', 'variety']
    data_transpose.columns = column_names
    
    #convert data to floats
    for column_name in column_names[:-1]:
        data_transpose[column_name] = pd.to_numeric(data_transpose[column_name]) 
    
    #set the index to the sample ID and sort
    data_indexed = data_transpose.set_index('sample_id')
    data_formatted = data_indexed.sort_index(axis=0) 

    return data_formatted
