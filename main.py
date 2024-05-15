#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:26:04 2024

@author: ruchak
"""

import pandas as pd
import functions as func


path_to_save = '/home/ruchak/solas_probe_data/combined_results/results_round7.b/'

#preprocess the kpiv data
#read the OWL DVL coherent data
kpiv = pd.read_csv('/home/ruchak/solas_probe_data/combined_results/cleaned_datasets/STST OWL combine_coherent.csv')
kpiv_preprocessing = func.PreProcessing(kpiv)
#drop na
kpiv_preprocessing.drop_na()   
#dropping duplicates
kpiv = kpiv_preprocessing.dropping_duplicates() 
#drop not useful columns
kpiv = kpiv.drop(columns = ['HD_ID', 'DVL duplicate count', 'Source Table'])

#getting the DVL data
solas_dvl = pd.read_csv('/home/ruchak/solas_probe_data/combined_results/cleaned_datasets/solas_dvl.csv')
#subset the solas_dvl dataset and join it with the kpiv dataset
solas_dvl = solas_dvl[['HD_NUM', 'CERT OR LIFE 202']] 

#join the KPIV and KPOV data  
joined = pd.merge(kpiv, solas_dvl, left_on = 'hd_id_10', right_on = 'HD_NUM').drop(columns = 'HD_NUM')
#drop duplicates
joined_preprocessing = func.PreProcessing(joined)
joined_preprocessing.dropping_duplicates() 
#drop any columns that have a single unique value
joined = joined_preprocessing.dropping_unique_vals()

#get the shipping date, contacts and solas sub-group
add_info = pd.read_csv('/home/ruchak/solas_probe_data/combined_results/cleaned_datasets/STST OWL combine_coherent_wafer_summary.csv')
add_info = add_info[['wafer_lot', 'Solas sub group', 'OWL_SHIP_DATE', 'Laser Design']]
#join the data with the KPIV and KPOV data
joined = pd.merge(joined, add_info, on = 'wafer_lot')
#save the data
joined.to_csv('/home/ruchak/solas_probe_data/combined_results/cleaned_datasets/kpiv_kpov_data.csv', index = False)

#remove selected categorical columns
drop_cols = ['hd_id_10', 'wafer_lot', 'wafer_lot_5', 'SHD_ID_0', 'hd_id_6', 'SHD_ID', 'hd3', 'BLKID']
#we keep this, so we can add this data after all the processing
categorical_cols = joined[drop_cols] 
#drop the categorical data       
joined = joined.drop(columns = drop_cols)

#feature engineering
feat_eng = func.feature_engineering(joined)
#one hot encode Laser Design
feat_eng.one_hot_encoding('Laser Design')
#handle the shipping date column
joined = feat_eng.date_handling('OWL_SHIP_DATE', 'OWL_SHIP')

#get all the different solas sub groups
all_solas_sub_group = joined['Solas sub group'].unique().tolist()

#we now need to analyze the different solas sub-groups separately
for sub_group in all_solas_sub_group:
    print(sub_group)
    
    #subset the dataframe based on the sub group
    df_group = joined[joined['Solas sub group'] == sub_group]
    
    model_df_group = func.models(df_group, sub_group, path_to_save)
    
    #drop all columns with only 1 unique value
    model_df_group.dropping_unique_vals()
    
    #get correlations
    model_df_group.get_correlations()
    
    train, test = model_df_group.get_train_test_set()
    
    #Add the heads, and other important information and save the results.
    len_train = len(train)
    len_test = len(test)
    train = pd.merge(categorical_cols, train, how = 'inner', left_index = True, right_index = True)
    test = pd.merge(categorical_cols, test, how = 'inner', left_index = True, right_index = True)
    
    #assert that the lengths before and after adding the categorical columns is same
    assert len_train == len(train)
    assert len_test == len(test)
    
    #save the training and testing datasets
    train.to_csv(path_to_save + sub_group + '/train_with_hd.csv', index = False)
    test.to_csv(path_to_save + sub_group + '/test_with_hd.csv', index = False)
    
    #standardize the data
    model_df_group.standardize()
    
    #train the XGBoost model
    model_df_group.XGBoost_training()
    model_df_group.feature_importances()
    
    #retrain the XGBoost model
    model_df_group.XGBoost_training(save_results =  True)
    
    #get ensemble model results
    model_df_group.get_tree_results()
    
    #get metrics for individual trees
    model_df_group.get_metrics()
    
    break