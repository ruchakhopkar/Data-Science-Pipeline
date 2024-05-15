#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:26:15 2024

@author: ruchak
"""
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import ClusterCentroids
from scipy.stats import pearsonr
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import numpy as np
import xgboost as xgb
from joblib import dump
plt.rcParams.update({'font.size': 22})

def get_metrics(y_train, y_train_predict):
    '''
    This function using the actual and predicted data to get some evaluation metrics on the data

    Parameters
    ----------
    y_train : numpy array
        The actual data
    y_train_predict : numpy array
        The predicted results

    Returns
    -------
    None.

    '''
    print('Accuracy score on dataset ', accuracy_score(y_train, y_train_predict))
    print('Precision score on dataset ', precision_score(y_train, y_train_predict))
    print('Recall score on dataset ', recall_score(y_train, y_train_predict))
    print('F1 score on dataset ', f1_score(y_train, y_train_predict))

class PreProcessing:
    def __init__(self, df):
        '''
        This is the constructor for the preprocessing class. The preprocessing class achieves the following:
            1. Dropping nan values
            2. Dropping duplicate rows and columns
            

        Parameters
        ----------
        df : DataFrame
            The DataFrame that needs to be preprocessed

        Returns
        -------
        None.

        '''
        self.df = df
        
    def drop_na(self):
        '''
        This function will drop a column if it has more than 50% of its values as nan
        Else, it will drop any rows with nan values if present
    
        Returns
        -------
        df : Pandas DataFrame
            The modified dataframe without any nan values
    
        '''
        #drop na
        self.df = self.df.dropna(thresh = 0.5*len(self.df), axis = 1).dropna()
        return self.df
    
    def dropping_duplicates(self):
        '''
        This function will first drop any duplicate rows and then drop any duplicate columns

        Returns
        -------
        df : Pandas DataFrame
            The modified dataframe without any duplicate values
        '''
        #drop duplicates
        dtypes = self.df.dtypes.to_dict()    #save all the dtypes, because when we transpose the dataset, we lose the information of dtypes
        self.df = self.df.drop_duplicates()   #drop any duplicate rows
        self.df = self.df.T.drop_duplicates().T   #drop any duplicate columns
        dtypes = dict((k,v) for k,v in dtypes.items() if k in set(self.df.columns.tolist()))
        self.df = self.df.astype(dtypes)  #change the dtypes to the ones we saved before

        return self.df
    
    def dropping_unique_vals(self):
        '''
        If a column has only 1 unique value, that doesnt add any extra information into the model.
        Hence a good way of dealing with this, is to drop the column with only 1 unique value.
        

        Returns
        -------
        df : Pandas DataFrame
            The modified dataframe without any columns that have a single unique value

        '''
        #drop all columns with only 1 unique value
        for col in self.df.columns:
            if len(self.df[col].unique()) == 1:
                self.df.drop(col,inplace=True,axis=1)
        
        return self.df
    
class feature_engineering:
    def __init__(self, df):
        '''
        This is the constructor for the feature engineering.

        Parameters
        ----------
        df : Pandas DataFrame
            The joined DataFrame that has both the KPIV and KPOV data

        Returns
        -------
        None.

        '''
        self.df = df
    
    def one_hot_encoding(self, col):
        '''
        This function is used to one hot encode any categorical variables in the dataset

        Parameters
        ----------
        col : string
            The column name which needs to be one hot encoded

        Returns
        -------
        Pandas DataFrame
            The changed DataFrame with the categorical column one-hot encoded

        '''
        #one hot encode the laser design
        ohe = OneHotEncoder(sparse_output = False)
        ohe_laser_design = ohe.fit_transform(self.df[col].values.reshape(-1,1)) 
        cols = [col + ' ' + x for x in ohe.get_feature_names_out([col])]
        ohe_laser_design = pd.DataFrame(data = ohe_laser_design, columns = cols)
        self.df = pd.concat([self.df, ohe_laser_design], axis = 1)
        self.df = self.df.drop(columns = col)
        return self.df
    
    def date_handling(self, col, rename_col):
        '''
        This function is used to handle the date columns in the DataFrame.
        This is done by extracting information from the data like day, month, day of the week, and the week number

        Parameters
        ----------
        col : String
            The column name of the date column
        rename_col : String
            The name to suffix to the different components of the date column

        Returns
        -------
        Pandas DataFrame
            The DataFrame with the changed date column

        '''
        #handle the shipping date
        self.df = self.df.sort_values(by = ['Solas sub group', col])
        self.df[col] = pd.to_datetime(self.df[col], errors = 'coerce')
        self.df[rename_col + '_DAY'] = self.df[col].dt.day
        self.df[rename_col + '_MONTH'] = self.df[col].dt.month
        self.df[rename_col + '_DAY_OF_WEEK'] = self.df[col].dt.weekday
        self.df[rename_col + '_WEEK'] = self.df[col].dt.isocalendar().week.astype('float')
        self.df = self.df.drop(columns = col)
        return self.df

class models:
    def __init__(self, df, sub_group, path_to_save):
        self.df = df
        self.y = 'CERT OR LIFE 202'
        self.sub_group = sub_group
        self.path_to_save = path_to_save
    
    def dropping_unique_vals(self):
        '''
        If a column has only 1 unique value, that doesnt add any extra information into the model.
        Hence a good way of dealing with this, is to drop the column with only 1 unique value.
        

        Returns
        -------
        df : Pandas DataFrame
            The modified dataframe without any columns that have a single unique value

        '''
        #drop all columns with only 1 unique value
        for col in self.df.columns:
            if len(self.df[col].unique()) == 1:
                self.df.drop(col,inplace=True,axis=1)
        
        return self.df
    
    def undersampling(self):
        '''
        This function performs an undersampling of the data. 
        Correlations depend on the sample size, and hence this becomes necessary. 

        Returns
        -------
        None.

        '''
        #under sample the class 0 dataset
        cc = ClusterCentroids(random_state = 42)    #We use the imblearn library to under-sample the data
        self.X_resampled, y_resampled = cc.fit_resample(self.df[[x for x in self.df.columns if x!= self.y]], \
                                                   self.df[self.y])
        self.X_resampled[self.y] = y_resampled
        
        
    def get_correlations(self):
        '''
        This function will generate and save the correlations between KPIV space and KPOV.
        We use the pearson correlation coefficient to find the linear correlation. 
        If the p-value is <0.05 or if we are more than 95% confident about the correlation, the boxplots 
        are also saved. 

        Returns
        -------
        None.

        '''
        self.undersampling()
        
        pvalues, correlations, xs, ys = [], [], [], []
        for i, col in enumerate(self.X_resampled.columns.tolist()):
            if col!= self.y:    
                c, p = pearsonr(self.X_resampled[col].values, self.X_resampled[self.y].values)   
                pvalues.append(p)
                correlations.append(c)
                xs.append(col)
                ys.append(self.y)
                
                if p<0.05:
                    #get the boxplots
                    sns.boxplot(data = self.X_resampled, x = self.y, y = col, hue = self.y)
                    os.makedirs(self.path_to_save + self.sub_group + '/correlations/boxplots/', mode = 0o777, exist_ok = True)
                    plt.savefig(self.path_to_save + self.sub_group + '/correlations/boxplots/' + col.replace('/', '_') + '.png', bbox_inches = 'tight')
                    plt.close()
        
        os.makedirs(self.path_to_save + self.sub_group + '/correlations/', mode = 0o777, exist_ok = True)
        corr_df = pd.DataFrame()
        corr_df['Input Var'] = xs
        corr_df['Output Var'] = ys
        corr_df['P-Values'] = pvalues
        corr_df['Correlations'] = correlations
        #save the correlation dataframe
        corr_df.to_csv(self.path_to_save + self.sub_group + '/correlations/corr.csv', index = False)
        
    def clustering(self, df_class):
        '''
        This function will cluster the data for a particular class to look for patterns

        Parameters
        ----------
        df_class : Pandas DataFrame
            The DataFrame subsetted by y that needs to be clustered

        Returns
        -------
        X_train : Pandas DataFrame
            The training dataset formed in this subset
        X_test : Pandas DataFrame
            The testing dataset formed in this subset

        '''
        
        #cluster the dataset to check the different patterns
        
        labels = OPTICS(min_samples = 2).fit_predict(df_class)
        df_class['Labels'] = labels
        
        #our training length will be 80% since we are doing 80/20 split
        training_len = int(len(df_class)*0.8)
        #we now get the required number of samples from every cluster
        per_cluster = training_len//len(df_class['Labels'].unique())
        
        X_train, X_test = pd.DataFrame(), pd.DataFrame()
        for i in range(len(df_class['Labels'].unique())):  #repeat for every cluster
            df_subset = df_class[df_class['Labels'] == i]   #subset the class by the label
            if len(df_subset)<per_cluster:  #if the number of samples in the cluster are less than the required samples
                X_train = pd.concat([X_train, df_subset])   #add all these samples to the training dataset
            else:
                train = df_subset.sample(n = per_cluster, random_state = 42)  #sample the number of samples required 
                X_test = pd.concat([X_test, df_subset.loc[df_subset.index.difference(train.index)]])  #add the remaining samples from the cluster to the test dataset
                X_train = pd.concat([X_train, train])
        
        #drop the labels column we just added by clustering
        X_train = X_train.drop(columns = 'Labels')
        X_test = X_test.drop(columns = 'Labels')
        
        #make sure the distribution of train and test are right
        X_test_sample = X_test.sample(n = int(0.8*len(df_class)) - len(X_train), random_state = 42)
        # X_test = X_test[~X_test.index.isin(X_test_sample.index)] #all the rows added in the training set need to be deleted from the testing set
        X_train = pd.concat([X_train, X_test_sample])
        
        return X_train, X_test       
        
        
    def get_train_test_set(self):
        '''
        This function gets the training and testing datasets used for the XGBoost model.
        The function:
            separates 0 and 1 classes
            runs clustering on each, to check for patterns and making sure that we include this diversity in training
            If length of training<80% of the total length, we get some samples from testing
            
        Returns
        -------
        train : Pandas DataFrame
                The dataset used for training
        test: Pandas DataFrame
              The dataset used for testing

        '''
        #get train and test datasets for XGBoost
        df_0 = self.df[self.df[self.y] == 0]
        df_1 = self.df[self.df[self.y] == 1]
        
        #get the clustering for df_0 so we have diversity while training the model
        X_train_0, X_test_0 = self.clustering(df_0)
        X_train_1, X_test_1 = self.clustering(df_1)
        
        #concatenate the 0 and 1 datasets to get the final train and test datasets
        self.train = pd.concat([X_train_0, X_train_1]).sample(frac = 1, random_state = 42)
        self.test = pd.concat([X_test_0, X_test_1]).sample(frac = 1, random_state = 42)
        
        return self.train, self.test
    
    def standardize(self):
        '''
        Standardize the data to have a mean = 0 and variance = 1.
        This helps the model treat all features equally

        Returns
        -------
        None.

        '''
        #standardize the numerical data
        ss = StandardScaler()
        
        #standardize the training data
        num_cols = [x for x in self.train.select_dtypes(include = 'number').columns.tolist() if x!=self.y]
        self.train[num_cols] = ss.fit_transform(self.train[num_cols])
        self.X_train, self.y_train = self.train[[x for x in self.train if x!=self.y]], self.train[self.y].values
        
        #standardize the testing data based on the training data
        self.test[num_cols] = ss.transform(self.test[num_cols])
        self.X_test, self.y_test = self.test[[x for x in self.test if x!=self.y]], self.test[self.y].values
        
        #append the train and test datasets so it can be saved
        total = pd.concat([self.train, self.test], axis = 0)
        total.to_csv(self.path_to_save + self.sub_group + '/' + 'joined.csv', index = False)
        
        #save the standardized datasets
        self.X_train.to_csv(self.path_to_save + self.sub_group + '/' + 'X_train.csv', index = False)
        self.X_test.to_csv(self.path_to_save + self.sub_group + '/' + 'X_test.csv', index = False)
        np.save(self.path_to_save + self.sub_group + '/' + 'y_train', self.y_train, allow_pickle=False)
        np.save(self.path_to_save + self.sub_group + '/' + 'y_test', self.y_test, allow_pickle=False)
        
    def XGBoost_training(self, save_results = False):
        '''
        This function will train the XGBoost on the X_train, y_train and evaluate it on X_test, y_test
        
        Parameters
        ----------
        save_results : Boolean
            Whether to get the feature importances plot and save it.
        
        Returns
        -------
        None.

        '''
        #fit XGBoost model
        
        scale_pos_weight = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
        self.clf = xgb.XGBClassifier(n_estimators = 300, base_score = 0.3, tree_method = 'gpu_hist', max_depth = 8, \
                                random_state = 42, objective = 'binary:logistic', scale_pos_weight = scale_pos_weight)
        self.clf.fit(self.X_train, self.y_train)
        
        #get metrics results for train and test datasets
        print('Training Results')
        y_train_predict = self.clf.predict(self.X_train)
        y_test_predict = self.clf.predict(self.X_test)
        
        get_metrics(self.y_train, y_train_predict)
        print('Testing Results')
        get_metrics(self.y_test, y_test_predict)
        
        if save_results:
            self.y_train_predict = y_train_predict
            self.y_test_predict = y_test_predict
            np.save(self.path_to_save + self.sub_group + '/model_results/test_predictions.npy', y_test_predict)
            np.save(self.path_to_save + self.sub_group + '/model_results/train_predictions.npy', y_train_predict)
        
    
    
    def feature_importances(self):
        '''
        This function will save the feature importances and subset the training and testing datasets to only include
        the top 80% of the most important features. 

        Returns
        -------
        None.

        '''
        #get the most important features
        xgb.plot_importance(self.clf)
        fig = plt.gcf()
        fig.set_size_inches(150, 100)
        os.makedirs(self.path_to_save + self.sub_group + '/model_results/', mode = 0o777, exist_ok = True)
        fig.savefig(self.path_to_save + self.sub_group + '/model_results/importances.png')
        plt.close()
        
        #get the feature importances
        feature_important = self.clf.get_booster().get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())
        
        feat_imp = pd.DataFrame()
        feat_imp['Columns'] = keys
        feat_imp['F score'] = values
        feat_imp = feat_imp.sort_values(by = "F score", ascending=False)
        feat_imp.to_csv(self.path_to_save + self.sub_group + '/model_results/feature_importances.csv', index = False)
        
        #retain the 80% most important columns in train and test set and redo the process
        feat_imp = feat_imp['Columns'].iloc[:int(0.8*len(feat_imp))]
        self.X_train = self.X_train[feat_imp]
        self.X_test = self.X_test[feat_imp]
    
        
    def get_tree_results(self):
        '''
        Get the ensemble results of the XGBoost model for both testing and train+test datasets

        Returns
        -------
        NONE

        '''
        #save all the individual trees
        os.makedirs(self.path_to_save + self.sub_group + '/model_results/trees/', mode = 0o777, exist_ok = True)
        #get the total number of trees
        self.num_t = len(self.clf.get_booster().get_dump())
        for i in range(self.num_t):
            xgb.plot_tree(self.clf, num_trees=i)
            fig = plt.gcf()
            fig.set_size_inches(150, 100)
            fig.savefig(self.path_to_save + self.sub_group + '/model_results/trees/tree' + str(i) + '.png')
            plt.close()
        
        #save the trees as atxt and csv files too
        self.clf.get_booster().dump_model(self.path_to_save + self.sub_group + '/model_results/out.txt')
        df = self.clf.get_booster().trees_to_dataframe()
        df.to_csv(self.path_to_save + self.sub_group + '/model_results/trees.csv', index = False)
        
        
        #get the predictions from individual trees
        preds = []
        for tree in self.clf.get_booster():
            ypred1 = tree.predict(xgb.DMatrix(self.X_test))
            preds.append(ypred1)
        preds = np.vstack(preds).T
        
        def myfunc(z):
            if z>0.5:
                return 1
            else:
                return 0
        
        #vectorize the function for faster results
        myfunc_vec = np.vectorize(myfunc)
        preds_1_0 = myfunc_vec(preds)
        
        #save the classification results
        predictions = pd.DataFrame(data = preds_1_0, columns = ['tree ' + str(i) + ' results' for i in range(self.num_t)])
        predictions['Ground Truth'] = self.y_test
        predictions.to_csv(self.path_to_save + self.sub_group + '/model_results/individual_results.csv', index = False)
        cm = confusion_matrix(self.y_test, self.y_test_predict)
        f, ax = plt.subplots(figsize=(12,12))
        sns.heatmap(cm, annot=True, linewidth=.5, linecolor="r", fmt=".0f", ax = ax)
        plt.savefig(self.path_to_save + self.sub_group + '/model_results/confusion_matrix.png')
        plt.close()
        preds = []
        
        #get the results from all the trees for both training and testing results
        for tree in self.clf.get_booster():
            ypred1 = tree.predict(xgb.DMatrix(pd.concat([self.X_train, self.X_test])))
            preds.append(ypred1)
        preds = np.vstack(preds).T
        
        preds_1_0 = myfunc_vec(preds)
        
        predictions = pd.DataFrame(data = preds_1_0, columns = ['tree ' + str(i) + ' results' for i in range(self.num_t)])
        predictions.to_csv(self.path_to_save + self.sub_group + '/model_results/all_individual_results.csv', index = False)
        
        #save the model
        dump(self.clf, self.path_to_save + self.sub_group + '/model_results/model.joblib')
    
    def get_metrics(self):
        '''
        Get the metrics like accuracy, precision, recall and F1 scores for the individual trees and save it

        Returns
        -------
        None.

        '''
        y_test_actual = np.load(self.path_to_save + self.sub_group + '/y_test.npy', allow_pickle = False)
        all_trees_results = pd.read_csv(self.path_to_save + self.sub_group + '/model_results/individual_results.csv')
        accuracies, precisions, recalls, f1s = [], [], [], []
        
        #get the evaluation metrics for all the trees individually
        for i in range(self.num_t):
            accuracies.append(accuracy_score(y_test_actual, all_trees_results['tree '+str(i) + ' results'])*100)
            precisions.append(precision_score(y_test_actual, all_trees_results['tree '+str(i) + ' results'])*100)
            recalls.append(recall_score(y_test_actual, all_trees_results['tree '+str(i) + ' results'])*100)
            f1s.append(f1_score(y_test_actual, all_trees_results['tree '+str(i) + ' results'])*100)
        
        trees = ['tree ' + str(i) for i in range(self.num_t)]
        
        #put the results in a DataFrame
        tree_results = pd.DataFrame()
        tree_results['Trees'] = trees
        tree_results['Accuracies'] = accuracies
        tree_results['Precisions'] = precisions
        tree_results['Recalls'] = recalls
        tree_results['F1 scores'] = f1s
        tree_results.to_csv(self.path_to_save + self.sub_group + '/evaluating_trees.csv', index = False)
    
    