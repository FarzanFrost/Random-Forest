# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 02:15:08 2022

@author: 19001444
"""

import pandas as pd

#Gathering data
location = "D:\\UCSC\\3rd year\\1st sem\\1. MLNC\\2. Assigment\\Random Forest\\Data\\covtype.data"
data = pd.read_csv( location )

print( data.head() )
print( data.tail() )

import seaborn
seaborn.heatmap( data.corr() , xticklabels = data.columns , yticklabels = data.columns )

data.describe()

x = data.iloc[ : , : 54 ]
y = data.iloc[ : , 54 ]

#spliting the data
from sklearn.model_selection import train_test_split

xTrain , xTest , yTrain , yTest = train_test_split( x , y )

#initailizing the random forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier( n_estimators = 100 , criterion = 'log_loss' , max_depth = 25 , min_samples_split = 15 , min_samples_leaf = 8 , bootstrap = False , max_features = 'log2' )

#training the random forest classifier
rfc.fit( xTrain , yTrain )

#having the random forest classifier to run on test data
predictedResults = rfc.predict( xTest )

#evaluation of the random forest classifier on train data
rfc.score( xTrain , yTrain )

#evaluting the random forest classifier on test data
rfc.score( xTest , yTest )

#analyzing true values vs predicted value using a matrix to understand miss classification classes and their numbers
from sklearn.metrics import confusion_matrix
confusion_matrix( yTest , predictedResults )

#features the mdoel 'thinks' are most important in determining the cover type
pd.DataFrame( rfc.feature_importances_ , index = xTrain.columns ).sort_values( by = 0 , ascending = False )

#don't run after this takes time
#Tuning the Random forest

#Number of trees in random forest
import numpy as np
nEstimators = np.linspace( 100 , 3000 , int( ( 3000 - 100 ) / 200 ) + 1 , dtype = int )

#Number of features to include at every split
maxFeatures = [ 'auto' , 'sqrt' ]

#Maximum number of levels in a decision tree
maxDepth = [ 1 , 5 , 10 , 20 , 50 , 75 , 100 , 150 , 200 ]

#Minimum number of samples required to split a node
minSamplesSplit = [ 1 , 2 , 5 , 10 , 15 , 20 , 30 ]

#Minimum number of smaples required at each leaf node
minSamplesLeaf = [ 1 , 2 , 3 , 4 ]

#Method for selecting samples for training each tree
bootstrap = [ True , False ]

#Criterion
criterion = [ 'gini' , 'entropy' ]

randomGrid = {
    
        'n_estimators' : nEstimators,
        'max_depth' : maxDepth,
        'min_samples_split' : minSamplesSplit,
        'min_samples_leaf' : minSamplesLeaf,
        'bootstrap' : bootstrap,
        'criterion' : criterion
    
    }

rfcBase = RandomForestClassifier()

from sklearn.model_selection import RandomizedSearchCV
rfcOnRandomSearch = RandomizedSearchCV( estimator = rfcBase , param_distributions = randomGrid , n_iter = 30 , cv = 5 , verbose = 2 , random_state = 27 , n_jobs = 4 )

rfcOnRandomSearch.fit( xTrain , yTrain )

rfcOnRandomSearch.best_params_

#evaluation of the random forest classifier on train data
rfcOnRandomSearch.score( xTrain , yTrain )

#evaluting the random forest classifier on test data
rfcOnRandomSearch.score( xTest , yTest )


