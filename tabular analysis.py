# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 11:25:44 2020


#This file mainly explores how topic weights influence number of likes using tabular dataset

"""

import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import datetime

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from statsmodels.formula.api  import ols
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#rawdata = pd.read_csv (r'/Users/cairo/Google Drive/wechat data/016.csv', encoding='utf-8')
#topicdata = pd.read_csv(r'/Users/Junhao/Google Drive/wechat data/TopicOutcomeAll30Topic.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')

topicdata = pd.read_csv(r'/Users/cairo/Google Drive/wechat data/TopicOutcomeAll20Topic.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')


#Q1: Which topics can predict the best influencer performance (indicated by likes and clicks of posts) in different lifespan of the account; what is the optimal level of topic diversity?

#Q4: To what extent should an account follow the hot topics in news? What is the optimal balance(i.e., topic proportion)  of following trending topics vs. sticking to account expertise? For instance, should an account on fitness share articles about the US election? (This might need extra data about hot news topic in Chinese media)


##################################
#linear regression and XGboost
##################################


###############################################################################################
#prepare data
topicdata.head()


y = topicdata.likeCount

#X = topicdata[topicdata.columns[-30:]]
X0 = topicdata.iloc[:,-20:]
X1 = topicdata[["clicksCount", "orderNum", "originalFlag"]]

X = pd.concat([X0, X1], axis=1)
X = sm.add_constant(X) #add a intercept term to the regression
#X.reset_index(drop=True)

#X.index = pd.RangeIndex(len(X.index))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


###############################################################################################
#linear regression with all variables
results = sm.OLS(y.astype(float), X.astype(float)).fit()
results.summary()

#linear regression with only the topic variables
results2 = sm.OLS(y.astype(float), X0.astype(float)).fit()
results2.summary()

###############################################################################################
#xgboost regression
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(X_train.astype(float),y_train.astype(float))

preds = xg_reg.predict(X_test.astype(float))

rmse = np.sqrt(mean_squared_error(y_test.astype(float), preds))
print("RMSE: %f" % (rmse))

mean_squared_error(y_test, preds)


###############################################################################################
#######################
#regression in neural networks with keras
######################
input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(64, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='relu'))

model.compile(loss='mean_squared_error', 
              optimizer='adam',
              metrics=['mean_absolute_percentage_error']
              
              )
model.summary()

history = model.fit(X_train , y_train,
                    epochs=5,
                    verbose=True,
                    #steps_per_epoch=5,
                    
                    validation_data=(X_test, y_test),
                    batch_size=300
                    )

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

##########################












