#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 23:21:15 2020

@author: cairo
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


#rawdata = pd.read_csv (r'/Users/cairo/Google Drive/wechat data/016.csv', encoding='utf-8')
topicdata = pd.read_csv(r'/Users/cairo/Google Drive/wechat data/TopicOutcomeAll20Topic.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')


DF_list= list()
for i in topicdata.account.unique():
   dff = topicdata[topicdata["account"].isin([i])]
   dff["std"] = dff.filter(like='topic', axis=1).astype(float).std(axis = 1)
   DF_list.append(dff)
   

qq4 = []
for j in DF_list:
    qq3 = []
    for i in j["publicTime"]:
        qq = datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S')
        qq2 = [qq.year, qq.month, qq.day]
        qq3.append(qq2)
    qq4.append(qq3)

DF_list2 = []
for index, item in enumerate(DF_list):
    df2 = pd.DataFrame(qq4[index]) 
    df2a = df2.rename(columns={0: "year", 1: "month", 2:"day"})
    df2a.index = item.index
    df4 = pd.concat([item, df2a], axis=1)
    DF_list2.append(df4)


DF_list2[0]

DF_list2[0].year
DF_list2[0].month

DF_list2[0].topic0
DF_list2[1].topic0
DF_list2[2].topic0
type(DF_list2[0].topic0)


pd.to_numeric(DF_list2[0].topic0)

DF_list2[0].dtypes


DF_list2[0]['topic5'].astype(float)
DF_list2[0]['topic6'].astype(float)

type(DF_list2[0]['topic6'][0])


DF_list2[0].groupby(['year','month']).agg(clicksCount=('topic6', 'mean')).reset_index(drop=False)


###########################
'''
monthstd_list=[]
for j in DF_list2:
    j.clicksCount = pd.to_numeric(j.clicksCount)
    j.likeCount = pd.to_numeric(j.likeCount)
    jj = j.groupby(['year','month']).agg(std=('std', 'mean'),clicksCount=('clicksCount', 'mean'), likeCount=('likeCount', 'mean')).reset_index(drop=False)
    monthstd_list.append(jj)

'''

monthstd_list=[]
for j in DF_list2:
    j.topic1 = j.topic1.astype(float)
    j.topic2 = j.topic2.astype(float)
    j.topic3 = j.topic3.astype(float)
    j.topic4 = j.topic4.astype(float)
    j.topic5 = j.topic5.astype(float)
    j.topic9 = j.topic9.astype(float)
    j.topic14 = j.topic14.astype(float)
    j.topic15 = j.topic15.astype(float)
    
    jj = j.groupby(['year','month']).agg(topic1=('topic1', 'mean'),topic2=('topic2', 'mean'), topic3=('topic3', 'mean')
                                         , topic4=('topic4', 'mean'), topic5=('topic5', 'mean'), topic9=('topic9', 'mean')
                                         , topic14=('topic14', 'mean'), topic15=('topic15', 'mean'), 
                                         clicksCount=('clicksCount', 'mean'), likeCount=('likeCount', 'mean')).reset_index(drop=False)
    monthstd_list.append(jj)
    


flat_monthstd = pd.concat(monthstd_list)



#flat_monthstd.groupby(flat_monthstd.index).to_frame()

grouped_df = flat_monthstd.groupby(flat_monthstd.index)


grouped_df.head()



grouped_df.unstack().iloc[:,1]

corrall2 = grouped_df.to_frame()


corrall2.plot(y='corr')


corrall = flat_monthstd.groupby(flat_monthstd.index)[['topic1','likeCount']].corr().unstack().iloc[:,1]

corrall2 = corrall.to_frame()
corrall2.columns
corrall2.columns = ['corr'] 
corrall2
type(corrall2)

corrall2.plot(y='corr')

################################################
################################################
################################################


corrall = flat_monthstd.groupby(flat_monthstd.index)[['topic9','likeCount']].corr().unstack().iloc[:,1]
corrall2 = corrall.to_frame()
corrall2.columns = ['corr'] 
corrall2.plot(y='corr')


corrall = flat_monthstd.groupby(flat_monthstd.index)[['topic14','likeCount']].corr().unstack().iloc[:,1]
corrall2 = corrall.to_frame()
corrall2.columns = ['corr'] 
corrall2.plot(y='corr')


corrall = flat_monthstd.groupby(flat_monthstd.index)[['topic14','likeCount']].corr().unstack().iloc[:,1]
corrall2 = corrall.to_frame()
corrall2.columns = ['corr'] 
corrall2.plot(y='corr')




