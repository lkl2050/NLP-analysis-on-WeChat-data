from gensim import corpora, models

import jieba.posseg as jp, jieba
from jieba import analyse
import pandas as pd
import numpy as np
import time

jieba.enable_parallel()

numTopic = 17
titleWeight = 10
summaryWeight = 1

stopwords = [line.strip() for line in open(r'/global/project/rotman_research/pc_kliu/WechatCode/LDA/extrastopword.txt', encoding='utf-8').readlines()]
swset = set(stopwords)


data1 = pd.read_pickle(r'/global/project/rotman_research/pc_kliu/WechatCode/LDA/cut021.pkl')
words_ls = data1['wordsCut'].values.tolist()
data1 = data1.drop(['wordsCut'], axis=1)

print("read cut021")


i = 20
while i>9:
    datatemp = pd.read_pickle(r'/global/project/rotman_research/pc_zgong/WechatCode/LDA/cut0%d'%i + '.pkl')
    words_lstemp = datatemp['wordsCut'].values.tolist()
    datatemp = datatemp.drop(['wordsCut'], axis = 1)
    words_ls.extend(words_lstemp)
    words_lstemp = []
    data1 = data1.append(datatemp, ignore_index = True)
    print("read cut0%d"%i)
    i = i-1


i = 9
while i>0:
    datatemp = pd.read_pickle(r'/global/project/rotman_research/pc_zgong/WechatCode/LDA/cut00%d'%i + '.pkl')
    words_lstemp = datatemp['wordsCut'].values.tolist()
    datatemp = datatemp.drop(['wordsCut'], axis = 1)
    words_ls.extend(words_lstemp)
    words_lstemp = []
    data1 = data1.append(datatemp, ignore_index = True)
    print("read cut00%d"%i)
    i = i-1

print("cut more stopwords")
NN = len(words_ls)
for i in range(0,NN):
    words_ls[i] = [x for x in words_ls[i] if x not in swset]

print("cread word list done")
# Construct Dictionary
dictionary = corpora.Dictionary(words_ls)
# Based on Dictionary, generate the vector of words and the matrix of words frequency
corpus = [dictionary.doc2bow(words) for words in words_ls]
print("start lda!")
# LDA Model: use num_topics to set the number of topics
#lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=numTopic)
lda = models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=numTopic, workers = 12)

# Print all topics: for each topic print 10 top words
#for topic in lda.print_topics(num_words=10):
#    print(topic)




# topic inference
#print(lda.inference(corpus))

topicsDF = pd.DataFrame(lda.inference(corpus)[0])
for i in range(0,numTopic):
    topicsDF.rename(columns={i:'topic%d'%i}, inplace=True)

print("topics transferred to dataframe")
    
    
datawithtopics = pd.concat([data1, topicsDF], axis=1, sort=False)
print("data merged")

datawithtopics.to_csv('/global/project/rotman_research/pc_zgong/WechatCode/LDA/TopicOutcomeAll17Topic.csv', sep=',', encoding='utf-8', index=False)


topickeywords = lda.print_topics(num_words=30)
dftopic = pd.DataFrame(data={"topics": topickeywords})
dftopic.to_csv('/global/project/rotman_research/pc_zgong/WechatCode/LDA/17TopicsWords.csv', sep=',', encoding='utf-8', index=False)

for i in range(0, lda.num_topics):
    print(lda.print_topic(i))

print("17 topics done")


print("Starting 18 topics.")
numTopic = 18

lda = models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=numTopic, workers = 12)

topicsDF = pd.DataFrame(lda.inference(corpus)[0])
for i in range(0,numTopic):
    topicsDF.rename(columns={i:'topic%d'%i}, inplace=True)

print("topics transferred to dataframe")

datawithtopics = pd.concat([data1, topicsDF], axis=1, sort=False)
print("data merged")

datawithtopics.to_csv('/global/project/rotman_research/pc_zgong/WechatCode/LDA/TopicOutcomeAll18Topic.csv', sep=',', encoding='utf-8', index=False)

topickeywords = lda.print_topics(num_words=30)
dftopic = pd.DataFrame(data={"topics": topickeywords})
dftopic.to_csv('/global/project/rotman_research/pc_zgong/WechatCode/LDA/18TopicsWords.csv', sep=',', encoding='utf-8', index=False)

for i in range(0, lda.num_topics):
    print(lda.print_topic(i))
print("18 topics done")
