# NLP analysis on WeChat data

## Context

## The Dataset

## Evaluation Metrics

## Notebooks

### 1.0 -Topic Modeling and Explorotary data analysis.ipynb

To get a general understanding of the content posted by all the influencer accounts, I conducte topic modeling using Latent Dirichlet Allocation(LDA). Given the large computing resource required, the LDA code is ran on the GPU server of my school.  

##### Examples of the word loading within each topics derived from LDA topic modeling (manually translated to English for illustration purpose)
<img width="785" alt="image" src="https://user-images.githubusercontent.com/10263993/134822953-fe38dcfc-64ce-4f58-86b6-5fb1bc8d1eec.png">


##### Plot the correlation between each topic with likeCount across time, the results show that the popularity of different topics differ signficantly over time. 
<img width="394" alt="image" src="https://user-images.githubusercontent.com/10263993/134822653-d2333d0f-3f29-4c14-8906-b5d016bd7961.png">


### 2.0 -Tabular analysis.ipynb

Tabular analysis to predict number of likes on the posted articles with different models. Based on mean squared error, th random rorest model performed best.
The predictors were:  clicksCount: number of clicks on the article, originalFlag: whether the article was originally created (i.e., not shared) by the account, orderNum: wechat allows influencers to upload several articles as a group each day. Influencers decide the order of the articles in the group. Normally, the firs article receives most views.

<img width="298" alt="image" src="https://user-images.githubusercontent.com/10263993/135197051-202a0b10-4c0e-44a8-9fe8-69f1c74ade35.png">


### 3.0 -Time series analysis.ipynb


### 4.0 - RNN and LSTM models.ipynb 

### 5.0 - BERT models.ipynb


