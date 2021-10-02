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


### 2.0 -Tabular analysis on likes prediction.ipynb

Tabular analysis to predict number of likes on the posted articles with different models. Based on mean squared error, th random rorest model performed best.
The predictors were:  clicksCount: number of clicks on the article, originalFlag: whether the article was originally created (i.e., not shared) by the account, orderNum: wechat allows influencers to upload several articles as a group each day. Influencers decide the order of the articles in the group. Normally, the firs article receives most views.

<img width="298" alt="image" src="https://user-images.githubusercontent.com/10263993/135197051-202a0b10-4c0e-44a8-9fe8-69f1c74ade35.png">


### 3.0 -Time series analysis on likes prediction.ipynb

### 4.0 - RNN and LSTM models on likes prediction.ipynb 

### 4.0 - LSTM models on ad prediction.ipynb 
To judge if an anticle is an ad, I outsourced the manual labelling of ad to an agency in China. I labelled about a hundred articles as examples and asked the agency to label 10,000 others. In the data, the label A means the article contains no ad, B means the whole article is ad, and C means the main body of the article is not ad, but it includes ad near the end of the article.

<img width="94" alt="image" src="https://user-images.githubusercontent.com/10263993/135727892-bb55584f-28a6-4a00-8146-19753827565b.png">


Using only features from the text of the articles, I adopted vanilla LSTM, bidirectional LSTM, and stacked LSTM to predict the ad label of the articles. The results below show relatively good prediction accuracy (> 80%) on the testing set. Th bidirectional LSTM model performed best and used the longest time to converge. This model showed signs of overfitting after the 2nd epoch.     

<img width="427" alt="image" src="https://user-images.githubusercontent.com/10263993/135727812-869d217a-5cd1-4d4b-9c85-4c845351c90c.png">



### 5.0 - BERT models.ipynb


