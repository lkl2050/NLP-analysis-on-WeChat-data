# NLP analysis on WeChat data

## Context

Similar to Facebook, WeChat has an information flow section which allows users to see their friends posts and updates. It is not surpring that plenty of influencer accounts are active on WeChat's information flow, trying to spred news, provide information, promote products and services etc.  

## The Dataset

I scrapped 5,517,352 articles from influencer accounts on WeChat. Depending on tasks, I put different sizes of subsets of the dataset into analysis. An sample dataset is shown below. 

<img width="1270" alt="image" src="https://user-images.githubusercontent.com/10263993/136125214-17f491b8-1fdc-43a3-9db8-0ea418d92b8b.png">

I used number of likes on the articles and ad labels as the dependent variables for the analysis. There are several existing features on each article that can be used as predictors:  \s\s
clicksCount: number of clicks on the article\s\s
originalFlag: whether the article was originally created (i.e., not shared) by the account\s\s
orderNum: wechat allows influencers to upload several articles as a group each day. Influencers decide the order of the articles in the group. Normally, the firs article receives most views.

## Evaluation Metrics
For predicting likes count, I used mean squared error (MSE) and mean absolute percentage error (MAPE) as the evaluation metrics of the regression task. 
For ad classification task, I mained used prediction accuracy as the evaluation metric. Since I am interested in both false positive and negative cases and the d label is relatively balanced, accuracy is a good metric to describe the prediction results.   

## Notebooks

### 1.0 - Explorotary data analysis.ipynb

To get a general understanding of the content posted by all the influencer accounts, I conducte topic modeling using Latent Dirichlet Allocation(LDA) (see appendix 1). Given the large computing resource required, the LDA code is ran on the GPU server of my school.  

##### Examples of the word loading within each topics derived from LDA topic modeling (manually translated to English for illustration purpose)
<img width="785" alt="image" src="https://user-images.githubusercontent.com/10263993/134822953-fe38dcfc-64ce-4f58-86b6-5fb1bc8d1eec.png">


##### Plot the correlation between each topic with likeCount across time, the results show that the popularity of different topics differ signficantly over time. 
<img width="394" alt="image" src="https://user-images.githubusercontent.com/10263993/134822653-d2333d0f-3f29-4c14-8906-b5d016bd7961.png">


### 2.0 -Tabular analysis on likes prediction.ipynb

Tabular analysis to predict number of likes on the posted articles with different models. Based on mean squared error, the random rorest model performed best.


<img width="298" alt="image" src="https://user-images.githubusercontent.com/10263993/135197051-202a0b10-4c0e-44a8-9fe8-69f1c74ade35.png">


### 3.0 -Time series analysis on likes prediction.ipynb
#### Analyze the time series with day as the time unit. This gives 1920 days of likes count as the time series dataset. 

#### Naive model forecasting
I first adopted an naive approach with assumes that the next expected point is equal to the last observed point. 
Naive forecast MSE: 40500.711534
Naive forecast MAPE: 3891825798079374.500000
<img width="989" alt="image" src="https://user-images.githubusercontent.com/10263993/135771601-98f67cf9-e688-4ed8-a7ed-24d203cebb89.png">

#### Simple average forecasting
This simple average  method forecasts the expected value equal to the average of all previously observed points.
Simple average MSE: 13333.904245
Simple average MAPE: 2825187969291605.500000
<img width="990" alt="image" src="https://user-images.githubusercontent.com/10263993/135772067-26e78f76-2025-43a8-819c-56f9390f463a.png">

#### ARIMA models

##### plot of the time series and checking if it's stationay
<img width="992" alt="image" src="https://user-images.githubusercontent.com/10263993/135772084-9148ab1b-8ab0-45e3-8983-cca8438eaaaf.png">

##### plot after decomposing the time series
<img width="710" alt="image" src="https://user-images.githubusercontent.com/10263993/135772100-2ceda14c-8c48-476a-883b-4e63531525e1.png">
<img width="976" alt="image" src="https://user-images.githubusercontent.com/10263993/135772128-c25612d8-efe4-443f-bd5b-53fdb7637027.png">


#####  Since the original data is not stationary, it needs differencing. The result below suggests differencing once is enough to achieve stationary (d = 1 in the ARIMA model)
<img width="511" alt="image" src="https://user-images.githubusercontent.com/10263993/135772136-4623d6e2-bc8f-4723-892d-483572893596.png">


#####  The ACF and PACF plot suggest p = 13, q = 1
<img width="507" alt="image" src="https://user-images.githubusercontent.com/10263993/135772162-2d68e5a3-aedf-41c0-9695-2f4726d056cf.png">
<img width="987" alt="image" src="https://user-images.githubusercontent.com/10263993/135772187-e1d2503a-04f3-4955-a8c7-5cb124df3cec.png">

#### Plot the predicted likes count vs. the observed likes count
ARIMA MSE: 6630.75
ARIMA MAPE: 101.35
<img width="1004" alt="image" src="https://user-images.githubusercontent.com/10263993/135772368-eae57a36-d9d8-4c23-b7ce-a64cd6536b11.png">

#### Given the MSE and MAPE results, the ARIMA model clearly performed better than the naive model and the simple average model. 

### Analyze the time series with month as the time unit
Because there is not much trend or seasonality that observed on the day level, I tried to analyze the data again on the month level. This gives 64 rows of data in the time series. 

#### Again, the original time series is not stationay (figure on top) and it became stationay after differencing (figure at bottom) 
<img width="1029" alt="image" src="https://user-images.githubusercontent.com/10263993/135772398-dc96b076-ebea-462a-8a34-79f03798b8c4.png">

<img width="1000" alt="image" src="https://user-images.githubusercontent.com/10263993/135772483-143b0c5f-75d2-4116-9876-d62d9f39868b.png">

#### Similar to above, ARIMA model was adopted to do forecast
ARIMA MSE: 111725.21
ARIMA MAPE: 94.77
<img width="410" alt="image" src="https://user-images.githubusercontent.com/10263993/135920240-670d22cc-e64c-4305-ae89-9abd2a574120.png">

#### Given the MAPE results, forecasting future months' likes is actually easier than forecasting future day's likes, possibly because more time-series information can be captured on the monthly data. 


### 4.0 - LSTM models on ad classification and likes prediction.ipynb 
To judge if an anticle is an ad, I outsourced the manual labelling of ad to an agency in China. I labelled about a hundred articles as examples and asked the agency to label 10,000 others. In the data, the label A means the article contains no ad, B means the whole article is ad, and C means the main body of the article is not ad, but it includes ad near the end of the article.

<img width="94" alt="image" src="https://user-images.githubusercontent.com/10263993/135727892-bb55584f-28a6-4a00-8146-19753827565b.png">

I used the tokenized words as the predictors in the models. Using only features extracted from the text of the articles, I adopted vanilla LSTM, bidirectional LSTM, and stacked LSTM to predict the ad label of the articles. The results below show relatively good prediction accuracy (> 80%) on the testing set. Th bidirectional LSTM model performed best and used the longest time to converge. This model showed signs of overfitting after the 2nd epoch.     

<img width="400" alt="image" src="https://user-images.githubusercontent.com/10263993/135920973-044075b5-c325-4e2f-ab8f-93e3d9125c91.png">

As a comparison, I trained a SVM classification model with the non-text predictors. The SVM model show an accuracy of 0.669835, which is clearly lower than the accuracy achieved by the LSTM models based on text features. 

Similar to the abov analysis, I adopted the three LSTM models to predict likes count with only features extracted from the text. The MSE results below shows that th bidrectional LSTM model performed best, and the other two models almost performed the same. 

<img width="508" alt="image" src="https://user-images.githubusercontent.com/10263993/135936259-adace7fe-21c8-4320-9329-e75f32c7b34e.png">

As a comparison, I trained a linear regression model with the non-text features. The model achieved an validation MSE of 1010437.907273, which is much higher than the MSE from the bidirectional LSTM: 828027.25

### 5.0 - Transfer learning with BERT models on likes prediction.ipynb
The above analysis with LSTM used only tokenized words as the input, they did no capture the semantic information among the words. This analysis adopted the pretrained BERT model with some fine tunining at the output layer, trying to achieve a better prediction result.

I adopted the pretrained model of bert-base-chinese from the transformers package. After data preprocessing to input the text data into the bert model, I constructed the model as below. I froze the bert layers so I don't need train them on my side. Instead, I used the semantic information from these pretrained layers to extract more insight from my own text data.     

<img width="876" alt="image" src="https://user-images.githubusercontent.com/10263993/136122483-6ed094ab-e350-43b6-be1c-af8dcbccefef.png">

Given limited computing resources, I only ran three epochs and the results are shown below. Given the decreasing loss over the epochs, the model is likely to be  underfit with three epochs. However, the results already showed a clear improvement over the linear regression and LSTM models in the previous section, because the validation MSE was 810718.6875, which is more than 2% lower than the results from the bidirectional LSTM models. Provided more computing resources for the current BERT model, it is expected to outperform other models more.

<img width="1002" alt="image" src="https://user-images.githubusercontent.com/10263993/136123715-b4ff96cd-73aa-4fd6-8a95-97610db1fb34.png">




