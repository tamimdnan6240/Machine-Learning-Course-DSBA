# Machine-Learning-Course-DSBA
This machine learning course an official uncc course. I did some spectacular tasks under the professor Dr. Depeng Xu.


All Lectures are in Youtube: https://www.youtube.com/playlist?list=PL_pVmAaAnxIRnSw6wiCpSvshFyCREZmlM
Dimension reduction
https://amueller.github.io/COMS4995-s20/slides/aml-13-dimensionality-reduction/#1
Clustering
https://amueller.github.io/COMS4995-s20/slides/aml-14-clustering-mixture-models/#1

Recommenders
https://uncc.instructure.com/courses/174941/files/19158266?wrap=1

Graph embedding
https://uncc.instructure.com/courses/174941/files/19158265?wrap=1

Time Series
https://amueller.github.io/COMS4995-s20/slides/aml-21-time-series/#1

Text
https://amueller.github.io/COMS4995-s20/slides/aml-15-text-data/#1

https://amueller.github.io/COMS4995-s20/slides/aml-16-topic-models/#1

https://amueller.github.io/COMS4995-s20/slides/aml-21-time-series/#48

Neural Networks
https://amueller.github.io/COMS4995-s20/slides/aml-18-neural-networks/#1

Supervised learning
slides: https://amueller.github.io/COMS4995-s20/slides/aml-03-supervised-learning

Preprocessing
slides: https://amueller.github.io/COMS4995-s20/slides/aml-04-preprocessing

Linear Regression
https://amueller.github.io/COMS4995-s20/slides/aml-05-linear-models-regression#

Linear Classification
https://amueller.github.io/COMS4995-s20/slides/aml-06-linear-models-classification#

Trees 
https://amueller.github.io/COMS4995-s20/slides/aml-07-trees-forests

Gradient boosting
https://amueller.github.io/COMS4995-s20/slides/aml-08-gradient-boosting#

Stacking
Model evaluation 
https://amueller.github.io/COMS4995-s20/slides/aml-09-model-evaluation

## Basic Codes from the Instructor: Depeng Xu ( Zip folder: Practiced and Solutions) 

file:///C:/Users/tamim/Downloads/text1.html
file:///C:/Users/tamim/Downloads/text2.html
file:///C:/Users/tamim/Downloads/text3.html
file:///C:/Users/tamim/Downloads/nn1.html
file:///C:/Users/tamim/Downloads/nn2.html

## Semester Project : 

Title: Predicting the Most Dominating Movie Genre for the Next Few Years

Group Members – Tamim Adnan(tadnan@uncc.edu), Yanran Sun (ysun35@uncc.edu)

Source Code: Machine_Learning_Group_Project_Both_Count_Percentage.py or file:///C:/Users/tamim/Downloads/Machine_Learning_Group_Project_Both_Count_Percentage.html

Research Topic: The movie industry has a huge impact on our economics and society. The U.S. movie industry is worth $95.45 billion as of 2022. The film industry employs roughly 2.1 million people in the U.S. and provides a substantial contribution to the U.S. economy pre-pandemic. Surprisingly, the movie industry continued to grow slowly throughout the pandemic. Watching movies is an excellent source of entertainment nowadays. Recently streaming services, such as Netflix, HBOMAX, Hulu, Disney Plus, etc., are prevalent platforms with a substantial number of subscribers. However, these movies are from different genres. In recent years, biographies, crime thrillers, horror, or drama genre movies are more popular than some other categories, such as romance. Gradually, audience interests are changing along with time. This change also influences the film's production professionals. Hence, this project aims to apply time series analysis to the movie datasets to understand which movie genres will be the most dominating category for the next few decades. 

Source of Data: We will use the IMDb Top 1000 Movies Dataset from Kaggle for necessary data analysis and time series modeling. The dataset in the form of an Excel CSV file can be found on this webpage: https://www.kaggle.com/datasets/hrishabhtiwari/imdb-top-1000-movies-dataset. The dataset contains 1000 top-rated movies with data scraped directly from IMDb. The dataset has the following information about a movie: name, rating, certificate, year of production, runtime, cast, genre, poster, description, and high-definition poster. The year of production ranges from 1920 to 2022. The rating is above 7.6, which means at least 76% of the audience loves the movie. For our project, we are interested in the genre and the year of production because we would like to apply time series analysis methods to predict the popular genres in the next decades. 

Problem formulation: There are altogether 21 individual genres in the dataset, such as “drama”, “action”, “family” and so on. Sometimes a movie could belong to different genres at the same time. For example, Schindler's List produced in 1993 belongs to “Biography”, “Drama”, and “History”. In this case, we count it in all three categories, namely biography-related movies, drama-related movies, and history-related movies. For a movie that belongs to a single genre, we count it directly into that category. For example, The Shawshank Redemption produced in 1994 belongs to the genre of “drama”, so we count it in the category of drama-related movies. After this data preprocessing, we will have the counts of all categories from 1920 to 2022 and we have altogether 21 categories. We could calculate the percentage of each category from 1920 to 2022. Based on the counts and the percentage, we could apply time series methods to predict the counts and the percentage of each category in the next few decades. 

The Implemented Methodology: The methodology for this study is based on data processing particularly. Due to largest size of dataset and repetition of a category in many combinations of genre, the dataset processing is a critical part of this study. Hence, this study followed a very short literature review to find which genres are effective nowadays. So, steps of methodology have been discussed in the following:  
1.	Literature review to find the popular genres: As movies in this dataset belong to multiple categories at the same time, therefore, we conducted a literature review to understand which genres are most popular in recent times. According to https://www.the-numbers.com/market/genres, the four most popular movie genres are Adventure, Action, Drama and Comedy because they have a market share of 26.73%, 21.77%, 14.64% and 13.89%. So, this study will conduct a timer series with Prophet model on these four popular genres to predict which genre will be more impactful after 100 years.
2.	Data Preprocessing
a.	Counting the specific genre-related movies every year by applying group by count: we filtered out to find only those genres-related movies. With group count, the data showed a trend in the number of genre movies yearly. 
b.	Percentage of each genre movie in a year among all other released movie genres in that year: This study prepared the dataset by creating a percentage of specific genre-type film from all other genre-type movies released in that year for making validation on the prediction. So, the following steps are the significant steps to conduct this project.
c.	Autocorrelation: In Autocorrelation, number lags are essential, which means that number of original data is parallel to the lag number. A set of positive lags follow a set of negative lags, and a positive lag amongst the positive's lags can measure the trend and seasonality of a dataset.  
d.	Data Preparation for Prophet Model: Prophet Model has a specific format of data frame so that it can process prediction on the targeted column with respect to time, yearly in this project. 

Result Analysis: Analysis of this project is mainly based on seasonality and trending, Stationarity and Non-Stationarity, and prediction of the prophet model on the dataset.  
1.	Seasonality and Trend Check: 
The Adventure genre-related movies have been detrended and autocorrelation were applied to check the trend and seasonality. The Figure 1 shows that both count and percentage base datasets have trends and seasonality. 

         
    Figure 1. Seasonality, trend, and Resid for Adventure (Count Based and Percentage Based)

Moreover, the seasonality and trend for drama have been depicted in Figure 2 for both count and percentage-based dataset. While the number of movie is likely to increase, the percentage has a fluctuation. 
         
    Figure 2. Seasonality, trend, and resid for Drama (Count Based and Percentage Based)

2.	Results from Prophet Model: It seems that Adventure genre-related movies will be more than 5 in 2026, according to Figure 1, which is nearly 20% of all genres-related movies in 2026. Likewise, the number of action movies will be 6 in 2026, which is almost 23% of genre-related movies in that year. On the other hand, the number of drama-related movies will be approximately 17 in 2026, which is 70% of all movie genres in 2026. And finally, the number of comedy-related movies in 2026 will be 4, which is close to 16% of all other genres in the year. For all four datasets, the number of movies will be increased, but percentages will be lessened. 

    
Figure 3. Prophet Forecasting based on Count and percentages for adventure-related genre
 
   
Figure 4. Prophet Forecasting based on Count and percentages for drama-related genre 

Conclusion: 
The dataset of this study is critical to set for specific genres of movies since most cases, one movie belongs to multiple categories. Therefore, conducting filtering operations to the movie genres can find out the count of a specific one. Therefore, this study counted adventure, action, drama, and comedy-related genres each year. Moreover, this study conducted an operation to find the percentages of each genre among all other genres in a year. Afterward, detrending and autocorrelation were applied to each newly dataset based on categories to check trend and seasonality. Then the datasets were analyzed with the prophet model for each dataset up to 2026. The prediction result showed that the number of movies for adventure, action, drama, and comedy would increase, but their percentage will lessen. 

Teamwork Statement: 
Table 1. Peer Evaluation
List of Tasks	Contributors
Data Processing for count based 	Tamim
Data Processing for percentage based 	Yanran
Time series for four genres	Yanran & Tamim Shared
Presentation Preparation	Yanran
Presentation Review and Modification	Tamim 
Report Writing: 
Research Topic, Source of Data Description and Problem formulation	Yanran
Report Writing: 
The Implemented Methodology, Result Analysis and Conclusion	Tamim 
Edit, review, and approve final report	Yanran & Tamim

Reference: 
1.	https://www.kaggle.com/code/prashant111/complete-guide-on-time-series-analysis-in-python
2.	https://www.ibm.com/docs/en/planning-analytics/2.0.0?topic=SSD29G_2.0.0/com.ibm.swg.ba.cognos.tm1_prism_gs.2.0.0.doc/papr_forecast_seasonality.htm
3.	https://medium.com/analytics-vidhya/predicting-stock-prices-using-facebooks-prophet-model b1716c733ea6#:~:text=Prophet%20is%20a%20procedure%20for,several%20seasons%20of%20historical%20data. 


