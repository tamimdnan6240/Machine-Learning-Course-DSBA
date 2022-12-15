#!/usr/bin/env python
# coding: utf-8

# In[160]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from os import listdir
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import metrics
from scipy.interpolate import interp1d
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, LinearRegression


# In[161]:


pd.set_option('display.max_columns', None)


# In[162]:


pd.set_option('display.max_row', None)


# In[163]:


# Change the direction here


# In[164]:


df_movies = pd.read_csv("IMDb_Top_1000_Movies_Dataset.csv")


# In[165]:


df_movies


# # Remove unecessary columns 

# In[166]:


df_movies = df_movies.drop(['Movie_Name','Movie_Certificate','Movie_Cast', 'Movie_Runtime', 'Movie_Poster','Movie_Description','Movie_Poster_HD','All_Movie_Info'], axis = 1)


# In[167]:


df_movies


# In[168]:


df_movies.dtypes


# # According to https://www.the-numbers.com/market/genres, most four popupar movies genres are Adventure, Action, Drama and Comedey because they have market share of 26.73%, 21.77%, 14.64% and 13.89%. So this study will conduct timer series on these 4 popupar genre to predict which genre will be more impactful after 100 years? 

# # Findout Adventure related genre

# In[192]:


targeted_genre = ['Adventure']
Adventure  = df_movies[df_movies['Movie_Genre'].map(lambda x: any(i in x for i in targeted_genre ))]
Adventure.shape


# In[193]:


Adventure.head()


# In[194]:


df = Adventure.groupby(by=['Movie_Year']).count()
df.head()


# In[195]:


df_all = df_movies.groupby(by=['Movie_Year']).count()


# In[196]:


int_df = pd.merge(df, df_all, how ='inner', on =['Movie_Year'])
int_df


# In[197]:


int_df['Percentage'] = (int_df['Movie_Genre_x']/int_df['Movie_Genre_y']) * 100
int_df


# # Seasonality Check without detrending

# In[198]:


def plot_df(df, y, title="", xlabel='Year', ylabel='Count of each genre Combination', dpi=100, use_index=True):
    plt.rc('font', size=60)
    plt.figure(figsize=(100, 100), dpi=dpi)
    plt.plot( y, color='blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
    


# In[199]:


plot_df(df, y=df['Movie_Genre'], title='Trend and Seasonality' )


# In[200]:


def plot_df_Percentage(df, y, title="", xlabel='Year', ylabel='Percentage of each genre Combination', dpi=100, use_index=True):
    plt.rc('font', size=60)
    plt.figure(figsize=(100, 100), dpi=dpi)
    plt.plot( y, color='blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


# In[201]:


plot_df(int_df, y=int_df['Percentage'], title='Trend and Seasonality' )


# In[202]:


# Auto Correlation
ppm = df. Movie_Genre
ppm.autocorr()


# In[203]:


ppm_p = int_df. Percentage
ppm_p.autocorr()


# In[204]:


ppm.autocorr(lag=26)
ppm_p.autocorr(lag=26)


# In[205]:


ppm.autocorr(lag=52)
ppm_p.autocorr(lag=52)


# In[206]:


plt.rc('font', size=10)


# In[207]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(ppm)


# In[208]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(ppm_p)


# In[209]:


from statsmodels.tsa.stattools import acf
autocorrelation = acf(ppm, nlags=201)
plt.plot(autocorrelation)
plt.ylabel("autocorrelaation")
plt.xlabel("lag")


# In[210]:


autocorrelation_p = acf(ppm_p, nlags=201)
plt.plot(autocorrelation_p)
plt.ylabel("autocorrelaation")
plt.xlabel("lag")


# # The dataset has not been detrended yet, so the seasonality is not right. We need to detrend the data then we can apply autocorrelation seasonality, trending to the dataset

# ## detrending

# In[211]:


ppm.diff().plot()
ppm_p.diff().plot()


# In[212]:


from statsmodels.tsa.stattools import acf
autocorrelation = acf(ppm.diff()[1:], nlags=105)
plt.plot(autocorrelation)
plt.ylabel("autocorrelation")
plt.xlabel("lag")


# In[213]:


autocorrelation_p = acf(ppm_p.diff()[1:], nlags=105)
plt.plot(autocorrelation_p)
plt.ylabel("autocorrelation")
plt.xlabel("lag")


# # Seasonal Model for Movie_Genre

# In[214]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ppm, model='additive', period= 20)
fig = decomposition.plot()
#fig.set_figheight(6)
fig.tight_layout()


# In[215]:


decomposition_p = seasonal_decompose(ppm_p, model='additive', period= 20)
fig = decomposition_p.plot()
#fig.set_figheight(6)
fig.tight_layout()


# # FB Prophet

# In[216]:


fb_ppm = pd.DataFrame(ppm).reset_index()
fb_ppm.columns = ['ds', 'y']
fb_ppm


# In[217]:


fb_ppm_p = pd.DataFrame(ppm_p).reset_index()
fb_ppm_p.columns = ['ds', 'y']
fb_ppm_p


# In[218]:


from prophet import Prophet
m = Prophet()
m.fit(fb_ppm[:500])


# In[219]:


m_p = Prophet()
m_p.fit(fb_ppm_p[:500])


# In[220]:


forecast = m.predict(fb_ppm)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[341]:


forecast_p = m_p.predict(fb_ppm_p)
forecast_p[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[342]:


fb_ppm.y.plot()
forecast.yhat.plot()


# In[343]:


fb_ppm.y.plot()
forecast.yhat.plot()


# In[345]:


m.plot(forecast)


# In[346]:


m_p.plot(forecast_p)


# In[339]:


future_dates = m.make_future_dataframe(periods = 100, freq = "Y", include_history = True)
prediction = m.predict(future_dates)
prediction.tail()


# In[340]:


future_dates = m_p.make_future_dataframe(periods = 100, freq = "Y", include_history = True)
prediction_p = m_p.predict(future_dates)
prediction_p.tail()


# # Findout Action related genre

# In[228]:


targeted_genre = ['Action']
Action  = df_movies[df_movies['Movie_Genre'].map(lambda x: any(i in x for i in targeted_genre ))]
Action.shape


# In[229]:


df = Action.groupby(by=['Movie_Year']).count()
df


# In[230]:


df_all = df_movies.groupby(by=['Movie_Year']).count()
int_df = pd.merge(df, df_all, how ='inner', on =['Movie_Year'])
int_df


# In[231]:


int_df['Percentage'] = (int_df['Movie_Genre_x']/int_df['Movie_Genre_y'])
int_df


# In[232]:


# Seasonality Check
def plot_df(df, y, title="", xlabel='Year', ylabel='Count of each genre Combination', dpi=100, use_index=True):
    plt.rc('font', size=60)
    plt.figure(figsize=(100, 100), dpi=dpi)
    plt.plot( y, color='blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


# In[233]:


plot_df(df, y=df['Movie_Genre'], title='Trend and Seasonality' )


# In[234]:


def plot_df_Percentage(df, y, title="", xlabel='Year', ylabel='Percentage of each genre Combination', dpi=100, use_index=True):
    plt.rc('font', size=60)
    plt.figure(figsize=(100, 100), dpi=dpi)
    plt.plot( y, color='blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


# In[235]:


plot_df(int_df, y=int_df['Percentage'], title='Trend and Seasonality' )


# In[236]:


# Auto Correlation
ppm = df. Movie_Genre
ppm.autocorr()


# In[237]:


ppm_p = int_df. Percentage
ppm_p.autocorr()


# In[238]:


ppm.autocorr(lag=26)
ppm_p.autocorr(lag=26)


# In[239]:


ppm.autocorr(lag=52)
ppm_p.autocorr(lag=52)


# In[240]:


plt.rc('font', size=10)


# In[241]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(ppm)


# In[242]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(ppm_p)


# In[243]:


from statsmodels.tsa.stattools import acf
autocorrelation = acf(ppm, nlags=201)
plt.plot(autocorrelation)
plt.ylabel("autocorrelaation")
plt.xlabel("lag")


# In[244]:


autocorrelation_p = acf(ppm_p, nlags=201)
plt.plot(autocorrelation_p)
plt.ylabel("autocorrelaation")
plt.xlabel("lag")


# In[245]:


# # detrending
ppm.diff().plot()
ppm_p.diff().plot()


# In[246]:


from statsmodels.tsa.stattools import acf
autocorrelation = acf(ppm.diff()[1:], nlags=105)
plt.plot(autocorrelation)
plt.ylabel("autocorrelation")
plt.xlabel("lag")


# In[247]:


autocorrelation_p = acf(ppm_p.diff()[1:], nlags=105)
plt.plot(autocorrelation_p)
plt.ylabel("autocorrelation")
plt.xlabel("lag")


# # Seasonal Model fopr Movie_Genre

# In[248]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ppm, model='additive', period= 20)
fig = decomposition.plot()
#fig.set_figheight(6)
fig.tight_layout()


# In[249]:


decomposition_p = seasonal_decompose(ppm_p, model='additive', period= 20)
fig = decomposition_p.plot()
#fig.set_figheight(6)
fig.tight_layout()


# In[250]:


# FB Prophet
fb_ppm = pd.DataFrame(ppm).reset_index()
fb_ppm.columns = ['ds', 'y']

fb_ppm


# In[251]:


fb_ppm_p = pd.DataFrame(ppm_p).reset_index()
fb_ppm_p.columns = ['ds', 'y']

fb_ppm_p


# In[252]:


from prophet import Prophet
m = Prophet()
m.fit(fb_ppm[:500])


# In[253]:


m_p = Prophet()
m_p.fit(fb_ppm_p[:500])


# In[254]:


forecast = m.predict(fb_ppm)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[255]:


forecast_p = m_p.predict(fb_ppm_p)
forecast_p[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[256]:


fb_ppm.y.plot()
forecast.yhat.plot()


# In[257]:


fb_ppm.y.plot()
forecast.yhat.plot()


# In[258]:


m.plot(forecast)


# In[259]:


m_p.plot(forecast_p)


# In[260]:


future_dates = m.make_future_dataframe(periods = 100, freq = "Y", include_history = True)
prediction = m.predict(future_dates)
prediction


# In[261]:


future_dates = m_p.make_future_dataframe(periods = 100, freq = "Y", include_history = True)
prediction_p = m_p.predict(future_dates)
prediction_p


# # Findout Drama related genre 

# In[262]:


targeted_genre = ['Drama']
Drama = df_movies[df_movies['Movie_Genre'].map(lambda x: any(i in x for i in targeted_genre ))]
Drama.shape


# In[263]:


Drama


# In[264]:


df = Drama.groupby(by=['Movie_Year']).count()


# In[265]:


df


# In[266]:


df_all = df_movies.groupby(by=['Movie_Year']).count()


# In[267]:


int_df = pd.merge(df, df_all, how ='inner', on =['Movie_Year'])


# In[268]:


int_df


# In[269]:


int_df['Percentage'] = (int_df['Movie_Genre_x']/int_df['Movie_Genre_y'])


# In[270]:


int_df


# # Seasonality Check

# In[271]:


def plot_df(df, y, title="", xlabel='Year', ylabel='Count of each genre Combination', dpi=100, use_index=True):
    plt.rc('font', size=60)
    plt.figure(figsize=(100, 100), dpi=dpi)
    plt.plot( y, color='blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
    

plot_df(df, y=df['Movie_Genre'], title='Trend and Seasonality' )


# In[272]:


def plot_df_Percentage(df, y, title="", xlabel='Year', ylabel='Percentage of each genre Combination', dpi=100, use_index=True):
    plt.rc('font', size=60)
    plt.figure(figsize=(100, 100), dpi=dpi)
    plt.plot( y, color='blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
    

plot_df(int_df, y=int_df['Percentage'], title='Trend and Seasonality' )


# # Auto Correlation

# In[273]:


ppm = df. Movie_Genre
ppm.autocorr()


# In[274]:


ppm_p = int_df. Percentage
ppm_p.autocorr()


# In[275]:


ppm.autocorr(lag=26)


# In[276]:


ppm_p.autocorr(lag=26)


# In[277]:


ppm.autocorr(lag=52)


# In[278]:


ppm_p.autocorr(lag=52)


# In[279]:


plt.rc('font', size=10)


# In[280]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(ppm)


# In[281]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(ppm_p)


# In[282]:


from statsmodels.tsa.stattools import acf
autocorrelation = acf(ppm, nlags=201)
plt.plot(autocorrelation)
plt.ylabel("autocorrelaation")
plt.xlabel("lag")


# In[283]:


autocorrelation_p = acf(ppm_p, nlags=201)
plt.plot(autocorrelation_p)
plt.ylabel("autocorrelaation")
plt.xlabel("lag")


# # # detrending

# In[284]:


ppm.diff().plot()


# In[285]:


ppm_p.diff().plot()


# In[286]:


from statsmodels.tsa.stattools import acf
autocorrelation = acf(ppm.diff()[1:], nlags=105)
plt.plot(autocorrelation)
plt.ylabel("autocorrelation")
plt.xlabel("lag")


# In[287]:


autocorrelation_p = acf(ppm_p.diff()[1:], nlags=105)
plt.plot(autocorrelation_p)
plt.ylabel("autocorrelation")
plt.xlabel("lag")


# # Seasonal Model for Movie_Genre  Column

# In[288]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ppm, model='additive', period= 20)
fig = decomposition.plot()
#fig.set_figheight(6)
fig.tight_layout()


# In[289]:


decomposition_p = seasonal_decompose(ppm_p, model='additive', period= 20)
fig = decomposition_p.plot()
#fig.set_figheight(6)
fig.tight_layout()


# # FB Prophet

# In[290]:


fb_ppm = pd.DataFrame(ppm).reset_index()
fb_ppm.columns = ['ds', 'y']


# In[291]:


fb_ppm


# In[292]:


fb_ppm_p = pd.DataFrame(ppm_p).reset_index()
fb_ppm_p.columns = ['ds', 'y']


# In[293]:


fb_ppm_p


# In[294]:


from prophet import Prophet
m = Prophet()
m.fit(fb_ppm[:500])


# In[295]:


m_p = Prophet()
m_p.fit(fb_ppm_p[:500])


# In[296]:


forecast = m.predict(fb_ppm)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[297]:


forecast_p = m_p.predict(fb_ppm_p)
forecast_p[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[298]:


fb_ppm.y.plot()
forecast.yhat.plot()


# In[299]:


fb_ppm_p.y.plot()
forecast_p.yhat.plot()


# In[300]:


m.plot(forecast)


# In[301]:


m_p.plot(forecast_p)


# In[302]:


future_dates = m.make_future_dataframe(periods = 100, freq = "Y", include_history = True)
prediction = m.predict(future_dates)
prediction


# In[303]:


future_dates = m_p.make_future_dataframe(periods = 100, freq = "Y", include_history = True)
prediction_p = m_p.predict(future_dates)
prediction_p


# # Findout Comedy related genre

# In[304]:


targeted_genre = ['Comedy']
Comedy = df_movies[df_movies['Movie_Genre'].map(lambda x: any(i in x for i in targeted_genre ))]
Comedy.shape


# In[305]:


df = Comedy.groupby(by=['Movie_Year']).count()


# In[306]:


df_all = df_movies.groupby(by=['Movie_Year']).count()
int_df = pd.merge(df, df_all, how ='inner', on =['Movie_Year'])
int_df


# In[307]:


int_df['Percentage'] = (int_df['Movie_Genre_x']/int_df['Movie_Genre_y'])
int_df


# # Seasonality Check

# In[308]:


def plot_df(df, y, title="", xlabel='Year', ylabel='Count of each genre Combination', dpi=100, use_index=True):
    plt.rc('font', size=60)
    plt.figure(figsize=(100, 100), dpi=dpi)
    plt.plot( y, color='blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


# In[309]:


plot_df(df, y=df['Movie_Genre'], title='Trend and Seasonality' )


# In[310]:


def plot_df_Percentage(df, y, title="", xlabel='Year', ylabel='Percentage of each genre Combination', dpi=100, use_index=True):
    plt.rc('font', size=60)
    plt.figure(figsize=(100, 100), dpi=dpi)
    plt.plot( y, color='blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


# In[311]:


plot_df(int_df, y=int_df['Percentage'], title='Trend and Seasonality' )


# In[312]:


# Auto Correlation
ppm = df. Movie_Genre
ppm.autocorr()


# In[313]:


ppm_p = int_df. Percentage
ppm_p.autocorr()


# In[314]:


ppm.autocorr(lag=26)


# In[315]:


ppm_p.autocorr(lag=26)


# In[316]:


ppm.autocorr(lag=52)


# In[317]:


ppm_p.autocorr(lag=52)


# In[318]:


plt.rc('font', size=10)


# In[319]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(ppm)


# In[320]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(ppm_p)


# In[321]:


from statsmodels.tsa.stattools import acf
autocorrelation = acf(ppm, nlags=201)
plt.plot(autocorrelation)
plt.ylabel("autocorrelaation")
plt.xlabel("lag")


# In[322]:


autocorrelation_p = acf(ppm_p, nlags=201)
plt.plot(autocorrelation_p)
plt.ylabel("autocorrelaation")
plt.xlabel("lag")


# In[323]:


# # detrending
ppm.diff().plot()
ppm_p.diff().plot()


# In[324]:


from statsmodels.tsa.stattools import acf
autocorrelation = acf(ppm.diff()[1:], nlags=105)
plt.plot(autocorrelation)
plt.ylabel("autocorrelation")
plt.xlabel("lag")


# In[325]:


autocorrelation_p = acf(ppm_p.diff()[1:], nlags=105)
plt.plot(autocorrelation_p)
plt.ylabel("autocorrelation")
plt.xlabel("lag")


# In[326]:


# Seasonal Model fopr Movie_Genre

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ppm, model='additive', period= 20)
fig = decomposition.plot()
#fig.set_figheight(6)
fig.tight_layout()


# In[327]:


decomposition_p = seasonal_decompose(ppm_p, model='additive', period= 20)
fig = decomposition_p.plot()
#fig.set_figheight(6)
fig.tight_layout()


# In[328]:


# FB Prophet
fb_ppm = pd.DataFrame(ppm).reset_index()
fb_ppm.columns = ['ds', 'y']

fb_ppm

fb_ppm_p = pd.DataFrame(ppm_p).reset_index()
fb_ppm_p.columns = ['ds', 'y']

fb_ppm_p


# In[329]:


from prophet import Prophet
m = Prophet()
m.fit(fb_ppm[:500])


# In[330]:


m_p = Prophet()
m_p.fit(fb_ppm_p[:500])


# In[331]:


forecast = m.predict(fb_ppm)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[332]:


forecast_p = m_p.predict(fb_ppm_p)
forecast_p[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[333]:


fb_ppm.y.plot()
forecast.yhat.plot()


# In[334]:


fb_ppm.y.plot()
forecast.yhat.plot()


# In[335]:


m.plot(forecast)


# In[336]:


m_p.plot(forecast_p)


# In[337]:


future_dates = m.make_future_dataframe(periods = 100, freq = "Y", include_history = True)
prediction = m.predict(future_dates)
prediction


# In[338]:


future_dates = m_p.make_future_dataframe(periods = 100, freq = "Y", include_history = True)
prediction_p = m_p.predict(future_dates)
prediction_p


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




