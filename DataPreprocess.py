import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import math


df = pd.read_csv("csvfile/source/data_sno1.0.csv")
df = df.rename(columns={"Unnamed: 0": "time"})

emptyidx=[]
for x in df.index:
    if df.time[x] is np.nan:
        emptyidx = emptyidx + [x]

df = df.drop(index = emptyidx)
df = df.drop(columns = ['act','lat','lng','tot',])

df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H%M%S', errors='ignore')
df = df.set_index(pd.DatetimeIndex(df['time']))
df = df.drop_duplicates()
float_idx = ['sno','HUMD','bemp','sbi','PRES', 'TEMP', 'WDIR', 'H_24R', 'WDSE', 'WSGust', 'SeaPres','GloblRad', 'CloudA', 'PrecpHour', 'UVI', 'Visb', 'WDGust', 'td']
df[float_idx] = df[float_idx].astype('float')

df['bemp'] = df['bemp'].apply(lambda x: round(x, 0))
df['sbi'] = df['sbi'].apply(lambda x: round(x, 0))

fill_past_mean_tag = ['bemp','sbi']
interpolate_tag = ['TEMP','WDIR','CloudA','H_24R','sno','PRES','HUMD','WDSE']
fillzero_tag = ['GloblRad','UVI']

for tag in fill_past_mean_tag:
    dfl = []
    ndf = df[tag]
    for month in range(0,11):
        x = month%3
        y = math.floor(month/3)
        data = ndf[ndf.index.month == (month+2)]
        idx = data.index[data.apply(np.isnan)]

        #get mean of each weekday
        meanss = []
        for wkday in range(0,7):
              for hr in range(0,24):
                means = round(data[(data.index.hour == hr)&(data.index.weekday == wkday)].mean())
                meanss = meanss + [means]

        #replace na data
        for i in idx:
            data.loc[i] = meanss[i.weekday()*23 + i.hour]

        dfl = dfl + [data]

    new_df = pd.concat(dfl)
    df[tag]= new_df.values


for tag in interpolate_tag:
    df[tag] = df[tag].interpolate()


for tag in fillzero_tag:
    df[tag] = df[tag].fillna(0)

df['weekday'] = df.index.weekday
df['hours'] = df.index.hour

#can be replace by list of anything
from datetime import date
from workalendar.asia import Taiwan
cal = Taiwan()
holidayidx = []
for t in cal.holidays(2018):
    date_str = t[0].strftime("%Y-%m-%d")
    holidayidx = holidayidx + [date_str]

df['holiday'] = df.index.isin(holidayidx)

for tag in ['bemp','sbi']:

    for i in range(1,13):
        df[tag+'_'+str(i)+'h'] = df[tag].shift(i)

    for i in range(1,8):
        df[tag+'_'+str(i)+'d'] = df[tag].shift(i*24)


ndf = df
ndf['predict_hour'] = 1
for tag in ['bemp','sbi']:
    ndf['y_' + tag] = df[tag].shift(-1)

for i in range(1,13):
    ndf2 = df
    ndf2['predict_hour'] = i
    for tag in ['bemp','sbi']:
        ndf2['y_' + tag] = df[tag].shift(-i)
        #ndf2.to_csv("csvfile/parsed/new_data_sno1_predict_"+tag+"_"+str(i)+"h.csv")
    ndf = ndf.append(ndf2,ignore_index=True)

ndf = ndf.dropna()

ndf.to_csv("csvfile/parsed/new_data_sno1_predict.csv")
