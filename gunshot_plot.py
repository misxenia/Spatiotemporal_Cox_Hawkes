
#
## This script runs the figure for the temporal distribution of the gunshots in the papers
#

%load_ext autoreload
%autoreload 2

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import pandas as pd

plt.style.use('seaborn-paper')
def custom_plot(width=10, height=10):
    mpl.rc('axes.spines', right=False, top=False)
    mpl.rc('axes', labelsize=20)
    mpl.rc('xtick', labelsize=16, top=False)
    mpl.rc('xtick.minor', visible=False)
    mpl.rc('ytick', labelsize=16, right=False)
    mpl.rc('ytick.minor', visible=False)
    mpl.rc('savefig', bbox='tight', format='pdf')
    mpl.rc('figure', figsize=(width, height))
    mpl.rc('legend',fontsize=16)
custom_plot(width=10, height=10)


df = pd.read_csv("~/gunfire_data_filtered_2006_2013.csv")

df.dropna(subset = ["X","Y","T"], inplace=True)

index_2013=pd.to_datetime(df['date']).dt.year==2013

indices=index_2013#&index_unique&index_holidays

x=np.round(df[indices]['T']-2532.870833)

df_2013 = df[indices]

x_ = plt.hist(x.to_numpy(),bins=52,density=False)

# plt.savefig(filename+mypath)
df_2013['Date'] = pd.to_datetime(df_2013['date'])

df_plot_data = df_2013.resample('W-Mon', on='Date')['Date'].count()

import matplotlib.dates as mdates

fig, ax = plt.subplots()
ax.bar(df_plot_data.index, df_plot_data.values, width=6)
# As given in your question I have chosen "YYMMDD"
date_formatter = mdates.DateFormatter('%b\n%Y')

# Set the major tick formatter to use your date formatter.
ax.xaxis.set_major_formatter(date_formatter)
# plt.gcf().autofmt_xdate()
ax.set_ylabel('Weekly gunshots')
ax.set_xlabel('Week in 2013')
plt.savefig("temporal_distribution.pdf")
