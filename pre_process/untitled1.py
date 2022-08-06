#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 16:21:34 2022

@author: kang
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.pyplot import figure

from os import listdir
from os.path import isfile, join
dir = "Data/"
sub_dir = "Whole_Book/"
onlyfiles = [f for f in listdir(dir+sub_dir) if isfile(join(dir+sub_dir, f))]
onlyfiles = sorted(onlyfiles)

filename = onlyfiles[1]
# filename = "AMZN_2021-04-01_34200000_57600000_orderbook_10.csv"
# %%
Dir = dir + sub_dir
name_lst = []
for i in range(40//4):
    name_lst.append("ask"+str(i+1))
    name_lst.append("ask"+str(i+1)+"_quantity")
    name_lst.append("bid"+str(i+1))
    name_lst.append("bid"+str(i+1)+"_quantity")
    
Flow = pd.read_csv(Dir + filename, names= name_lst)
# Flow.columns = name_lst
# , names=name_lst
# ----- %%
FlowTest = Flow.iloc[:,:]

# FlowTest.iloc[:,[i%2==0 for i in range(FlowTest.shape[1])]] 
# new = FlowTest.iloc[:,[i%2==0 for i in range(FlowTest.shape[1])]]
# New = new.apply(lambda x: x/10000,axis=1)

for i in range(20):
    FlowTest.iloc[:,2*i] = FlowTest.iloc[:,2*i].apply(lambda x: x/10000)
    
#%%
New = FlowTest.iloc[20400,:]
theDataAsk_Price = []
theDataAsk_Volume =  []
theDataBid_Price = []
theDataBid_Volume =  []
for i in range(New.shape[0]//4):
    theDataAsk_Price.append(New.iloc[4*i])
    theDataAsk_Volume.append(New.iloc[4*i+1])
    theDataBid_Price.append(New.iloc[4*i+2])
    theDataBid_Volume.append(New.iloc[4*i+3])
theDataAsk = pd.DataFrame({
    "Price" : theDataAsk_Price,
    "Volume": theDataAsk_Volume
    })
theDataAsk.sort_values(by=['Price'],inplace=True,ascending=False)
theDataBid = pd.DataFrame({
    "Price" : theDataBid_Price,
    "Volume": theDataBid_Volume
    })


# Chart
# fig = plt.figure()
# ax = fig.add_subplot(1,1,figsize=(12, 8))
fig, ax = plt.subplots(1,1,figsize=(18, 5))
 
# plt.ylim(0,max(theDataBid['Volume'].max(),theDataAsk['Volume'].max()) + 200)
plt.xlim(min(theDataBid['Price'].min(),theDataAsk['Price'].min()), max(theDataBid['Price'].max(),theDataAsk['Price'].max()))
# plt.suptitle('Limit Order Book Volume for ' + ticker + ' at ' + str(random_no))
plt.ylabel('Volume')
plt.xlabel('Price')


# price = list(theDataBid['Price'])
# price.extend(list(theDataAsk['Price']))
# plt.xticks(sorted(price))
 
ax.bar(theDataBid['Price'], theDataBid['Volume'], width = 0.02, color='#0303fe', label='Bid')
ax.bar(theDataAsk['Price'], theDataAsk['Volume'], width = 0.02, color='#fc1b04', label='Ask')
        
plt.legend()       
plt.show()

fig.savefig('img.png', dpi=300)


# %%
# import seaborn as sns
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
# sns.set_context('notebook')
# sns.set_theme(style="ticks",palette="Paired")
# f, ax = plt.subplots(figsize=(12, 8))
# sns.despine(f)

# sns.barplot(theDataBid['Price'], theDataBid['Volume'])
# sns.barplot(theDataAsk['Price'], theDataAsk['Volume'])

# plt.show()
#%%

# price_test = FlowTest.iloc[:,[i%2==0 for i in range(FlowTest.shape[1])]] 

# Flow.apply(lambda x: x/1000, axis = 1)

# price = Flow.iloc[:,[i%2==0 for i in range(Flow.shape[1])]]


# %%
# lst = []
# for i in range(price.shape[0]):
#     lst.extend(list(price.iloc[i,:]))
# price_set = set(lst)
    
# %%
book = FlowTest
dir = "Data/"
sub_dir = "Whole_Flow/"
onlyfiles = [f for f in listdir(dir+sub_dir) if isfile(join(dir+sub_dir, f))]
onlyfiles = sorted(onlyfiles)
filename = onlyfiles[1]
Dir = dir + sub_dir
flow = pd.read_csv(Dir + filename, names= ["time","type","id","quantity","price","direction","comment"])
# flow2 = flow[(flow.time>=36000) & (flow.time<=36600)]
# index_flow2 = flow2.index
book2 = book[(flow.time>=36000) & (flow.time<=36600)]
#%%
price = book2.iloc[:,[i%2==0 for i in range(book2.shape[1])]]
lst = []
for i in range(price.shape[0]):
    lst.extend(list(price.iloc[i,:]))
price_set = set(lst)
    # print(">>>{0}".format(len(price_set)))
min(price_set),max(price_set)
#%%
N = 40
# for i in range(N//4):
#     book3 = 
lst = [i for i in range(N) if (i%4==2) or (i%4==3)]
book3 = book2.iloc[:,lst]
book3_price = book3.iloc[:,[i for i in range(N//2) if i%2==0 ]]
book3_quantity = book3.iloc[:,[i for i in range(N//2) if i%2==1 ]]
# %%
diff = book3.diff()
index = 3
# column_index = 
diff.iloc[index,:] == 0
to_be_inversed_flow = pd.DataFrame((book3.iloc[index-1,:] * (1-(diff.iloc[index,:] == 0)))).T
to_come_flow = pd.DataFrame((book3.iloc[index,:] * (1-(diff.iloc[index,:] == 0)))).T
