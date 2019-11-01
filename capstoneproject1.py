# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:58:53 2019

@author: SMA
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import os
from collections import defaultdict

os.chdir("C:/Users/SMA/Desktop/DataScienceSpringboard/Capstone1/santander-product-recommendation")

df =  pd.read_csv("train_ver2.csv",parse_dates=["fecha_dato"])

#### are in every month customers unique? 

count_values = df.ncodpers.value_counts()
count_values.hist()

### most have 17, i.e. 17 months history in the bank 

### manipulate col names to english 
col_holder =df.columns 

df.columns =["Date","id","EmployeeBank","ResidenceCountry","Sex","Age","first_contract_date","new_customer",
             "seniority_time","primary_cust_in_month","last_date_as_primary","cust_type","cust_relation_type",
             "residence_index","foreigner_birth","spouse_with_employee","channel_used","deceased","address_type",
             "province_code","province_name","activity_index","gross_income","customer_segment","Saving_Account",
             "Guarantees","Current_Account","Derivada_Account","Payroll_Account","Junior_Account","MP_Account",
             "P_Accouont","PP_Account","St_deposit","MT_deposit","LT_deposit","e_account","Funds","Mortage",
             "Pensions","Loans","Taxes","Credit_Card","Securities","Home_Account","Payroll","Pension","Direct_Debit"]


df.info()
#### cleaning steps:
## Age is object
#### convert first_contract date to datetime and measure difference to current date
### Residence Country: count values and classify binary if country is not spain - already done in residence index
### last date as primary, is date column; however it says that the account has been shared with wife etc and this might have value; thus
### difference to current date could be have value 
### cust_type has different interpretation : 1.0 versus 1.00
###  what does N mean in cust_relation_type ?
### spouse with employee has many NAs  / correct it
### deceased kick the dead ones out ? did deceased ones buy something before leaving ? 
### province code and name-- kick it out? or find out if  its a city..makes sense?
### activity index: is eqaully spaced has meaning 
### customer segment is differntyl labeled ..
## gross income has many Na values ... // maybe permutate it 
### count number frequency of items in general 

## make strings out of sparse mat --> simplest cluster 

new_Agew = df.Age.apply(cleanAge)
df["Age"] = new_Agew


u =df[["cust_type"]].replace({"P":5})
df["cust_type"] = u.astype(float)


df["Sex"].replace({"H":"M","V":"W"},inplace=True)

nana = naCounts(df)



m = df.Sex.value_counts()

plt.bar(x=["V","H"],height=[m.H,m.V])

nana = naCounts(df)


#### reduce the whole set to a number of ids which are in data for at least 4 times

count_of_ids = df.id.value_counts()

selection1 = count_of_ids[count_of_ids > 5]

np.random.seed(123)

selection2 = np.random.choice(selection1.index.values,5000)

df2 = df.copy()

df2.set_index("id",inplace=True)

df = df2.loc[selection2]

df.reset_index(inplace=True)

df.set_index(["Date","id"],inplace=True,drop=False)

df.sort_index(level=0,inplace=True)

uni_id = df.index.get_level_values(1).unique()
len(uni_id)

#i = uni_id[0]
container = {"Product":list(),"reI":list(),"Date":list(),"id":list(),"ProductN":list(),"reIN":list()}  

for i in uni_id:
    sub = df.xs(i,level=1)
    sub = sub.loc[:,"Saving_Account":]
    targetS(i,container,sub)

df_container = pd.DataFrame(container)

df_container.set_index(["Date","id"],inplace=True,drop=True)

df_j = pd.concat([df,df_container],axis=1,levels=[0,1]) 


#### make a check

iv = df_j.index.values 
co = list(zip(df_j["Date"],df_j["id"])) 
contr = list()

for m in range(0,df_j.shape[0]):
    a,b = iv[m]
    if b== df_j["id"][m] and a== df_j["Date"][m]:
        contr.append(True)
    else:
        break


