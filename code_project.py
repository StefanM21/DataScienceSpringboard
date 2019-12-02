# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:42:52 2019

@author: SMA
"""
### load modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import os
from collections import defaultdict
import missingno
pd.set_option('display.float_format', lambda x: '%.5f' % x)

### load data
os.chdir("C:/Users/SMA/Desktop/DataScienceSpringboard/Capstone1/santander-product-recommendation")

df =  pd.read_csv("train_ver2.csv",parse_dates=["fecha_dato","fecha_alta"])
df.columns =["Date","id","EmployeeBank","ResidenceCountry","Sex","Age","first_contract_date","new_customer",
             "seniority_time","primary_cust_in_month","last_date_as_primary","cust_type","cust_relation_type",
             "residence_index","foreigner_birth","spouse_with_employee","channel_used","deceased","address_type",
             "province_code","province_name","activity_index","gross_income","customer_segment","Saving_Account",
             "Guarantees","Current_Account","Derivada_Account","Payroll_Account","Junior_Account","MP_Account",
             "P_Accouont","PP_Account","St_deposit","MT_deposit","LT_deposit","e_account","Funds","Mortage",
             "Pensions","Loans","Taxes","Credit_Card","Securities","Home_Account","Payroll","Pension","Direct_Debit"]
df.info()
### reduce size of sparse matrix 
colS = df.loc[:,"Saving_Account":].columns
for k in colS:
    df[k] = pd.to_numeric(df[k],downcast="unsigned")
    

df.Date.value_counts().plot(kind="bar")

df.id.value_counts().size
df.groupby("Date").id.nunique().plot(kind="bar")

df["EmployeeBank"] = df["EmployeeBank"].astype("category")

def cleanString(x):
    if isinstance(x,int):
        return(x)
    elif isinstance(x,str):
        vor = x.rsplit(" ")
        if "n" in x.lower():
            return(np.nan)
        else:
            return([int(s) for s in vor if s!=""][0])
            



df.ResidenceCountry.value_counts()
df.groupby("Date").ResidenceCountry.nunique()
df["ResidenceCountry"] = df["ResidenceCountry"].astype("category")

df.Sex.value_counts()
df["Sex"].replace({"V":"W","H":"M"},inplace=True)
df.groupby("Date").Sex.value_counts(normalize=True).plot(kind="bar",stacked=True) 
df["Sex"] = df["Sex"].astype("category")


df["Age"] = df["Age"].apply(cleanString)

df.Age.describe()


### replace date feature to time differences 
df.first_contract_date.value_counts()
df.first_contract_date.isna().sum()
delta_t = (df.Date  - df.first_contract_date)/np.timedelta64(1,"D")
df["first_contract_date"]= delta_t
df[df["first_contract_date"]<0] = np.nan 


df.new_customer.value_counts(normalize=True)
df.groupby("Date").new_customer.value_counts(normalize=True).plot(kind="bar",stacked=True)
df["new_customer"] = df.new_customer.astype("category")


df["seniority_time"]= df.seniority_time.apply(cleanString)
df["seniority_time"] = df.seniority_time.replace({-999999.0:np.nan})


df.primary_cust_in_month = df.primary_cust_in_month.astype("category")

df.last_date_as_primary

df.cust_type.value_counts()

df["cust_type"] = df.cust_type.replace({"P":5}).astype(float)
df["cust_type"] = df.cust_type.astype("category")

df.cust_relation_type.value_counts()
df["cust_relation_type"] = df["cust_relation_type"].replace({"N":np.nan})
df.groupby("Date").cust_relation_type.value_counts()
df["cust_relation_type"] = df["cust_relation_type"].astype("category")

df["residence_index"].value_counts()
df.residence_index.isna().sum()
df["residence_index"]= df.residence_index.replace({"S":"Y"}).astype("category")

df.foreigner_birth.value_counts()
df["foreigner_birth"] = df.foreigner_birth.replace({"S":"Y"}).astype("category")

df.spouse_with_employee.isna().sum()
df["spouse_with_employee"] = df.spouse_with_employee.replace({"S":"Y"}).astype("category")

df.channel_used.value_counts()
df.channel_used.unique()
df["channel_used"] = df.channel_used.astype("category")

df["deceased"].value_counts()
df["deceased"] = df.deceased.replace({"S":"Y"}).astype("category")

df.address_type.value_counts()
df["address_type"] = df.address_type.astype("category")

df["province_code"].value_counts()


df.province_name.value_counts()
df["province_name"] = df.province_name.astype("category")

df.activity_index.value_counts()
df["activity_index"] = df.activity_index.astype("category")

df["customer_segment"].value_counts()
df["customer_segment"].replace({"02 - PARTICULARES":2,"03 - UNIVERSITARIO":3,"01 - TOP":1},inplace=True)
df["customer_segment"]  = df["customer_segment"].astype("category")
df.gross_income.describe()

### screen for outliers
out = defaultdict(list)

cols = df.columns[2:19]
for col in cols:
    mcol = df[col]
    if df.dtypes[col] =="float64":
        dev = mcol.mean() + 3*mcol.std()
        ixx_to = np.where(mcol>dev)[0].tolist()
        out[col].extend([len(ixx_to)/len(mcol),ixx_to])
    else: 
        dev = mcol.value_counts(normalize=True)
        ixm = dev[dev< 0.015].index 
        ixx_to = []
        for i in ixm:
            ixx_to.extend(np.where(i==mcol)[0])
        out[col].extend([len(ixx_to)/len(mcol),ixx_to])


nameS = list();values=list()           
for k in out.keys():
    if out[k][0]>0:
        values.append(out[k][0])
        nameS.append(k)
        
outliers = pd.Series(values,index=nameS)

### Age and Gross Income is checked manually
### Age: most people's age is limited and children are not likely to take responsibility over an account 
age = df.Age.copy()
age[age >110]= np.nan
age[age <12]= np.nan
### very unlikely that people are having more than 2.5 million gross income.. 
gI = df.gross_income.copy()
gI[gI>2500000] =  np.nan

df["Age"]=age
df["gross_income"]=gI


## now calculate target values, i.e. sparse matrix of products that will be added the next month

### separate sparse matrix and sort values accoding to client id and Date
### drop unique id entries: there is nothing they will add

sparse = df.loc[:,"Saving_Account":"Direct_Debit"].copy()

sparse["Date"] = df["Date"]
sparse["id"] = df["id"]

su = sparse[sparse.id.duplicated(False)].copy()

su.sort_values(["id","Date"],inplace=True)

su_date_id = su[["Date","id"]]
su.drop(columns= ["Date","id"],inplace=True)


### calculate product migration over whole matrix; therefore identification of last entries along customer ids is necessary
### then simple boolean comparision

last_one = np.where(su_date_id["id"] != su_date_id["id"].shift(-1))[0]

a = su.to_numpy()

ref = su.to_numpy() 
comp = su.shift(-1).to_numpy()

boolS = (ref!= comp) & (comp==1)

new_boolS = np.delete(boolS,last_one,axis=0)
#### give back indices where changes in product basket occurred and transform to sparse matrix output 
x,y = np.where(new_boolS ==True)

s1 = su_date_id["id"].to_numpy()
s1 =np.delete(s1,last_one,axis=0)

s2 = su_date_id["Date"].to_numpy()
s2 = np.delete(s2,last_one,axis=0)

data_sp = {"ids":s1[x],"Dates":s2[x],"values_":y.tolist()}

sp_target = pd.DataFrame(data_sp)

sp_target["ik"]= 1
spx = sp_target.pivot_table(index=["Dates","ids"],columns="values_",values="ik")
spx.fillna(0,inplace=True)

### concatenate sparse target matrix with frame

spx.sort_index(level=[0,1],inplace=True)

spx.columns = df.columns[24:] + "_target"

spx.index.names =["Date","id"]

df.set_index(["Date","id"],inplace=True)
df.sort_index(level=[0,1],inplace=True)

nd = pd.concat([df,spx],axis=1,ignore_index=False)


#### fill Na with zero; i.e. Na result from those who did not have any change in basket and those who were unique
repl = nd.loc[:,"Saving_Account_target":].copy()
repl.fillna(0,inplace=True)
nd.loc[:,"Saving_Account_target":] = repl.to_numpy()

### generate additional features: a) how many products has a client purchased in comparsion to the month before 
### b) how many products has a client sold off before 

first_one = np.where(su_date_id["id"]!= su_date_id["id"].shift(1))[0]
compp = su.shift(1).to_numpy()
boolSN = (ref==1) & (compp != ref)
boolSL = (ref==0) & (compp != ref)
boolSN = np.delete(boolSN,first_one,axis=0)
boolSL = np.delete(boolSL,first_one,axis=0)
s_1 = np.delete(su_date_id["id"].to_numpy(),first_one,axis=0)
s_2 = np.delete(su_date_id["Date"].to_numpy(),first_one,axis=0)
switching = {"id":s_1,"Date":s_2,"new_ones":boolSN.sum(axis=1),"leave_ones":boolSL.sum(axis=1)}
switching = pd.DataFrame(switching)
switching.set_index(["Date","id"],inplace=True)
switching.sort_index(level=[0,1],inplace=True) 

#### merge an fill missings with 0 
nd = pd.concat([nd,switching],axis=1,ignore_index=False)
ndex = nd[["new_ones","leave_ones"]].copy()
ndex.fillna(0,inplace=True)
nd.loc[:,["new_ones","leave_ones"]] = ndex.to_numpy()

### clean up
for i in nd.loc[:,"Saving_Account_target":].columns:
    nd[i] = pd.to_numeric(nd[i],downcast="unsigned")
df = nd.copy()

del(nd)
os.chdir("...")
df.to_pickle("df_targets.pkl")

#### NA visualization
df.loc[:,:"customer_segment"].isna().sum().plot(kind="bar")

### na frequency per rows 
df.loc[:,:"customer_segment"].isna().sum(axis=1).hist()

### is there a systematic dependence between NA values?
missingno.heatmap(df.loc[:,:"customer_segment"])

#### first removals:
### spouse_with_employee, last_date_as_primary,
### desceased ones True // we cannot predict anything on deceased ones
### province_code// same info as province_name
### drop na values in basket as well 
df.drop(["spouse_with_employee","last_date_as_primary","province_code"],axis=1,inplace=True)

mask = (df["Payroll"].isna()) | (df["Pension"].isna())

df = df[~mask].copy()

mask = df.deceased=="Y"
df = df[~mask]
df.drop("deceased",inplace=True,axis=1)

cols = ["Sex","Age","seniority_time","cust_type","cust_relation_type","channel_used","province_name",
        "gross_income","customer_segment"]


for col in cols:
    df[col] = replace_NA_Persons(df,col)
df.isna().sum()[:19]

df.isna().sum(axis=1).value_counts()
####
#### there are 11255 rows which are full of NA readings in their clients' features
### moreover, there are 89K wih 5 NA readings 
### it is fairly difficult to find a proper reasoning to replace these values with lookup tables

rem_mask = df.isna().sum(axis=1) >10
    
df = df[~rem_mask.to_numpy()]
df.isna().sum()[:19]
(df.cust_relation_type.isna() & df.cust_type.isna()).sum()

nacols = ["Sex","Age","seniority_time","cust_type","cust_relation_type","channel_used","province_name","gross_income","customer_segment",
         "first_contract_date"]

for ff in nacols:
    df[ff] = lookuptable(ff,df)

df.isna().sum()[:20]
## A couple of NA values have not been replaced possibly due to ambiguities in the lookup frame
## they will be removed; moreover, the column deceased is getting removed as well due unique values
df.dropna(axis=0,inplace=True)


######plottings
##### 4 plots are generated: distributional with and without correction; plot in relation to target and time dependence
### first: number of targets and binary Target is calculated
num_targets =  df.loc[:,"Saving_Account_target":"Direct_Debit_target"].copy().sum(axis=1).to_numpy()
df["Number_targets"] =num_targets
num_targets[num_targets > 0] = 1
df["Target"] = num_targets



### due to high number of labels in channel_used, province_name, ResidenceCountry an overwrite is done, i.e. creation of a
### a fake dataset for plotting 

def overwrite(x,vals):
    if x not in vals:
        return ("O")
    else:
        return(x)

cor_ch = df.channel_used.copy()
cor_ch = cor_ch.apply(overwrite,vals=["KHE","KAT","KFC"]).astype("category")

cor_ch2 = df.province_name.copy()
cor_ch2 = cor_ch2.apply(overwrite,vals=df.province_name.value_counts().iloc[0:6].index.values).astype("category")

cor_ch3 = df.ResidenceCountry.copy()
cor_ch3 = cor_ch3.apply(overwrite,vals=df.ResidenceCountry.value_counts().iloc[0:6].index.values).astype("category")



fake_da = df.copy()
fake_da["province_name"] = cor_ch2.to_numpy()
fake_da["channel_used"] = cor_ch.to_numpy()
fake_da["ResidenceCountry"] = cor_ch3.to_numpy()


cols = df.columns[:17].tolist()
cols.extend(["new_ones","leave_ones"])
mask = (df["Target"] ==1).to_numpy()
#os.chdir("D:/Project_II/plots2")
for col in cols:
    colS = fake_da[col].copy()
    fig,axS = plt.subplots(2,2,figsize=(10,10),constrained_layout=True)
    #fig.tight_layout()
    if fake_da.dtypes[col].name =="category":
        if col =="province_name":
            colS = cor_ch2
        if col =="channel_used":
            colS = cor_ch
        oS =colS.value_counts(normalize=True).reset_index()
        sns.barplot(x="index",y=col,data=oS,ax=axS[0,0])
        axS[0,0].set_title("Proportions",loc="left")
        colS= colS.reset_index(level=1)
        colS.drop_duplicates(subset="id",keep="first",inplace=True)
        oS = colS[col].value_counts(normalize=True).reset_index()
        sns.barplot(x="index",y=col,data=oS,ax=axS[0,1])
        axS[0,1].set_title("Proportions with corrections",loc="left")
        #
        wT = fake_da[mask][col]
        nT = fake_da[~mask][col]
        bf = pd.concat([wT.value_counts(normalize="True"),nT.value_counts(normalize=True)],axis=1,ignore_index=False)
        bf.columns= ["purchase","no_purchase"]
        bf.T.plot(kind="bar",stacked=True,ax=axS[1,0])
        axS[1,0].set_title("Proportoins separated by target",loc="left")
        #
        hh = fake_da[col].groupby(level=0).value_counts(normalize=True)
        hh.unstack().plot(kind="bar",stacked=True,ax=axS[1,1])
        axS[1,1].set_title("Proportions over time",loc="left")
                
    else:
        sns.distplot(colS,ax =axS[0,0])
        axS[0,0].set_title(label="Histogram",loc="left")
        colS = colS.reset_index(level=1)
        colS.drop_duplicates("id",inplace=True)
        sns.distplot(colS[col],ax=axS[0,1])
        axS[0,1].set_title("Histogram with correction",loc="left")
        
        co =fake_da[[col,"Target"]]
        sns.boxplot(x="Target",y=col,data=co,ax=axS[1,0])
        axS[1,0].set_title("boxplot separate by target",loc="left")
        fake_da[col].groupby(level=0).mean().plot(ax=axS[1,1])
        axS[1,1].set_title(label="Median over time",loc="left")
    fig.suptitle("{} : desciptive graphs".format(col),fontsize=20)
    #plt.savefig(col+".png")
    
    
### graphing the product basket 
product_basket = mm.loc[:,"Saving_Account":"Direct_Debit"]

perc_av = product_basket.sum(axis=0)
perc_av = perc_av/perc_av.sum()
plt.figure(figsize=(10,10))
perc_av.plot(kind="bar")
plt.title("Percentage of products")

plt.figure(figsize=(10,10))
product_basket.sum(axis=1).hist()
plt.title("Frequency of Basket Size")


### statistic CramersV and heatmap 
from itertools import permutations
import researchpy
cols = df.columns[:19]
perm = permutations(cols,2)       
outP = {"indeX":list(),"colS":list(),"valueS":list()}
ii = 0
for hh in list(perm)[67:]:
    col1 = mm[hh[0]]
    col2 = mm[hh[1]]
    if mm.dtypes[hh[0]].name =="float64":
        col1 =pd.qcut(col1,5)
    if mm.dtypes[hh[1]].name =="float64":
        col2 = pd.qcut(col2,5)
    
    a,b = researchpy.crosstab(col1,col2,test="chi-square")
    outP["valueS"].append(b.iloc[2,1])
    outP["indeX"].append(hh[0])
    outP["colS"].append(hh[1]) 
    ii+=1
    print(ii)

rouP = pd.DataFrame(outP)
mmK = rouP.pivot_table(index="indeX",columns="colS",values="valueS")
mmK.fillna(1,inplace=True)

mmK.to_pickle("heatmap.pkl")

fig=plt.figure(figsize=(10,10))
sns.heatmap(mmK,cmap="BuPu")               
plt.savefig("heatmap_1.png")


##chi square over samples 
ok = {"feature":list(),"p-value":list(),"Cramers":list()}
for hh in cols:
    col1 = mm[hh]
    col2 = mm["Target"]
    if mm.dtypes[hh].name =="float64":
        col1 =binarize(mm,hh,q=5)
    a,b = researchpy.crosstab(col2,col1,test="chi-square")
    ok["Cramers"].append(b.iloc[2,1])
    ok["p-value"].append(b.iloc[1,1])
    ok["feature"].append(hh)
    



