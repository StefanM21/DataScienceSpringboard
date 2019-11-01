# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:35:24 2019

@author: SMA
"""

def cleanAge(x):
    if isinstance(x,int):
        return(x)
    elif isinstance(x,str):
        vor = x.rsplit(" ")
        if "n" in x.lower():
            return(np.nan)
        else:
            return([int(s) for s in vor if s!=""][0])
            
    
def naCounts(df):
    out_di = defaultdict(list) 
    for i,k in df.iteritems():
        na_vec = pd.isna(k)
        ratio=sum(na_vec)/len(na_vec)
        out_di[i].extend([ratio,na_vec])
        print("Nas of {}: {}".format(i,ratio))
    return(out_di) 
    
#container = {"Product":list(),"reI":list(),"Date":list(),"id":list(),"ProductN":list(),"reIN":list()}  
    
def targetS(ix,container,sub):
    sshape = sub.shape[0]
    datevals = sub.index.values
    for k in range(1,sshape):
        ref = sub.iloc[k-1]
        comp = sub.iloc[k]
        bols = comp != ref
        ixx =100
        name_vals = " "
        ixn = 100
        name_vals_n=" "
        
        bolsp = (bols==True) & (comp[bols]==1)
        bolsn = (bols==True) & (comp[bols]==0)
        if sum(bolsp) >0: 
            name_vals = comp[bolsp].index.values.tolist()[0]
            ixx = np.where(bolsp==True)[0].tolist()[0]
        if sum(bolsn) >0:
            name_vals_n = comp[bolsn].index.values.tolist()
            ixn = np.where(bolsn==True)[0].tolist()[0]
        #print(ixn)
        container["Product"].append(name_vals[0])
        container["reI"].append(ixx)        
        container["reIN"].append(ixn)
        container["Date"].append(datevals[k-1])
        container["id"].append(ix)
        container["ProductN"].append(name_vals_n[0])
        

def na_dropper(df,col,find_alt=False):
    """ functions looks up nas in col and deletes rows by id"""
    kill =list()    
    boolNA = pd.isna(df[col])
    idu = df[boolNA]["id"].unique()
    ixval = df.index.values
    if find_alt == False:
        for s in idu:
            kill.extend(ixval[s==df.id].tolist()) 
    else:
        for s in idu:
            bol = s==df.id
            sv = df.id[bol]
            val = df[sv.iloc[0] == df.id][col]
            if len(val) >0:
                df[bol][col] = val
            else:
                kill.extend(ixval.id[bol].to.list())
                
                
   