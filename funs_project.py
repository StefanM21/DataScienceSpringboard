# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:50:04 2019

@author: SMA
"""

def replace_NA_Persons(df,colX):
    colM = df[colX].copy()
    masK = colM.isna().to_numpy()
    iDD = colM[masK].reset_index("id")["id"]
    to_searchIN = colM.reset_index("Date")
    to_search = to_searchIN.loc[set(iDD),:]    
    to_searchIN = to_search.reset_index("id")
    if df.dtypes[colX] =="float64":
        replS = to_searchIN.groupby("id")[colX].mean()
    else:
        replS = to_searchIN.groupby("id")[colX].apply(lambda x: x.unique()[0])
    replacing = dict(zip(replS.index.values.tolist(),replS.tolist()))
    colK = colM[masK]
    ixx = colK.index
    colV = colK.reset_index("id")
    res=colV["id"].map(replacing).to_numpy()
    colM[masK] = res       
    return(colM) 
    
    
    
## the function defines a lookup table 
 ### this function separates into a reference NONA field which does not contain NAs in feature X; then for continuous
### variables deciles are calculated and according to them for other other variables the most common values are taken
### for comparision; for categorical NA features simply the the labels are taken    
    
def lookuptable(colX,mm):
    colNA = mm[colX]
    mask = colNA.isna().to_numpy()
    NAfield = mm[mask].loc[:,:"customer_segment"].copy()
    NONA = mm[~mask].loc[:,:"customer_segment"].copy()
    
    colto_use = NAfield.isna().sum() ==0
    colto_use[colX]=True
    
    NAfield= NAfield[colto_use.index[colto_use]]
    NONA = NONA[colto_use.index[colto_use]]
    
    NONA.reset_index("id",inplace=True)    
    NONA.drop_duplicates("id",inplace=True)    
    NONA.drop("id",axis=1,inplace=True)
   
    refCOL = NONA[colX]
    for g in NONA.columns:
        if NONA.dtypes[g].name =="float64":
            NONA[g] = pd.qcut(NONA[g],10,duplicates="drop")
    
    refTabel = NONA.groupby(colX)[NONA.columns.drop(colX)].apply(lambda x: x.mode())
    r_1 = pd.concat([NONA[colX],refCOL],axis=1)
    r_1.columns = [colX,colX+"_"]
    if mm.dtypes[colX].name=="category":
        meanS = refTabel.reset_index(level=0)[colX].values
    else:
        meanS =  r_1.groupby(colX).mean().to_numpy()    
    xa = NAfield.pop(colX)
    xa = xa.to_numpy()
    xa1 = NONA.pop(colX)
    for gg in range(refTabel.shape[0]):
        booLK = refTabel.iloc[gg,:] == NAfield
        xa[np.where(booLK.sum(axis=1)> NAfield.shape[1]-5 )] = meanS[gg]
    
    output_col = mm[colX].copy()
    output_col.loc[NAfield.index] = xa    
    return(output_col)


