import numpy as np
import pandas as pd
from snf2 import fuse, make_affinity, affinity_matrix
import time
from collections import defaultdict
import tqdm
molMeta = pd.read_csv("rawdata/HDD/colData.csv", 
                               usecols= ['HDD.Compound.ID',"Pubchem.CID","LINCS.CMap.Name",'GEOM.Source.SMILES',"JUMP.CP.ID"])


VIABILITY_map = molMeta[["HDD.Compound.ID","Pubchem.CID"]]

VIABILITY = pd.read_csv("procdata/NCI60.csv",index_col=0)
VIABILITY = VIABILITY.dropna(subset=['cid'])
VIABILITY["Pubchem.CID"]=[int(x) for x in VIABILITY['cid']]
VIABILITY = VIABILITY_map.merge(VIABILITY,on="Pubchem.CID")
VIABILITY = VIABILITY.drop(labels=['cid','Pubchem.CID','treatmentid'],axis=1)
VIABILITY = VIABILITY.set_index('HDD.Compound.ID')


res = defaultdict(list)
for n in tqdm.tqdm(np.arange(100,10000,100)):
    V = VIABILITY.iloc[:n,:]
    s = time.time()    
    comp = 1- V.T.corr()

    for k in [5,10,15,20,50]:
        aff = affinity_matrix(comp,k=k)
        e = time.time()
    
        res['Num. Obs'].append(n)
        res['Computation Time'].append(e-s)
        res['Num. Neighbors'].append(k)

res = pd.DataFrame(res)
res.to_csv("compute_time.csv")