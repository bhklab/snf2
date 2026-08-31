from snf2 import fuse, make_affinity, affinity_matrix
import pandas as pd

import numpy as np
from itertools import combinations, product
from sklearn.impute import SimpleImputer
from scipy import stats
import sys
from sklearn.metrics import roc_auc_score, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
# from mvfusion.mv_integrator import MultiViewIntegrator
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

# mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4,
#                                                        fpSize=1024,
#                                                        useBondTypes=True,
#                                                        includeChirality=True)


#data pipelining. 
# df = pd.read_parquet(dirs.RAWDATA / "JUMP" / "cpcnn.parquet")

####
#
#
#   As a first step we fix the drug universe to be all the drugs with MOA
#       CHOICE: for each modality do separate overlap or to do overlap of all modalities.
#               ACTION: look at all common samples right now because that  matches DNF1 approach
#
####

SEED = 1234
np.random.seed(SEED)


molMeta = pd.read_csv("rawdata/HDD/colData.csv", 
                               usecols= ['HDD.Compound.ID',"Pubchem.CID","LINCS.CMap.Name",'GEOM.Source.SMILES',"JUMP.CP.ID"])


JUMP_map = molMeta[["HDD.Compound.ID","JUMP.CP.ID"]]
JUMP_map = JUMP_map.dropna(subset=['JUMP.CP.ID'])
LINCS_map = molMeta[["HDD.Compound.ID","LINCS.CMap.Name"]]
LINCS_map  = LINCS_map.dropna(subset=["LINCS.CMap.Name"])
GEOM_map = molMeta[["HDD.Compound.ID",'GEOM.Source.SMILES']]
VIABILITY_map = molMeta[["HDD.Compound.ID","Pubchem.CID"]]


VIABILITY = pd.read_csv("procdata/NCI60.csv",index_col=0)
VIABILITY = VIABILITY.dropna(subset=['cid'])
VIABILITY["Pubchem.CID"]=[int(x) for x in VIABILITY['cid']]
VIABILITY = VIABILITY_map.merge(VIABILITY,on="Pubchem.CID")
VIABILITY = VIABILITY.drop(labels=['cid','Pubchem.CID','treatmentid'],axis=1)
VIABILITY = VIABILITY.set_index('HDD.Compound.ID')
# print(sorted(VIABILITY.isna().sum()))

# X = VIABILITY.values
# imp = IterativeImputer(missing_values=np.nan,random_state=SEED)
# X = imp.fit_transform(X)
# VIABILITY = pd.DataFrame(X, index=VIABILITY.index,columns = VIABILITY.columns)
viability_mols = set(VIABILITY.index)


##
#
#    prep jump data
###

JUMP_data = pd.read_parquet("rawdata/JUMP/cpcnn.parquet")
JUMP_meta = pd.read_csv("rawdata/JUMP/colData.tsv",sep="\t",usecols=["Sample.ID","JUMP.CP.ID"])



JUMP_data = JUMP_meta.merge(JUMP_data,on="Sample.ID")

JUMP_data = JUMP_map.merge(JUMP_data,on="JUMP.CP.ID")

JUMP_data = JUMP_data.drop(labels=["Sample.ID","JUMP.CP.ID"],axis=1)

JUMP_data = pd.DataFrame(JUMP_data.groupby("HDD.Compound.ID").mean())

jump_mols = set(JUMP_data.index)
# print(jump_mols)
# JUMP_mols = JUMP_data[]
####
#
# Prep LINCS
#
###

LINCS_data = pd.read_parquet("rawdata/LINCS/signatures.parquet")
# LINCS_meta = pd.read_csv(dirs.RAWDATA / "LINCS"/ "colData.tsv",sep="\t")

LINCS_data = LINCS_map.merge(LINCS_data,on= "LINCS.CMap.Name")
LINCS_data = LINCS_data.drop(labels=["LINCS.CMap.Name"],axis=1)
# print(LINCS_data.iloc[:5,:5])

LINCS_mols = set(LINCS_data['HDD.Compound.ID'])
LINCS_data = LINCS_data.set_index('HDD.Compound.ID')


####
#
# Prep GEOM
#
####

GEOM_data = pd.read_parquet("rawdata/GEOM/WHIM_hp.parquet")

GEOM_data = GEOM_map.merge(GEOM_data,on='GEOM.Source.SMILES')
GEOM_data = GEOM_data.drop(labels=['GEOM.Source.SMILES'],axis=1)

GEOM_mols = set(GEOM_data['HDD.Compound.ID'])
GEOM_data = GEOM_data.set_index('HDD.Compound.ID')

# WHIM_TYPES = ['WHIMS_avg.parquet','WHIMS_wavg.parquet','WHIMS_hp.parquet']




colData = pd.read_csv("rawdata/HDD/colData.csv")
colData = colData.dropna(subset=['Mechanism.of.Action'])


# mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4,
#                                                        fpSize=1024,
#                                                        useBondTypes=True,
#                                                        includeChirality=True)
# print("HDD.Compound.ID" in geom.columns)

count = pd.DataFrame(colData['Mechanism.of.Action'].value_counts()).reset_index()
count = count[count['count']>1]
colData = colData[colData['Mechanism.of.Action'].isin(count['Mechanism.of.Action'])]



keep_mols = set(colData['HDD.Compound.ID'])
keep_mols = keep_mols.intersection(GEOM_mols)
keep_mols = keep_mols.intersection(LINCS_mols)
keep_mols = keep_mols.intersection(jump_mols)
keep_mols = keep_mols.intersection(viability_mols)
keep_mols = sorted(list(keep_mols))
LINCS = LINCS_data.loc[keep_mols,]
print(LINCS.iloc[:5,:5])
print(LINCS.shape)
GEOM = GEOM_data.loc[keep_mols,]
print(GEOM.shape)
print(GEOM.iloc[:5,:5])
JUMP = JUMP_data.loc[keep_mols,]
print(JUMP.iloc[:5,:5])
print(JUMP.shape)
RESP = VIABILITY.loc[keep_mols,]
print(RESP.iloc[:5,:5])
print(RESP.shape)


N = len(keep_mols)

X = pd.DataFrame(np.zeros((N,N)),index=keep_mols,columns=keep_mols)

for moa in pd.unique(colData['Mechanism.of.Action']):
    temp = colData[colData['Mechanism.of.Action']==moa]
    for m1,m2 in combinations(temp['HDD.Compound.ID'],2):
        if m1 in keep_mols and m2 in keep_mols:
            X.loc[m1,m2]=1
            X.loc[m2,m1] =1
            # X[mol_to_idx[m2],mol_to_idx[m1]]=1




k = 20
name_to_view =  {"GEOM":GEOM, "LINCS":LINCS,"JUMP":JUMP,"NCI60":RESP}

all_combos = []
for i in range(2,len(name_to_view)+1):
    all_combos+=[x for x in list(combinations(name_to_view.keys(),i))]


results = defaultdict(list)
for combo in all_combos:
    views = [name_to_view[x] for x in combo]
    affinities= []
    for view in combo:
        if view == 'NCI60':
            comp = 1- RESP.T.corr()
            aff = affinity_matrix(comp,k=k)
            affinities.append(aff)
        else:
            aff = make_affinity(name_to_view[view].values,n_neighbors=k)
            affinities.append(aff)

    # affinities = [make_affinity(x,K=k) for x in [view.values for view in views]]

    FUSED=fuse(affinities)
    # integrator = MultiViewIntegrator(
    #             views = views,
    #             view_names=combo,
    #                 metrics = ['sqeuclidean']*len(views),
    #                 neighborhood_size= k,
    #                 mu= 0.4,
    #                 alignment_epochs=500,
    #                 emb_dim = 10,
    #                 seed = 30)
    # embeds_final, S_final, model = integrator.neural_integration()

    for name, aff in zip(
        ['SNF'],
        [FUSED]):
        print(name)
        
        
        fpr, tpr, thresh = metrics.roc_curve(X.values.flatten(), aff.flatten())
        auc = metrics.roc_auc_score(X.values.flatten(), aff.flatten())
        
        results['Model'].append(name)
        results['Modalities'].append("+".join(combo))
        results['AUC'].append(auc)
results = pd.DataFrame(results)
print(results)
results.to_csv('ablation.csv')



# hyperparameter analysis
# umap moa clustering
# ADC pipeline and qc
# single modality 


results = defaultdict(list)
for k in range(5,100):


    affinities= []
    view_to_aff = {}
    for view in ['GEOM','LINCS','JUMP','NCI60']:
        print(f"working on {view}")
        if view == 'NCI60':
            comp = 1- RESP.T.corr().fillna(0)
            aff = affinity_matrix(comp,k=k)
        else:
            aff = make_affinity(name_to_view[view].values,n_neighbors=k)


        affinities.append(aff)
        view_to_aff[view]=aff


  
    FUSED= fuse(affinities)

    print(affinities[0]==affinities[1])

    print("fusion done?")

    plt.figure(0).clf()
    for name, data in zip(
        ['GEOM','LINCS','JUMP','NCI60','SNF'],
        [GEOM,LINCS,JUMP,RESP,FUSED]):
        
        if name not in ['SNF']:
           if name == 'NCI60':
                comp = 1- RESP.T.corr().fillna(0)
                aff = affinity_matrix(comp,k=k)
            else:
               aff = make_affinity(name_to_view[view].values,n_neighbors=k)
           
        else:
            aff=data
        fpr, tpr, thresh = metrics.roc_curve(X.values.flatten(), aff.flatten())
        auc = metrics.roc_auc_score(X.values.flatten(), aff.flatten())
        
        results['K'].append(k)
        results['AUC'].append(auc)
        results['Model'].append(name)
            
        plt.plot(fpr,tpr,label=f"{name} - {auc}")


    plt.legend(loc=0)
    plt.plot([0, 1], [0, 1], linestyle='--')
    if k%5==0:
        plt.savefig(f"moa_roc_{k}.png")
    plt.close()



results = pd.DataFrame(results)
results.to_csv('auc.csv')


