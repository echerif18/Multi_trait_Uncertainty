import pandas as pd
import numpy as np


ls_tr = ['LMA_g_m2', 'N_area_mg_cm2', 'LAI_m2_m2', 'C_area_mg_cm2',
       'Chl_area_ug_cm2', 'EWT_mg_cm2', 'Car_area_ug_cm2', 'P_area_mg_cm2',
       'Lignin_mg_cm2', 'Cellulose_mg_cm2', 'Fiber_mg_cm2', 'Anth_area_ug_cm2',
       'NSC_mg_cm2', 'Mg_area_mg_cm2', 'Ca_area_mg_cm2',
       'Potassium_area_mg_cm2', 'Boron_area_mg_cm2', 'Cu_area_mg_cm2',
       'S_area_mg_cm2', 'Mn_area_mg_cm2']

# Traits = ['LMA_g_m2', 'N_area_mg_cm2',
# 'LAI_m2_m2', 'C_area_mg_cm2',
# 'Chl_area_ug_cm2', 'EWT_mg_cm2',
# 'Car_area_ug_cm2', 'P_area_mg_cm2',
# 'Lignin_mg_cm2', 'Cellulose_mg_cm2', 
# 'Fiber_mg_cm2', 'Anth_area_ug_cm2',
# 'NSC_mg_cm2', 'Mg_area_mg_cm2',
# 'Ca_area_mg_cm2', 'Potassium_area_mg_cm2',
# 'Boron_area_mg_cm2', 'Cu_area_mg_cm2', 
# 'S_area_mg_cm2',  'Mn_area_mg_cm2',
# # 'Al_area_mg_cm2','Flavonoids_area_mg_cm2',
# # 'Iron_area_mg_cm2','Phenolics_area_mg_cm2',
# # 'Protein_g_m2', 'Starch_area_mg_cm2',
# # 'Sugar_area_mg_cm2','Zn _area_mg_cm2'
# ]


Traits = ['LMA (g/m²)', 'N content (mg/cm²)', 'LAI (m²/m²)', 'C content (mg/cm²)', 'Chl content (μg/cm²)', 'EWT (mg/cm²)', 
'Carotenoid content (μg/cm²)', 'Phosphorus content (mg/cm²)', 'Lignin (mg/cm²)', 'Cellulose (mg/cm²)', 
'Fiber (mg/cm²)',
'Anthocyanin content (μg/cm²)',
'NSC (mg/cm²)',
'Magnesium content (mg/cm²)',
'Ca content (mg/cm²)',
'Potassium content (mg/cm²)',
'Boron content (mg/cm²)',
'Copper content (mg/cm²)',
'Sulfur content (mg/cm²)',
'Manganese content (mg/cm²)']


def read_db(file, sp=False, encoding=None):
    db = pd.read_csv(file, encoding=encoding, low_memory=False)
    db.drop(['Unnamed: 0'], axis=1, inplace=True)
    if (sp):
        features = db.loc[:, "400":]
        labels = db.drop(features.columns, axis=1)
        return db, features, labels
    else:
        return db

def meta(num_samp, dict_LC, dict_sc):
    ls = []
    num = []
    j = 1

    for i in num_samp:
        ls = ls + [j for i in range(i)]
        num = num + [i for k in range(i)]
        j = j + 1

    w = pd.DataFrame(ls, columns = ['dataset'])
    w.loc[:, 'numSamples'] = num
    
    w.loc[:, 'LandCover'] = w.loc[:, 'dataset'].map(dict_LC)
    w.loc[:, 'Tool'] = w.loc[:, 'dataset'].map(dict_sc)
    return w        