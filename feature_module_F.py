import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import savgol_filter

import rasterio
import multiprocessing
import time

### Apply savgol filter for a wavelength filter, 
def filter_segment(features_noWtab, order=1,der= False):
    #features_noWtab: Segment of the signal
    #order: Order of the savgol filter
    #der: If with first derivative
    
#     part1 = features_noWtab.loc[:,indx]
    part1 = features_noWtab.copy()
    if (der):
        fr1 = savgol_filter(part1, 65, 1,deriv=1)
    else:
        fr1 = savgol_filter(part1, 65, order)
    fr1 = pd.DataFrame(data=fr1, columns=part1.columns)
    return fr1

###### transformation methods #######
def feature_preparation(features, inval = [1351,1431, 1801, 2051], frmax=2451, order=1,der= False):
    # features: The original reflectance signal
    #order: Order of the savgol filter
    #der: If with first derivative
    
    other = features.copy()
    other.columns = other.columns.astype('int')
    other[other<0] = np.nan
    
    #####Substitute high values with the mean of neighbour values
    other[other>1] = np.nan
    other = (other.ffill() + other.bfill())/2
    other = other.interpolate(method='linear',limit_area=None, axis=1, limit_direction='both')
    
    wt_ab = [i for i in range(inval[0],inval[1])]+[i for i in range(inval[2],inval[3])]+[i for i in range(2451,2501)] 

    features_Wtab = other.loc[:,wt_ab]
    features_noWtab=other.drop(wt_ab,axis=1)
    
    fr1 = filter_segment(features_noWtab.loc[:,:inval[0]-1], order = order, der = der)
    fr2 = filter_segment(features_noWtab.loc[:,inval[1]:inval[2]-1], order = order,der = der)
    fr3 = filter_segment(features_noWtab.loc[:,inval[3]:frmax], order = order,der = der)    
    
    
    inter = pd.concat([fr1,fr2,fr3], axis=1, join='inner')
    inter[inter<0]=0
    
    return inter