import os

import pandas as pd
import numpy as np
import math
from pickle import dump,load

from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score 
from sklearn.preprocessing import PowerTransformer

from model_builder import *
# from Resnet1D_builder import *
from EfficientNet1D_builder import *
# from data_module_F import Traits

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger

# from tensorflow.keras.models import model_from_json
from tensorflow.keras import callbacks

# import keras_tuner
import tensorflow_addons as tfa
from tensorflow_addons.metrics import RSquare

            
batch_size = 64 #128
AUTOTUNE = tf.data.AUTOTUNE

####### create a new directory #######
def create_path(path_gf):
    if not(os.path.exists(path_gf)):
        os.mkdir(path_gf) 

        
################## Data #####################       


def data_prep(minl, gap_fil, Traits, i = 20, w_train=None, multi=False): #i = len(Traits)-1
    
    ##########Testing/validation data preparation (only for the last added trait)#######
    if (multi):
        train_x = gap_fil.loc[:, minl:]
        train_y = gap_fil.loc[train_x.index, Traits[:i + 1]]
    else:
        train_x = gap_fil.loc[gap_fil[gap_fil[Traits[i]].notnull()].index, minl:]
        train_y = gap_fil.loc[train_x.index, Traits[i:i + 1]]
    
    if(w_train is not None):
        samp_w_tr = samp_w(w_train, train_x)  # >>>>>>samples weights calculation
        return train_x, train_y, samp_w_tr
    else:
        return train_x, train_y

######## calculate sample weights from meta data #########
def samp_w(w_train, train_x):
    wstr = 100 - 100 * (w_train.loc[train_x.index, :].groupby(['dataset'])['numSamples'].count() /
                        w_train.loc[train_x.index, :].shape[0])
    samp_w_tr = np.array(w_train.loc[train_x.index, 'dataset'].map(dict(wstr)), dtype='float')
    return samp_w_tr

######## create tensor data sets for training and test ##########
def dataset(train_x, train_y, samp_w_tr, scaler_list, Traits, shuffle=False, augment=False):
    if (samp_w_tr is None):
        ds = tf.data.Dataset.from_tensor_slices((train_x,
            scaler_list.transform(np.array(train_y)),
                                                 None))
    else:
        if ( (samp_w_tr.sum().sum() !=0)):
            ds = tf.data.Dataset.from_tensor_slices((train_x, scaler_list.transform(np.array(train_y)),
                                                     samp_w_tr))
        else:
            ds = tf.data.Dataset.from_tensor_slices((train_x, scaler_list.transform(np.array(train_y)),
                                                     None))        
    ds = prepare(ds, shuffle, augment)
    return ds

def prepare(ds, shuffle=False, augment=False, batch_size= 64):
    #### Preparation of the dataset (spectra, labels and weights) in 32 batch with shuffeling and augmentation, the precesses are repreated 2 times ###

    if shuffle:
        ds = ds.shuffle(len(ds), reshuffle_each_iteration=True)
    
    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y,z: (data_augmentation(x, y,z)), num_parallel_calls=AUTOTUNE)
        
    # Batch all datasets.
    ds = ds.batch(batch_size)
    
    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE).repeat(2)


######## save the scaler ######
def save_scaler(train_y, save=False, dir_n=None, k=None):
    scaler = PowerTransformer(method='box-cox').fit(np.array(train_y))  
    if(save):
        if not (os.path.exists(dir_n)):
            os.mkdir(dir_n)
        dump(scaler, open(dir_n + '/scaler_{}.pkl'.format(k), 'wb'))  # save the scaler

    return scaler        
        
        
######### up-sampling of the data based on the meta data ##########        
def balanceData(db_train, w_train, Traits, random_state=300):
    ### The maximum number of samples within a dataset ##
    mx = pd.concat([w_train.reset_index(drop=True),db_train.reset_index(drop=True)], axis=1).groupby('dataset')[Traits].count().max().max()
    fill = pd.concat([w_train.reset_index(drop=True),db_train.reset_index(drop=True)], axis=1).groupby('dataset').sample(n=mx,random_state=random_state,replace=True).reset_index(drop=True)
    return fill       


############## Data augmentation ###########
#This method was adopted from https://github.com/EBjerrum/Deep-Chemometrics/blob/master/ChemUtils.py with some modifications 
def dataaugment(x, betashift = 0.05, slopeshift = 0.05,multishift = 0.05, kind=None):
    #Shift of baseline
    #calculate arrays
    beta = np.random.random(size=(1))*2*betashift-betashift
    slope = np.random.random(size=(1))*2*slopeshift-slopeshift + 1

    #Calculate relative position
    if (len(x.shape)==1):
        axis = np.array(range(x.shape[0]))/float(x.shape[0])
    else:
        axis = np.array(range(x.shape[1]))/float(x.shape[1])

    #Calculate offset to be added
    offset = slope*(axis) + beta - axis - slope/2. + 0.5

    #Multiplicative
    multi = np.random.random(size=(1))*2*multishift-multishift + 1
    if (kind =='offset'):
        return x + offset
    elif (kind == 'multi'):
        return multi*x
    else:
        return multi*x + offset       

def data_augmentation(x,y,z):
    data_std = tf.math.reduce_std(x,0)
    
    if  tf.random.uniform([], 0, 1) < 0.15:
        x = dataaugment(x, betashift = data_std, slopeshift = data_std, multishift = data_std)
   
    return x,y,z



def model_definition(input_shape, output_shape,
                 num_Layer = None,
                 kernelSize= None,
                 f= None,
                 ac= None,
                 dropR= None,
                 lr= 0.000005,
                 units= None, lamb = 0.01, num_dense = 1, loss = HubercustomLoss(threshold=1), dir_n = None, max_trials = 20, kind=None):
    if (kind=='opt'):
        ob=keras_tuner.Objective("val_root_mean_squared_error", direction="min")
            
        model = CustomTuner(
            hypermodel= model_opt(input_shape, output_shape),
            objective=ob,
            max_trials= max_trials,
            directory = dir_n,
            project_name="OpTrials"
        )
    
    elif (kind=='resnet'):
        model = ResNet50(input_shape = input_shape, output_shape=output_shape)
    elif (kind=='efficientnet'):
        model = EfficientNet_1dB0(input_shape = input_shape, output_shape=output_shape)
    else:
        model = create_model(input_shape, output_shape, 1 ,[51,51,51],[64,64,3],'relu',0.1,256)

    optimizer = Adam(learning_rate = lr, clipnorm=1.0)    
    model.compile(
        optimizer, loss=loss, metrics=[MaskedRmse(),MaskedR2()] )
        
    return model


##### Save models aarchitecture and trained weights #######
def save_model(best_model, path_trial, path_best, path_w):
    model_json = best_model.to_json()
    with open(path_trial, "w") as json_file:
        json_file.write(model_json)  

    best_model.load_weights(path_best)
    best_model.save_weights(path_w)

###### Fill the data set from previoisly trained model #########
def fill_gp(gap_fil, best_model, scaler_list, Traits, j):
    pds = scaler_list.inverse_transform(best_model.predict(gap_fil.loc[gap_fil[gap_fil[Traits[j]].isna()].index, "400":]))
    fp = pd.DataFrame(pds, index=gap_fil[gap_fil[Traits[j]].isna()].index)[j]

    gap_fil.loc[gap_fil[gap_fil[Traits[j]].isna()].index, Traits[j]] = gap_fil.loc[gap_fil[gap_fil[Traits[j]].isna()].index, Traits[j]].fillna(fp)
    return gap_fil


    
#### Model evaluation function ##########
def all_scores(test_tr,Traits,obs_pf, pred_df,samp_w_ts=None, method = None, save =False, dir_n = None):
    r2_tab = []
    RMSE_tab = []
    nrmse_tab = []
    mae_tab = []
    b_tab = []

    for j in test_tr:

        f = pred_df[j+ ' Predictions'].reset_index(drop=True) # + ' Predictions'
        y = obs_pf[j].reset_index(drop=True)

        idx = np.union1d(f[f.isna()].index,y[y.isna()].index)

        f.drop(idx, axis = 0, inplace=True)
        y.drop(idx, axis = 0, inplace=True)
        
        
        if (y.notnull().sum()):
            if (samp_w_ts is not None):
                we = pd.DataFrame(samp_w_ts).loc[f.index,:]
            else:
                we = None

            if (we is not None) and (we.sum().sum() !=0):
                r2_tab.append(r2_score(y,f,sample_weight= we))

                RMSE=math.sqrt(mean_squared_error(y,f,sample_weight= we))
                RMSE_tab.append(RMSE)
                nrmse_tab.append((RMSE*100)/(np.nanquantile(np.array(y),0.99) - np.nanquantile(np.array(y),0.01)))

                mae_tab.append(mean_absolute_error(y,f,sample_weight= we))

                bias=np.sum(np.array(y)-np.array(f))/len(f)
                b_tab.append(bias)
            else:
                r2_tab.append(r2_score(y,f))

                RMSE=math.sqrt(mean_squared_error(y,f))
                RMSE_tab.append(RMSE)
                nrmse_tab.append((RMSE*100)/(np.nanquantile(np.array(y),0.99) - np.nanquantile(np.array(y),0.01)))

                mae_tab.append(mean_absolute_error(y,f))

                bias=np.sum(np.array(y)-np.array(f))/len(f)
                b_tab.append(bias)
        else:
            r2_tab.append(np.nan)
            RMSE_tab.append(np.nan)
            nrmse_tab.append(np.nan)
            mae_tab.append(np.nan)
            b_tab.append(np.nan)
            pass        

    test = pd.DataFrame([r2_tab, RMSE_tab, nrmse_tab,mae_tab,b_tab], columns= test_tr[:len(test_tr)], index=['r2_score','RMSE','nRMSE (%)','MAE','Bias'])
    if(save):
        test.to_csv(dir_n + 'scores_all_{}.csv'.format(method))
    return test
