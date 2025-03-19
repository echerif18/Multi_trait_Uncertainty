from distFuncs import *
from ImgModule import *
from Un_Module import *

import numpy as np
import pandas as pd


import rasterio
import multiprocessing
import time

import faiss

import time 
import os
import json  
import gc


import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
tf.random.set_seed(155)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" ### do not use the GPUs

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# Get the current process ID
current_process = psutil.Process(os.getpid())

# Specify the number of CPU cores you want to use (replace with the desired number)
desired_num_cores = 5

# Limit the CPU cores usage of the current process
limit_cpu_cores(current_process, desired_num_cores)

import argparse

# Create the parser
my_parser = argparse.ArgumentParser(description='Calcul distance for inference')

# Add the arguments
my_parser.add_argument('--routedata',
                       metavar='route',
                       type=str,
                       help='Path to data')

my_parser.add_argument('--modelpath',
                       metavar='modelpath',
                       type=str,
                       help='the path to prediction models')

my_parser.add_argument('--inferpath',
                       metavar='inferpath',
                       type=str,
                       help='the path to inference data')

my_parser.add_argument('--metapath',
                       metavar='metapath',
                       type=str,
                       help='the path to metadata about scene')


my_parser.add_argument('--output_dir',
                       metavar='output_dir',
                       default=str(os.path.join(os.getcwd(),  'distances/')),
                       type=str,
                       help='the path to output_file')


my_parser.add_argument('--neigh',
                       metavar='neigh',
                       type=int,default=50,
                       help='Number of nearest neighbors')

my_parser.add_argument('--lay',
                       metavar='lay',
                       type=int,default= -4,
                       help='Model depth')

my_parser.add_argument('--gpu',
                       metavar='gpu',
                       type=int,default= -1,
                       help='Use of GPU for Faiss')

my_parser.add_argument('--norVec',
                       metavar='norVec',
                       type=bool, default=False,
                       help='L2 Normalization')


my_parser.add_argument('--emb',
                       metavar='emb',
                       type=bool, default=False,
                       help='Embedding space')

my_parser.add_argument('--sp',
                       metavar='sp',
                       type=bool, default=False,
                       help='Feature space')

my_parser.add_argument('--nor',
                       metavar='nor',
                       type=bool, default=False,
                       help='Normilazed distance over training statistics')

my_parser.add_argument('--log',
                       metavar='log',
                       type=bool, default=False,
                       help='Log scale of the distance')

my_parser.add_argument('--sceneText',
                       metavar='sceneText',
                       type=str,default='',
                       help='Label of the scene')


# Execute the parse_args() method
args = my_parser.parse_args()

path_data = args.routedata
dir_model = args.modelpath ## data path
enmap_im_path = args.inferpath
bands_path = args.metapath

neigh = args.neigh ## experiment name 
emb = args.emb
sp = args.sp
lay = args.lay
norVec = args.norVec

nor = args.nor
log = args.log

gpu = args.gpu

sceneText = args.sceneText

output_dir = args.output_dir
gpu = args.gpu



if __name__ == "__main__":
    
    ## Loading Training data ###
    db, fr, lb = read_db(path_data, sp=True)

    ##### Loading test data ###
    src, df, idx_null = image_processing(enmap_im_path, bands_path)
    df_transformed = transform_data(df)

#     #### Calcul distances to all ######################
    data_ts = pd.DataFrame()
    quantiles = [0.5] ### 0.05, 0.5, 0.95
    
    print(emb)
    print(sp)
    # print(quantile)

    start_t = time.perf_counter()
    print("starting calcul ...")

    ######### Embedding space #####
    if(emb):  
        #### Load the model ##
        best_model, scaler_list = load_model(dir_model)
        #### Embedding layer ##
        activation_modelL = Model(inputs = best_model.input, outputs= Flatten()(best_model.layers[lay].output)) ## for effiecient net
        
#         ######### Model predictions ########
#         ### Create the dataset object with a fixed batch size ###
        ds = tf.data.Dataset.from_tensor_slices(df_transformed)        
        ds = prepare(ds, shuffle=False, augment=False)
        
        # Traning ######
        activations_trL = activation_modelL.predict(fr,verbose=1)  

        tf.keras.backend.clear_session()
        gc.collect()
        tf.keras.backend.clear_session()

        # Test scene_ clip ######
        activations_tsL = activation_modelL.predict(ds,verbose=1) #df_transformed
        
        tf.keras.backend.clear_session()
        gc.collect()
        tf.keras.backend.clear_session()

        ###### Normalization of embedding vectors ### this normalization is different that the normalized ditances !!!
        if(norVec):
            ####### Normalization methods ##
            faiss.normalize_L2(activations_trL) ## L2 normalizatio
            faiss.normalize_L2(activations_tsL)

        ######## Mean #########
        dist_ts_EmbEucL, dist_ts_EmbCosL, dist_ts_EmbSpL = dist_allPreds(activations_trL, activations_tsL, neigh, norVec = norVec, avg = True, gpu = gpu)
        
        ########## Merge distances ######
        if (norVec):
            cols = list(data_ts.columns) + ['avg_dist_EmbLEuc', 'avg_dist_EmbLCos', 'avg_dist_EmbLSp']
            data_ts = pd.concat([data_ts, dist_ts_EmbEucL.drop(idx_null, axis=0),dist_ts_EmbCosL.drop(idx_null, axis=0), dist_ts_EmbSpL.drop(idx_null, axis=0)],axis=1) 
        else:
            cols = list(data_ts.columns) + ['avg_dist_EmbLEuc']
            data_ts = pd.concat([data_ts, dist_ts_EmbEucL.drop(idx_null, axis=0)],axis=1) 
        data_ts.columns = cols
    
        if(nor):
            ##  Normalised distance 
            dist_ts_EmbEucL_nor, dist_ts_EmbCosL_nor, dist_ts_EmbSpL_nor = dist_allPreds(activations_trL, activations_tsL, neigh, norVec = norVec, nor = True, avg = True, gpu = gpu) 
        
            if (norVec):
                cols = list(data_ts.columns) + ['avg_dist_EmbLEuc_nor', 'avg_dist_EmbLCos_nor', 'avg_dist_EmbLSp_nor']
                data_ts = pd.concat([data_ts, dist_ts_EmbEucL_nor.drop(idx_null, axis=0), dist_ts_EmbCosL_nor.drop(idx_null, axis=0), dist_ts_EmbSpL_nor.drop(idx_null, axis=0)],axis=1)
            else:
                cols = list(data_ts.columns) + ['avg_dist_EmbLEuc_nor'.format(int(distqu*100))]
                data_ts = pd.concat([data_ts, dist_ts_EmbEucL_nor.drop(idx_null, axis=0)],axis=1) 
            data_ts.columns = cols
        
        if(log):
            ## Log scale of the distance
            dist_ts_EmbEucL_log = np.log(dist_ts_EmbEucL)
            
            if (norVec):
                dist_ts_EmbCosL_log = np.log(dist_ts_EmbCosL)
                dist_ts_EmbSpL_log = np.log(dist_ts_EmbSpL)
                
                cols = list(data_ts.columns) + ['avg_dist_EmbLEuc_log', 'avg_dist_EmbLCos_log', 'avg_dist_EmbLSp_log']
                data_ts = pd.concat([data_ts, dist_ts_EmbEucL_log.drop(idx_null, axis=0),dist_ts_EmbCosL_log.drop(idx_null, axis=0), dist_ts_EmbSpL_log.drop(idx_null, axis=0)],axis=1) 
            else:
                cols = list(data_ts.columns) + ['avg_dist_EmbLEuc_log']
                data_ts = pd.concat([data_ts, dist_ts_EmbEucL_log.drop(idx_null, axis=0)],axis=1) 
            data_ts.columns = cols
   
        ############### quantiles ###
        for distqu in quantiles:
            ## Raw distance ###
            dist_ts_EmbEucL, dist_ts_EmbCosL, dist_ts_EmbSpL = dist_allPreds(activations_trL, activations_tsL, neigh, norVec = norVec, qu = distqu, gpu = gpu)
            
            ########### Merge distances ######
            if (norVec):
                cols = list(data_ts.columns) + ['qu{}_dist_EmbLEuc'.format(int(distqu*100)), 'qu{}_dist_EmbLCos'.format(int(distqu*100)), 'qu{}_dist_EmbLSp'.format(int(distqu*100))]
                data_ts = pd.concat([data_ts, dist_ts_EmbEucL.drop(idx_null, axis=0),dist_ts_EmbCosL.drop(idx_null, axis=0), dist_ts_EmbSpL.drop(idx_null, axis=0)],axis=1) 
            else:
                cols = list(data_ts.columns) + ['qu{}_dist_EmbLEuc'.format(int(distqu*100))]
                data_ts = pd.concat([data_ts, dist_ts_EmbEucL.drop(idx_null, axis=0)],axis=1) 
            data_ts.columns = cols
        
            if(nor):
                ##  Normalised distance 
                dist_ts_EmbEucL_nor, dist_ts_EmbCosL_nor, dist_ts_EmbSpL_nor = dist_allPreds(activations_trL, activations_tsL, neigh, norVec = norVec, nor = True, qu = distqu, gpu = gpu) 
            
                if (norVec):
                    cols = list(data_ts.columns) + ['qu{}_dist_EmbLEuc_nor'.format(int(distqu*100)), 'qu{}_dist_EmbLCos_nor'.format(int(distqu*100)), 'qu{}_dist_EmbLSp_nor'.format(int(distqu*100))]
                    data_ts = pd.concat([data_ts, dist_ts_EmbEucL_nor.drop(idx_null, axis=0),dist_ts_EmbCosL_nor.drop(idx_null, axis=0), dist_ts_EmbSpL_nor.drop(idx_null, axis=0)],axis=1)
                else:
                    cols = list(data_ts.columns) + ['qu{}_dist_EmbLEuc_nor'.format(int(distqu*100))]
                    data_ts = pd.concat([data_ts, dist_ts_EmbEucL_nor.drop(idx_null, axis=0)],axis=1) 
                data_ts.columns = cols
            
            if(log):
                ## Log scale of the distance
                dist_ts_EmbEucL_log = np.log(dist_ts_EmbEucL)
                
                if (norVec):
                    dist_ts_EmbCosL_log = np.log(dist_ts_EmbCosL)
                    dist_ts_EmbSpL_log = np.log(dist_ts_EmbSpL)
                    
                    cols = list(data_ts.columns) + ['qu{}_dist_EmbLEuc_log'.format(int(distqu*100)), 'qu{}_dist_EmbLCos_log'.format(int(distqu*100)), 'qu{}_dist_EmbLSp_log'.format(int(distqu*100))]
                    data_ts = pd.concat([data_ts, dist_ts_EmbEucL_log.drop(idx_null, axis=0),dist_ts_EmbCosL_log.drop(idx_null, axis=0), dist_ts_EmbSpL_log.drop(idx_null, axis=0)],axis=1) 
                else:
                    cols = list(data_ts.columns) + ['qu{}_dist_EmbLEuc_log'.format(int(distqu*100))]
                    data_ts = pd.concat([data_ts, dist_ts_EmbEucL_log.drop(idx_null, axis=0)],axis=1) 
                data_ts.columns = cols
        
        
#     ######### Spectral space #####
    if(sp):
        train = np.ascontiguousarray(fr.values).astype(np.float32)
        test = np.ascontiguousarray(df_transformed.values).astype(np.float32)

        if(norVec):
            faiss.normalize_L2(train)
            faiss.normalize_L2(test)

        ############ Mean ####
        dist_ts_SpEuc, dist_ts_SpCos, dist_ts_SpSp = dist_allPreds(train, test, neigh, norVec = norVec, avg = True, gpu = gpu)
        
        ########### Merge distances ######
        if (norVec):
            cols = list(data_ts.columns) + ['avg_dist_SpEuc', 'avg_dist_SpCos', 'avg_dist_SpSp']
            data_ts = pd.concat([data_ts, dist_ts_SpEuc.drop(idx_null, axis=0),dist_ts_SpCos.drop(idx_null, axis=0), dist_ts_SpSp.drop(idx_null, axis=0)],axis=1) 
        else:
            cols = list(data_ts.columns) + ['avg_dist_SpEuc']
            data_ts = pd.concat([data_ts, dist_ts_SpEuc.drop(idx_null, axis=0)],axis=1) 
        data_ts.columns = cols
        
        if(nor):
            ##  Normalised distance 
            dist_ts_SpEuc_nor, dist_ts_SpCosL_nor, dist_ts_SpSpL_nor = dist_allPreds(train, test, neigh, nor = True, norVec = norVec, avg = True, gpu = gpu)
            
            ########### Merge distances ######
            if (norVec):
                cols = list(data_ts.columns) + ['avg_dist_SpEuc_nor', 'avg_dist_SpCos_nor', 'avg_dist_SpSp_nor']
                data_ts = pd.concat([data_ts, dist_ts_SpEuc_nor.drop(idx_null, axis=0), dist_ts_SpCosL_nor.drop(idx_null, axis=0), dist_ts_SpSpL_nor.drop(idx_null, axis=0)],axis=1) 
            else:
                cols = list(data_ts.columns) + ['avg_dist_SpEuc_nor']
                data_ts = pd.concat([data_ts, dist_ts_SpEuc_nor.drop(idx_null, axis=0)],axis=1) 
            data_ts.columns = cols
        
        if(log):
            ## Log scale 
            dist_ts_SpEuc_log = np.log(dist_ts_SpEuc)
            
            ########### Merge distances ######
            if (norVec):
                dist_ts_SpCos_log = np.log(dist_ts_SpCos)
                dist_ts_SpSp_log = np.log(dist_ts_SpSp)
                
                cols = list(data_ts.columns) + ['avg_dist_SpEuc_log', 'avg_dist_SpCos_log', 'avg_dist_SpSp_log']
                data_ts = pd.concat([data_ts, dist_ts_SpEuc_log.drop(idx_null, axis=0),dist_ts_SpCos_log.drop(idx_null, axis=0), dist_ts_SpSp_log.drop(idx_null, axis=0)],axis=1) 
            else:
                cols = list(data_ts.columns) + ['avg_dist_SpEuc_log']
                data_ts = pd.concat([data_ts, dist_ts_SpEuc_log.drop(idx_null, axis=0)],axis=1) 
            data_ts.columns = cols
            
        ######## Quantile #####
        for distqu in quantiles:
            ## Raw distance ###
            dist_ts_SpEuc, dist_ts_SpCos, dist_ts_SpSp = dist_allPreds(train, test, neigh, norVec = norVec, qu = distqu, gpu = gpu)
            
            ########### Merge distances ######
            if (norVec):
                cols = list(data_ts.columns) + ['qu{}_dist_SpEuc'.format(int(distqu*100)), 'qu{}_dist_SpCos'.format(int(distqu*100)), 'qu{}_dist_SpSp'.format(int(distqu*100))]
                data_ts = pd.concat([data_ts, dist_ts_SpEuc.drop(idx_null, axis=0),dist_ts_SpCos.drop(idx_null, axis=0), dist_ts_SpSp.drop(idx_null, axis=0)],axis=1) 
            else:
                cols = list(data_ts.columns) + ['qu{}_dist_SpEuc'.format(int(distqu*100))]
                data_ts = pd.concat([data_ts, dist_ts_SpEuc.drop(idx_null, axis=0)],axis=1) 
            data_ts.columns = cols
            
            if(nor):
                ##  Normalised distance 
                dist_ts_SpEuc_nor, dist_ts_SpCosL_nor, dist_ts_SpSpL_nor = dist_allPreds(train, test, neigh, nor = True, norVec = norVec, qu=distqu, gpu = gpu)
                
                ########### Merge distances ######
                if (norVec):
                    cols = list(data_ts.columns) + ['qu{}_dist_SpEuc_nor'.format(int(distqu*100)), 'qu{}_dist_SpCos_nor'.format(int(distqu*100)), 'qu{}_dist_SpSp_nor'.format(int(distqu*100))]
                    data_ts = pd.concat([data_ts, dist_ts_SpEuc_nor.drop(idx_null, axis=0), dist_ts_SpCosL_nor.drop(idx_null, axis=0), dist_ts_SpSpL_nor.drop(idx_null, axis=0)],axis=1) 
                else:
                    cols = list(data_ts.columns) + ['qu{}_dist_SpEuc_nor'.format(int(distqu*100))]
                    data_ts = pd.concat([data_ts, dist_ts_SpEuc_nor.drop(idx_null, axis=0)],axis=1) 
                data_ts.columns = cols
            
            if(log):
                ## Log scale 
                dist_ts_SpEuc_log = np.log(dist_ts_SpEuc)
                
                ########### Merge distances ######
                if (norVec):
                    dist_ts_SpCos_log = np.log(dist_ts_SpCos)
                    dist_ts_SpSp_log = np.log(dist_ts_SpSp)
                    
                    cols = list(data_ts.columns) + ['qu{}_dist_SpEuc_log'.format(int(distqu*100)), 'qu{}_dist_SpCos_log'.format(int(distqu*100)), 'qu{}_dist_SpSp_log'.format(int(distqu*100))]
                    data_ts = pd.concat([data_ts, dist_ts_SpEuc_log.drop(idx_null, axis=0),dist_ts_SpCos_log.drop(idx_null, axis=0), dist_ts_SpSp_log.drop(idx_null, axis=0)],axis=1) 
                else:
                    cols = list(data_ts.columns) + ['qu{}_dist_SpEuc_log'.format(int(distqu*100))]
                    data_ts = pd.concat([data_ts, dist_ts_SpEuc_log.drop(idx_null, axis=0)],axis=1) 
                data_ts.columns = cols

    # print(data_ts.shape)
    data_ts.to_csv(os.path.join(output_dir, '{}DistTransPreds_QuDistTrans_{}neighFaiss_{}.csv'.format(len(data_ts.columns), neigh, sceneText)))

    end_t = time.perf_counter()
    total_duration = end_t - start_t
    print(f"Distance calculations took {total_duration:.2f}s total")
    
    # results_exp = {}
    results_exp['UNtime_min'] = total_duration/60
    
    # Serialize data into file:
    json.dump( results_exp, open(os.path.join(output_dir, 'time_UN_Caldis_{}.json'.format(sceneText)), 'w' ) )
