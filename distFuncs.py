import numpy as np
import pandas as pd
import time
import faiss
import os

from tensorflow import keras
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from joblib import Parallel, delayed
import psutil # type: ignore
import random


def limit_cpu_cores(process, num_cores):
    # Get the total number of available CPU cores
    total_cores = psutil.cpu_count()

    # Ensure num_cores does not exceed the total number of cores
    num_cores = min(num_cores, total_cores)
    rd = random.randint(1,total_cores
                       )
    # Get the process object
    process = psutil.Process(process.pid)

    # Set the CPU affinity to the first num_cores cores
    process.cpu_affinity(list(range(rd, rd + num_cores)))
    
def read_db(file, sp=False, encoding=None, low_memory=False):
    db = pd.read_csv(file, encoding=encoding)
    db.drop(['Unnamed: 0'], axis=1, inplace=True)
    if (sp):
        features = db.loc[:, "400":]
        labels = db.drop(features.columns, axis=1)
        return db, features, labels
    else:
        return db


# def dis_func(activations_tr, activations_ts = None, kind = 'Euc', neigh=5, gpu=-1):
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
#     res = faiss.StandardGpuResources()
#     deviceId = 0
#     co = faiss.GpuClonerOptions()
#     # # co.useFloat16 = use_float16
#     d = activations_tr.shape[1]

#     if (kind == 'Euc'):
#         index = faiss.IndexFlatL2(d)
#     else:
#         index = faiss.IndexFlatIP(d)
        
#     index = faiss.index_cpu_to_gpu(res, deviceId, index, co)
#     index.add(activations_tr)
    
#     if (activations_ts is not None):
#         distances_tr, indices_tr = index.search(activations_ts, neigh)
#     else:
#         distances_tr, indices_tr = index.search(activations_tr, neigh)
#     return distances_tr, indices_tr


def dis_func(activations_tr, activations_ts=None, kind='Euc', neigh=5, gpu=-1):
    # Ensure FAISS detects GPUs
    num_gpus = faiss.get_num_gpus()
    assert num_gpus > 0, "FAISS is not detecting any GPU!"

    # Assign device ID properly
    deviceId = 0 if gpu == -1 else gpu
    
    # Initialize GPU resources with limited memory
    res = faiss.StandardGpuResources()
    res.setTempMemory(256 * 1024 * 1024)  # 256MB

    d = activations_tr.shape[1]  # Dimension of input data

    # Choose metric
    index = faiss.IndexFlatL2(d) if kind == 'Euc' else faiss.IndexFlatIP(d)

    # Move index to GPU
    co = faiss.GpuClonerOptions()
    gpu_index = faiss.index_cpu_to_gpu(res, deviceId, index, co)
    
    # Add training data
    gpu_index.add(activations_tr)

    # Perform search
    query_data = activations_tr if activations_ts is None else activations_ts
    distances_tr, indices_tr = gpu_index.search(query_data, neigh)

    return distances_tr, indices_tr



############ Last with paralleism ###
# Step 2: Calculate distances using scikit-learn in parallel
def calculate_distances(i,activations_trL, indices_ts, activations_tsL=None, kind='Euc'):
    indices_row = indices_ts[i]
    
    if(activations_tsL is not None):
        if(kind=='Euc'):
            distances_row = euclidean_distances([activations_tsL[i]], activations_trL[indices_row])
        else:
            distances_row = cosine_distances([activations_tsL[i]], activations_trL[indices_row])
    else:
        if(kind=='Euc'):
            distances_row = euclidean_distances([activations_trL[i]], activations_trL[indices_row])
        else:
            distances_row = cosine_distances([activations_trL[i]], activations_trL[indices_row])
    return distances_row.flatten()


##### new with Faiss #####
def dist_allPreds(activations_trL, activations_tsL, neigh, nor=False, norVec=True, qu=0.5, gpu = -1, avg = False):
    dist_ts_EmbCosLqu = None
    dist_ts_EmbSpLqu = None
    
    print("Start Emb/feature similarity measure ...")
    start_t = time.perf_counter()
    
    distances_ts_embeucL, _ = dis_func(activations_trL, activations_ts = activations_tsL, kind = 'Euc', neigh = neigh, gpu=gpu)
    distances_ts_embeucL = np.sqrt(distances_ts_embeucL) ########## new !!!!
    
    if(qu is not None):
        dist_ts_EmbEucLqu = pd.DataFrame(np.quantile(distances_ts_embeucL, qu, axis=1)) ##### Conversion of distances #####
    if(avg):
        dist_ts_EmbEucLqu = pd.DataFrame(np.mean(distances_ts_embeucL, axis=1)) ##### Conversion of distances #####
    
    if(norVec):
        distances_ts_embcosL, _ = dis_func(activations_trL, activations_ts = activations_tsL, kind = 'Cos', neigh = neigh, gpu = gpu)
        
        if(qu is not None):
            dist_ts_EmbCosLqu = pd.DataFrame(1- np.quantile(distances_ts_embcosL, qu, axis=1)) ## convert to cos distance
            dist_ts_EmbSpLqu = pd.DataFrame(np.arccos(1- np.quantile(distances_ts_embcosL, qu, axis=1))) ##### Conversion of distances 
            
        if(avg):
            dist_ts_EmbCosLqu = pd.DataFrame(1- np.mean(distances_ts_embcosL, axis=1)) ## convert to cos distance
            dist_ts_EmbSpLqu = pd.DataFrame(np.arccos(1- np.mean(distances_ts_embcosL, axis=1))) ##### Conversion of distances #####

    
    if (nor): 
        dist_ts_EmbCosLqu_nor = None
        dist_ts_EmbSpLqu_nor = None
        
        distances_tr_embeucL, _ = dis_func(activations_trL, kind = 'Euc', neigh = neigh, gpu = gpu) ## distance WITHIN TRAINING
        distances_tr_embeucL = np.sqrt(distances_tr_embeucL) ###new!!!
        
        dist_ts_EmbEucLqu_nor = dist_ts_EmbEucLqu/(np.mean(distances_tr_embeucL))
        
        if(norVec):
            distances_tr_embcosL, _ = dis_func(activations_trL, kind = 'Cos', neigh = neigh, gpu = gpu) ## distance WITHIN TRAINING
            dist_ts_EmbCosLqu_nor = pd.DataFrame(dist_ts_EmbCosLqu/(1-distances_tr_embcosL.mean())) ## convert to cos distance
            dist_ts_EmbSpLqu_nor = pd.DataFrame(dist_ts_EmbSpLqu/(np.arccos(1- distances_tr_embcosL.mean())))
        
        end_t = time.perf_counter()
        total_duration = end_t - start_t
        print(f"Emb/feature similarity measure took {total_duration:.2f}s total")
        
        return dist_ts_EmbEucLqu_nor, dist_ts_EmbCosLqu_nor, dist_ts_EmbSpLqu_nor
    else:
        end_t = time.perf_counter()
        total_duration = end_t - start_t
        print(f"Emb/feature similarity measure took {total_duration:.2f}s total")
        return dist_ts_EmbEucLqu, dist_ts_EmbCosLqu, dist_ts_EmbSpLqu
