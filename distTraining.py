from distFuncs import *
from model_module_F import *
from data_module_F import Traits
from Un_Module import load_model

from tensorflow.keras.models import model_from_json
from numpy import linalg as LA


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" ### do not use the GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# gpus = tf.config.list_physical_devices('GPU')

# # ####### Fixed use of RAM #####
# if gpus:
#   # Restrict TensorFlow to only allocate 15*1GB of memory on the first GPU
#     try:
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [tf.config.LogicalDeviceConfiguration(memory_limit=1024*4)])
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#             print(e)

import psutil
# Get the current process ID
current_process = psutil.Process(os.getpid())

# Specify the number of CPU cores you want to use (replace with the desired number)
desired_num_cores = 5

# Limit the CPU cores usage of the current process
limit_cpu_cores(current_process, desired_num_cores)


import argparse

# Create the parser
my_parser = argparse.ArgumentParser(description='Calcul distance')

# Add the arguments
my_parser.add_argument('--routedata',
                       metavar='route',
                       type=str,
                       help='Path to data')

my_parser.add_argument('--modelpath',
                       metavar='path',
                       type=str,
                       help='the path to transferability models')


my_parser.add_argument('--neigh',
                       metavar='neigh',
                       type=int,default=5,
                       help='Number of nearest neighbors')

my_parser.add_argument('--lay',
                       metavar='lay',
                       type=int,default= -4,
                       help='Model depth')

my_parser.add_argument('--nor',
                       metavar='nor',
                       type=bool, default=False,
                       help='distance Normalization')

my_parser.add_argument('--norVec',
                       metavar='norVec',
                       type=bool, default=False,
                       help='L2 Normalization')

my_parser.add_argument('--log',
                       metavar='log',
                       type=bool, default=False,
                       help='Log scale distance')

my_parser.add_argument('--emb',
                       metavar='emb',
                       type=bool, default=False,
                       help='Embedding space')

my_parser.add_argument('--sp',
                       metavar='sp',
                       type=bool, default=False,
                       help='Feature space')


my_parser.add_argument('--start',
                       metavar='start',
                       type=int,default= 1,
                       help='dataset index start')


my_parser.add_argument('--end',
                       metavar='end',
                       type=int,default= 48,
                       help='dataset index end')

my_parser.add_argument('--sceneText',
                       metavar='sceneText',
                       type=str,default='',
                       help='Label of the scene')

my_parser.add_argument('--gpu',
                       metavar='gpu',
                       type=int,default= -1,
                       help='Use of GPU for Faiss')

# Execute the parse_args() method
args = my_parser.parse_args()

path_data = args.routedata
path_model = args.modelpath  ## data path

lay = args.lay
neigh = args.neigh ## experiment name 
emb = args.emb
sp = args.sp

nor = args.nor
norVec = args.norVec
log = args.log

start = args.start
end = args.end

sceneText = args.sceneText
gpu = args.gpu

####### create a new directory #######
def create_path(path_gf):
    if not(os.path.exists(path_gf)):
        os.mkdir(path_gf) 


dir_path = os.getcwd()
path_gf = os.path.join(dir_path,  'distances/')
create_path(path_gf)

if __name__ == "__main__":
    
    ###########Training data ##
    db, fr, lb = read_db(path_data, sp=True)
    
    print(emb)
    print(sp)
    # print(quantile)
    ######### Calcul distance to all ####
    min_dist_all = pd.DataFrame()
    quantiles = [0.05, 0.5, 0.95]

    for gp in range(start,end):
        
        print('Iteration {}'.format(gp))

        data_ts = pd.DataFrame()
        data_ts.columns = []

        db_test = db.groupby("dataset").get_group(gp)
        db_train = db.drop(db_test.index, axis=0)

        X_train, train_y, samp_w_tr = data_prep('400', db_train, Traits, w_train = db_train.loc[:,:'Site'], multi= True)
        X_test, test_y, samp_w_ts = data_prep('400', db_test, Traits, w_train = db_test.loc[:,:'Site'], multi= True)

        ##### Distance calcul ##
        ######### version with L2 normalization ####
        if (emb):
            #### Load the model ##
            best_model, scaler_list = load_model(path_model, gp=gp)
            #### Embedding layer ##
            activation_modelL = Model(inputs = best_model.input, outputs= Flatten()(best_model.layers[lay].output)) ## for effiecient net
        
            ######### Model predictions ########
            activations_trL = activation_modelL.predict(X_train,verbose=1)   
            activations_tsL = activation_modelL.predict(X_test,verbose=1) #df_transformed
        
            ###### Normalization of embedding vectors ### this normalization is different that the normalized ditances !!!
            if(norVec):
                ####### Normalization methods ##
                faiss.normalize_L2(activations_trL) ## L2 normalizatio
                faiss.normalize_L2(activations_tsL)
        
            ######## Mean #########
            dist_ts_EmbEucL, dist_ts_EmbCosL, dist_ts_EmbSpL = dist_allPreds(activations_trL, activations_tsL, neigh, norVec = norVec, avg = True, gpu = gpu)
            
            ########### Merge distances ######
            if (norVec):
                cols = list(data_ts.columns) + ['avg_dist_EmbLEuc', 'avg_dist_EmbLCos', 'avg_dist_EmbLSp']
                data_ts = pd.concat([data_ts, dist_ts_EmbEucL,dist_ts_EmbCosL, dist_ts_EmbSpL],axis=1) 
            else:
                cols = list(data_ts.columns) + ['avg_dist_EmbLEuc']
                data_ts = pd.concat([data_ts, dist_ts_EmbEucL],axis=1) 
            data_ts.columns = cols
        
            if(nor):
                ##  Normalised distance 
                dist_ts_EmbEucL_nor, dist_ts_EmbCosL_nor, dist_ts_EmbSpL_nor = dist_allPreds(activations_trL, activations_tsL, neigh, norVec = norVec, nor = True, avg = True, gpu = gpu) 
            
                if (norVec):
                    cols = list(data_ts.columns) + ['avg_dist_EmbLEuc_nor', 'avg_dist_EmbLCos_nor', 'avg_dist_EmbLSp_nor']
                    data_ts = pd.concat([data_ts, dist_ts_EmbEucL_nor, dist_ts_EmbCosL_nor, dist_ts_EmbSpL_nor],axis=1)
                else:
                    cols = list(data_ts.columns) + ['avg_dist_EmbLEuc_nor'.format(int(distqu*100))]
                    data_ts = pd.concat([data_ts, dist_ts_EmbEucL_nor],axis=1) 
                data_ts.columns = cols
            
            if(log):
                ## Log scale of the distance
                dist_ts_EmbEucL_log = np.log(dist_ts_EmbEucL)
                
                if (norVec):
                    dist_ts_EmbCosL_log = np.log(dist_ts_EmbCosL)
                    dist_ts_EmbSpL_log = np.log(dist_ts_EmbSpL)
                    
                    cols = list(data_ts.columns) + ['avg_dist_EmbLEuc_log', 'avg_dist_EmbLCos_log', 'avg_dist_EmbLSp_log']
                    data_ts = pd.concat([data_ts, dist_ts_EmbEucL_log, dist_ts_EmbCosL_log, dist_ts_EmbSpL_log],axis=1) 
                else:
                    cols = list(data_ts.columns) + ['avg_dist_EmbLEuc_log']
                    data_ts = pd.concat([data_ts, dist_ts_EmbEucL_log],axis=1) 
                data_ts.columns = cols
        
            ############### quantiles ###
            for distqu in quantiles:
                ## Raw distance ###
                dist_ts_EmbEucL, dist_ts_EmbCosL, dist_ts_EmbSpL = dist_allPreds(activations_trL, activations_tsL, neigh, norVec = norVec, qu = distqu, gpu = gpu)
                
                ########### Merge distances ######
                if (norVec):
                    cols = list(data_ts.columns) + ['qu{}_dist_EmbLEuc'.format(int(distqu*100)), 'qu{}_dist_EmbLCos'.format(int(distqu*100)), 'qu{}_dist_EmbLSp'.format(int(distqu*100))]
                    data_ts = pd.concat([data_ts, dist_ts_EmbEucL, dist_ts_EmbCosL, dist_ts_EmbSpL],axis=1) 
                else:
                    cols = list(data_ts.columns) + ['qu{}_dist_EmbLEuc'.format(int(distqu*100))]
                    data_ts = pd.concat([data_ts, dist_ts_EmbEucL],axis=1) 
                data_ts.columns = cols
            
                if(nor):
                    ##  Normalised distance 
                    dist_ts_EmbEucL_nor, dist_ts_EmbCosL_nor, dist_ts_EmbSpL_nor = dist_allPreds(activations_trL, activations_tsL, neigh, norVec = norVec, nor = True, qu = distqu, gpu = gpu) 
                
                    if (norVec):
                        cols = list(data_ts.columns) + ['qu{}_dist_EmbLEuc_nor'.format(int(distqu*100)), 'qu{}_dist_EmbLCos_nor'.format(int(distqu*100)), 'qu{}_dist_EmbLSp_nor'.format(int(distqu*100))]
                        data_ts = pd.concat([data_ts, dist_ts_EmbEucL_nor, dist_ts_EmbCosL_nor, dist_ts_EmbSpL_nor],axis=1)
                    else:
                        cols = list(data_ts.columns) + ['qu{}_dist_EmbLEuc_nor'.format(int(distqu*100))]
                        data_ts = pd.concat([data_ts, dist_ts_EmbEucL_nor],axis=1) 
                    data_ts.columns = cols
                
                if(log):
                    ## Log scale of the distance
                    dist_ts_EmbEucL_log = np.log(dist_ts_EmbEucL)
                    
                    if (norVec):
                        dist_ts_EmbCosL_log = np.log(dist_ts_EmbCosL)
                        dist_ts_EmbSpL_log = np.log(dist_ts_EmbSpL)
                        
                        cols = list(data_ts.columns) + ['qu{}_dist_EmbLEuc_log'.format(int(distqu*100)), 'qu{}_dist_EmbLCos_log'.format(int(distqu*100)), 'qu{}_dist_EmbLSp_log'.format(int(distqu*100))]
                        data_ts = pd.concat([data_ts, dist_ts_EmbEucL_log,dist_ts_EmbCosL_log, dist_ts_EmbSpL_log],axis=1) 
                    else:
                        cols = list(data_ts.columns) + ['qu{}_dist_EmbLEuc_log'.format(int(distqu*100))]
                        data_ts = pd.concat([data_ts, dist_ts_EmbEucL_log],axis=1) 
                    data_ts.columns = cols
                
        if (sp):
            train = np.ascontiguousarray(X_train.values).astype(np.float32)
            test = np.ascontiguousarray(X_test.values).astype(np.float32)#/mean_fr
        
            ###### Normalization of embedding vectors ### this normalization is different that the normalized ditances !!!
            if(norVec):
                ####### Normalization methods ##
                faiss.normalize_L2(train) ## L2 normalization
                faiss.normalize_L2(test)
        
            ############ Mean ####
            dist_ts_SpEucL, dist_ts_SpCosL, dist_ts_SpSpL = dist_allPreds(train, test, neigh, norVec = norVec, avg = True, gpu = gpu)
            
            ########### Merge distances ######
            if (norVec):
                cols = list(data_ts.columns) + ['avg_dist_SpEuc', 'avg_dist_SpCos', 'avg_dist_SpSp']
                data_ts = pd.concat([data_ts, dist_ts_SpEucL, dist_ts_SpCosL, dist_ts_SpSpL],axis=1) 
            else:
                cols = list(data_ts.columns) + ['avg_dist_SpEuc']
                data_ts = pd.concat([data_ts, dist_ts_SpEucL],axis=1) 
            data_ts.columns = cols
            
            if(nor):
                ##  Normalised distance 
                dist_ts_SpEucL_nor, dist_ts_SpCosL_nor, dist_ts_SpSpL_nor = dist_allPreds(train, test, neigh, nor = True, norVec = norVec, avg = True, gpu = gpu)
                
                ########### Merge distances ######
                if (norVec):
                    cols = list(data_ts.columns) + ['avg_dist_SpEuc_nor', 'avg_dist_SpCos_nor', 'avg_dist_SpSp_nor']
                    data_ts = pd.concat([data_ts, dist_ts_SpEucL_nor, dist_ts_SpCosL_nor, dist_ts_SpSpL_nor],axis=1) 
                else:
                    cols = list(data_ts.columns) + ['avg_dist_SpEuc_nor']
                    data_ts = pd.concat([data_ts, dist_ts_SpEucL_nor],axis=1) 
                data_ts.columns = cols
            
            if(log):
                ## Log scale 
                dist_ts_SpEuc_log = np.log(dist_ts_SpEucL)
                
                ########### Merge distances ######
                if (norVec):
                    dist_ts_SpCos_log = np.log(dist_ts_SpCosL)
                    dist_ts_SpSp_log = np.log(dist_ts_SpSpL)
                
                    cols = list(data_ts.columns) + ['avg_dist_SpEuc_log', 'avg_dist_SpCos_log', 'avg_dist_SpSp_log']
                    data_ts = pd.concat([data_ts, dist_ts_SpEuc_log, dist_ts_SpCos_log, dist_ts_SpSp_log],axis=1) 
                else:
                    cols = list(data_ts.columns) + ['avg_dist_SpEuc_log']
                    data_ts = pd.concat([data_ts, dist_ts_SpEuc_log],axis=1) 
                data_ts.columns = cols
                
            ######## Quantile #####
            for distqu in quantiles:
                ## Raw distance ###
                dist_ts_SpEucL, dist_ts_SpCosL, dist_ts_SpSpL = dist_allPreds(train, test, neigh, norVec = norVec, qu = distqu, gpu = gpu)
                
                ########### Merge distances ######
                if (norVec):
                    cols = list(data_ts.columns) + ['qu{}_dist_SpEuc'.format(int(distqu*100)), 'qu{}_dist_SpCos'.format(int(distqu*100)), 'qu{}_dist_SpSp'.format(int(distqu*100))]
                    data_ts = pd.concat([data_ts, dist_ts_SpEucL, dist_ts_SpCosL, dist_ts_SpSpL],axis=1)
                else:
                    cols = list(data_ts.columns) + ['qu{}_dist_SpEuc'.format(int(distqu*100))]
                    data_ts = pd.concat([data_ts, dist_ts_SpEucL],axis=1)
                data_ts.columns = cols
                
                if(nor):
                    ##  Normalised distance 
                    dist_ts_SpEucL_nor, dist_ts_SpCosL_nor, dist_ts_SpSpL_nor = dist_allPreds(train, test, neigh, nor = True, norVec = norVec, qu=distqu, gpu = gpu)
                    
                    ########### Merge distances ######
                    if (norVec):
                        cols = list(data_ts.columns) + ['qu{}_dist_SpEuc_nor'.format(int(distqu*100)), 'qu{}_dist_SpCos_nor'.format(int(distqu*100)), 'qu{}_dist_SpSp_nor'.format(int(distqu*100))]
                        data_ts = pd.concat([data_ts, dist_ts_SpEucL_nor, dist_ts_SpCosL_nor, dist_ts_SpSpL_nor],axis=1)
                    else:
                        cols = list(data_ts.columns) + ['qu{}_dist_SpEuc_nor'.format(int(distqu*100))]
                        data_ts = pd.concat([data_ts, dist_ts_SpEucL_nor],axis=1)
                    data_ts.columns = cols
                
                if(log):
                    ## Log scale 
                    dist_ts_SpEuc_log = np.log(dist_ts_SpEucL)
                    
                    ########### Merge distances ######
                    if (norVec):
                        dist_ts_SpCos_log = np.log(dist_ts_SpCosL)
                        dist_ts_SpSp_log = np.log(dist_ts_SpSpL)
                    
                        cols = list(data_ts.columns) + ['qu{}_dist_SpEuc_log'.format(int(distqu*100)), 'qu{}_dist_SpCos_log'.format(int(distqu*100)), 'qu{}_dist_SpSp_log'.format(int(distqu*100))]
                        data_ts = pd.concat([data_ts, dist_ts_SpEuc_log, dist_ts_SpCos_log, dist_ts_SpSp_log],axis=1)
                    else:
                        cols = list(data_ts.columns) + ['qu{}_dist_SpEuc_log'.format(int(distqu*100))]
                        data_ts = pd.concat([data_ts, dist_ts_SpEuc_log],axis=1)
                    data_ts.columns = cols

        min_dist_all = pd.concat([min_dist_all, data_ts],axis=0)
        min_dist_all.columns = data_ts.columns
        
    print(min_dist_all.shape)
    min_dist_all.to_csv(os.path.join(path_gf, '{}DistTransPreds_QuDistTrans_{}neighFaiss_Training{}datasets{}.csv'.format(len(data_ts.columns), neigh, end, sceneText)))