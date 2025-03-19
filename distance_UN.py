from feature_module_F import *
from data_module_F import *

# from Vis_module import falseColVis, Vegindex_Vis
from Un_Module import load_model

import tensorflow as tf

import time 
import os
import json


from pickle import dump,load
import statsmodels.api as sm

import argparse

# # Create the parser
my_parser = argparse.ArgumentParser(description='Calcul distance for inference')

# Add the arguments
# my_parser.add_argument('--predictors_path',
#                        metavar='predictors_path',
#                        type=str,
#                        help='Best predictors ')

my_parser.add_argument('--modelpath',
                       metavar='modelpath',
                       type=str,
                       help='the path to UN models')

my_parser.add_argument('--path_dist_ts',
                       metavar='path_dist_ts',
                       type=str,
                       help='the path to inference data')

my_parser.add_argument('--sceneText',
                       metavar='sceneText',
                       type=str,default='',
                       help='Label of the scene')

my_parser.add_argument('--output_dir',
                       metavar='output_dir',
                       default=str(os.path.join(os.getcwd(),  'un_predictions/')),
                       type=str,
                       help='the path to output_file')


# # Execute the parse_args() method
args = my_parser.parse_args()
dir_path = args.modelpath ## data path
# predictors_path = args.predictors_path

path_dist_ts = args.path_dist_ts
output_dir = args.output_dir

sceneText = args.sceneText


# Read data from file:
subset = ['qu50_dist_EmbLCos_nor',
'qu50_dist_SpCos_nor'] #list(json.load( open(predictors_path) )['best_preds'])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" ### do not use the GPUs

if __name__ == "__main__":
    ######### Model predictions ########
    data_ts = pd.read_csv(path_dist_ts).drop(['Unnamed: 0'], axis=1)
    
    start_t = time.perf_counter()
    
    print("starting predictions")
    un_pred_ts = []
    
    for tr in range(len(Traits)):
        scaler_x_ts = load(open(os.path.join(dir_path, 'x_scaler_{}.pkl'.format(tr)), 'rb'))
        scaler_y_ts = load(open(os.path.join(dir_path, 'y_scaler_{}.pkl'.format(tr)), 'rb'))
        model_un = load(open(os.path.join(dir_path, 'model_{}.pkl'.format(tr)), 'rb'))
        
        xdata = data_ts.loc[:,subset].astype('float64')
        if(scaler_x_ts is not None):
            xdata = pd.DataFrame(scaler_x_ts.transform(xdata.values.reshape(-1, xdata.shape[1])),columns = xdata.columns) #float64
        else:
            xdata = pd.DataFrame(xdata.values.reshape(-1, xdata.shape[1]),columns = xdata.columns) #float64
    
        X_new = sm.add_constant(xdata)
        
        if(scaler_y_ts is not None):
            y_pred_ts = pd.DataFrame(scaler_y_ts.inverse_transform(model_un.predict(X_new).values.reshape(-1, 1)), index = data_ts.index)
        else:
            y_pred_ts = pd.DataFrame(model_un.predict(X_new).values.reshape(-1, 1), index = data_ts.index)
                    
        un_pred_ts.append(y_pred_ts.values)
    
    
    end_t = time.perf_counter()
    total_duration = end_t - start_t
    print(f"etl took {total_duration:.2f}s total")
    
    results_exp = {}
    results_exp['UNtime_min'] = total_duration/60
    
    # Serialize data into file:
    json.dump(results_exp, open(os.path.join(output_dir, 'Executiontime_UnDist_{}.json'.format(sceneText)), 'w' ) )
    
    # Optionally, calculate uncertainty metrics (e.g., standard deviation)
    uncertainty = pd.DataFrame(np.mean(un_pred_ts, axis=2).T, columns=Traits)
    uncertainty.to_csv(os.path.join(output_dir, 'UN_dist_{}.csv'.format(sceneText)))