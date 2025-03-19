from feature_module_F import *
from data_module_F import *
from Un_Module import load_model

import tensorflow as tf

import time 
import os
import json


import psutil
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


# Get the current process ID
current_process = psutil.Process(os.getpid())

# Specify the number of CPU cores you want to use (replace with the desired number)
desired_num_cores = 6

# Limit the CPU cores usage of the current process
limit_cpu_cores(current_process, desired_num_cores)



import argparse


# # Create the parser
my_parser = argparse.ArgumentParser(description='Calcul distance for inference')

# Add the arguments
my_parser.add_argument('--inferpath',
                       metavar='inferpath',
                       default=None,
                       type=str,
                       help='the path to inference data')

my_parser.add_argument('--metapath',
                       metavar='metapath',
                       default=None,
                       type=str,
                       help='the path to metadata about scene')

my_parser.add_argument('--fr_path',
                       metavar='fr_path',
                       default=None,
                       type=str,
                       help='the path to inference data processed')

my_parser.add_argument('--modelpath',
                       metavar='modelpath',
                       type=str,
                       help='the path to UN models')

my_parser.add_argument('--sceneText',
                       metavar='sceneText',
                       type=str,default='',
                       help='Label of the scene')

my_parser.add_argument('--output_dir',
                       metavar='output_dir',
                       default=str(os.path.join(os.getcwd(),  'other_models/')),
                       type=str,
                       help='the path to output_file')

my_parser.add_argument('--start',
                       metavar='start',
                       type=int,default=1,
                       help='start transferability model')

my_parser.add_argument('--end',
                       metavar='end',
                       type=int,default=48,
                       help='end transferability model')


# # Execute the parse_args() method
args = my_parser.parse_args()

path_model = args.modelpath ## data path
enmap_im_path = args.inferpath
bands_path = args.metapath

output_dir = args.output_dir
sceneText = args.sceneText

fr_path = args.fr_path

start = args.start
end = args.end


######## GPU RAM memory if GPU available ##########
os.environ["CUDA_VISIBLE_DEVICES"]="0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


if __name__ == "__main__":
    ##### Loading test data ###
    start_t = time.perf_counter()
    
    if (fr_path is None):
    
        src, df, idx_null = image_processing(enmap_im_path, bands_path)
        ####### Prepare fro multi-processing ##
        df_transformed = transform_data(df)
    else:
        df_transformed = pd.read_csv(fr_path).drop(['Unnamed: 0'], axis=1).loc[:, "400":]
    
    end_t = time.perf_counter()
    total_duration = end_t - start_t
    
    results_exp = {}
    results_exp['img_min'] = total_duration/60
    
    
    #########
    predictions = []
    start_t = time.perf_counter()
    print("starting predictions")
    
    for gp in range(start,end):
        
        print('Iteration {}'.format(gp))
        #### Load the model ##
        best_model, scaler_list = load_model(path_model, gp=gp)
    
        ######### Model predictions ########
        tf_preds = scaler_list.inverse_transform(best_model.predict(df_transformed, verbose=1, batch_size=128)) #df_transformed
        preds = pd.DataFrame(tf_preds, columns=Traits)
    
        # preds.loc[idx_null] = np.nan
        predictions.append(preds)
    
    end_t = time.perf_counter()
    total_duration = end_t - start_t
    print(f"ensemble predictions took {total_duration:.2f}s total")
    
    # results_exp = {}
    results_exp['UNtime_min'] = total_duration/60
    
    # Serialize data into file:
    # json.dump( results_exp, open( "/net/scratch/echerif/gitcodetest/uncertainty/other_models/time_preds_ens_sc{}_Neon.json".format(sc), 'w' ) )
    json.dump( results_exp, open(os.path.join(output_dir, 'time_UN_mean_ens_{}.json'.format(sceneText)), 'w' ) )
    
    # Calculate the mean prediction
    mean_prediction_ensemble = pd.DataFrame(np.mean(predictions, axis=0), columns=Traits)
    # Optionally, calculate uncertainty metrics (e.g., standard deviation)
    uncertainty_ensemble = np.std(predictions, axis=0)
    
    # mean_prediction_ensemble.to_csv('/net/scratch/echerif/gitcodetest/uncertainty/other_models/mean_predictions_ens_sc{}_Neon.csv'.format(sc))
    mean_prediction_ensemble.to_csv(os.path.join(output_dir, 'UN_mean_ens_{}.csv'.format(sceneText)))
    
    # pd.DataFrame(uncertainty_ensemble).to_csv('/net/scratch/echerif/gitcodetest/uncertainty/other_models/un_std_ens_sc{}_Neon.csv'.format(sc))
    pd.DataFrame(uncertainty_ensemble).to_csv(os.path.join(output_dir, 'UN_std_ens_{}.csv'.format(sceneText)))