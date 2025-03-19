import numpy as np
import pandas as pd
import rasterio
import multiprocessing
import time


from data_module_F import *
from feature_module_F import filter_segment, feature_preparation


# ######## Load imagery methods + Vis ######
###########
def image_processing(enmap_im_path, bands_path):
    bands = pd.read_csv((bands_path))['bands'].astype(float)
    src = rasterio.open(enmap_im_path)
    array = src.read()
    sp_px = np.stack([array[i].reshape(-1,1) for i in range(array.shape[0])],axis=0)
    sp_px = np.swapaxes(sp_px.mean(axis=2),0,1) #transpose
    
    assert (sp_px.shape[1] == bands.shape[0]), "The number of bands is not correct. Check the number of spectral bands in the imagery!"
    
    df = pd.DataFrame(sp_px, columns = bands.to_list())
    df[df< df.quantile(0.01).min()+10] = np.nan ## eliminate corrupted pixels and replace with nan
    
    idx_null = df[df.T.isna().all()].index
    return src, df, idx_null

def process_dataframe(veg_spec):
    start_t = time.perf_counter()
    veg_reindex = veg_spec.reindex(columns = sorted(veg_spec.columns.tolist() + [i for i in range(400,2501) if i not in veg_spec.columns.tolist()]))#.interpolate(method='linear',limit_area=None, axis=1, limit_direction='both')

    veg_reindex = veg_reindex/10000
    veg_reindex.columns = veg_reindex.columns.astype(int)
    
    # inter = veg_reindex.loc[:,~veg_reindex.columns.duplicated()] ## remove column duplicates 
    inter = feature_preparation(veg_reindex, inval = [1251, 1530, 1801, 2051], order=1) #>>> 1522 ##, inval = [1251, 1530, 1801, 2051]>> 1522, default>> 1721
    # inter = feature_preparation(veg_reindex, order=1) ## >>> 1721 , inval = [1251,1530, 1801, 2051]>> 1522, default>> 1721
    ############ Remove duplicated columns #######
    inter = inter.loc[:,~inter.columns.duplicated()] ## remove column duplicates 
    
    return inter.loc[:,400:]

####### Prepare fro multi-processing ##
def transform_data(df):
    # Define the number of CPUs to use
    num_cpus = multiprocessing.cpu_count()
    # Create a multiprocessing pool with the specified number of CPUs
    pool = multiprocessing.Pool(num_cpus)
    # Split the DataFrame into chunks to be processed in parallel
    df_chunks = [chunk for chunk in np.array_split(df, num_cpus)]

    start_t = time.perf_counter()

    print("starting processing")
    with multiprocessing.Pool(num_cpus) as pool:
        results = pool.map(process_dataframe, df_chunks)
        pool.close()
        pool.join()

    end_t = time.perf_counter()
    total_duration = end_t - start_t
    print(f"Image transformation took {total_duration:.2f}s total") 
    
    df_transformed = pd.concat(results).reset_index(drop=True)
    
    return df_transformed