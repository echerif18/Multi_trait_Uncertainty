from data_module_F import *
from feature_module_F import *
from model_module_F import *
from tensorflow.keras.models import model_from_json


import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices ## patsy 0.5.3

from sklearn.preprocessing import RobustScaler,PowerTransformer, QuantileTransformer, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error,mean_pinball_loss, d2_pinball_score, r2_score, explained_variance_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import QuantileRegressor

from pickle import dump,load


def load_model(dir_data, gp = None):
    if(gp is not None):
        json_file = open(dir_data + 'Model_db{}.json'.format(gp), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        best_model = model_from_json(loaded_model_json)

        scaler_list = load(open(dir_data + 'scaler_db{}.pkl'.format(gp), 'rb'))

        # load weights into new model
        best_model.load_weights(dir_data + 'Trial_db{}_weights.h5'.format(gp))
    else:
        json_file = open(dir_data + 'Model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        best_model = model_from_json(loaded_model_json)

        scaler_list = load(open(dir_data+ 'scaler_global.pkl', 'rb'))

        # load weights into new model
        best_model.load_weights(dir_data+ 'Trial_weights.h5')
    
    return best_model, scaler_list


def prepare(ds, shuffle=False, augment=False):
    #### Preparation of the dataset (spectra, labels and weights) in 32 batch with shuffeling and augmentation, the precesses are repreated 2 times ###

    if shuffle:
        ds = ds.shuffle(len(ds), reshuffle_each_iteration=True)
    
    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y,z: (data_augmentation(x, y,z)), num_parallel_calls=AUTOTUNE)
        
    # Batch all datasets.
    ds = ds.batch(batch_size)
    
    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE).repeat(1)



def model_Un_training(dis_err_tr, subset, ls_tr, x_trans=False, y_trans=False, save=False, path_model_tr=None):
    for tr in range(len(ls_tr)):
    
        xdata = dis_err_tr.loc[:,subset] 
        ydata = dis_err_tr.loc[:,ls_tr[tr]]
        
        idx = ydata[ydata.isna()].index
        
        xdata.drop(idx, axis = 0, inplace=True)
        ydata.drop(idx, axis = 0, inplace=True)

        ####### weights ###
        weights = np.ones(len(ydata))
        
        hist, bin_edges = np.histogram(ydata, bins=5)
        weights[ydata< bin_edges[1]] = hist[0]/np.sum(hist)
        weights[(ydata >= bin_edges[1]) & (ydata < bin_edges[2])] = hist[1]/np.sum(hist)
        weights[(ydata >= bin_edges[2]) & (ydata < bin_edges[3])] = hist[2]/np.sum(hist)
        weights[(ydata >= bin_edges[3]) & (ydata < bin_edges[4])] = hist[3]/np.sum(hist)
        weights[(ydata >= bin_edges[4])] = hist[4]/np.sum(hist)
        
        weights = 1- weights
        
        if(x_trans):
            transformer_x = PowerTransformer().fit(xdata.values.reshape(-1, xdata.shape[1])) #PowerTransformer() method='box-cox' StandardScaler
            xdata = pd.DataFrame(transformer_x.transform(xdata.values.reshape(-1, xdata.shape[1])),columns=xdata.columns)
        else:
            transformer_x = None
        
        if(y_trans):
            transformer_y = PowerTransformer(method='box-cox').fit(ydata.values.reshape(-1, 1))
            ydata = pd.DataFrame(transformer_y.transform(ydata.values.reshape(-1, 1)))[0]
        else:
            transformer_y = None
            
        xdata.reset_index(drop=True, inplace=True)
        ydata.reset_index(drop=True, inplace=True)
        
        formula = 'y ~ X'
        y_train, X_train = dmatrices(formula, {"X": xdata,'y':ydata, 'np':np}, return_type='dataframe')
        
        quantile = 0.95
        median_model = sm.QuantReg(endog=y_train, exog=X_train)
        model_un = median_model.fit(q= quantile,sample_weight=weights)
        
        # median_model = sm.OLS(endog=y_train, exog=X_train)
        # model_un = median_model.fit(sample_weight=weights) #sample_weight=weights
        
        if(save):
            dump(transformer_x, open(os.path.join(path_model_tr, 'x_scaler_{}.pkl'.format(tr)), 'wb'))
            dump(transformer_y, open(os.path.join(path_model_tr, 'y_scaler_{}.pkl'.format(tr)), 'wb'))
            dump(model_un, open(os.path.join(path_model_tr, 'model_{}.pkl'.format(tr)), 'wb'))



def apply_model(tr, dir_path, data_tr, subset):

    scaler_x_ts = load(open(os.path.join(dir_path, 'x_scaler_{}.pkl'.format(tr)), 'rb'))
    scaler_y_ts = load(open(os.path.join(dir_path, 'y_scaler_{}.pkl'.format(tr)), 'rb'))
    model_un = load(open(os.path.join(dir_path, 'model_{}.pkl'.format(tr)), 'rb'))

    xdata = data_tr.loc[:, subset].astype('float64')
    
    if(scaler_x_ts is not None):
        xdata = pd.DataFrame(scaler_x_ts.transform(xdata.values.reshape(-1, xdata.shape[1])),columns = xdata.columns) #float64
    else:
        xdata = pd.DataFrame(xdata.values.reshape(-1, xdata.shape[1]),columns = xdata.columns) #float64
    
    # Create polynomial features to model the exponential relationship
    poly = PolynomialFeatures(degree=1, include_bias=False)
    
    X_poly = poly.fit_transform(xdata)
    X_new = sm.add_constant(X_poly)

    
    if(scaler_y_ts is not None):
        y_pred_ts = pd.DataFrame(scaler_y_ts.inverse_transform(pd.DataFrame(model_un.predict(X_new)).values.reshape(-1, 1)), index = xdata.index)
    else:
        y_pred_ts = pd.DataFrame(pd.DataFrame(model_un.predict(X_new)).values.reshape(-1, 1), index = xdata.index)

    return y_pred_ts

