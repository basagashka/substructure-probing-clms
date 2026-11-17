import os
import json
import joblib
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

from rdkit import Chem
import rdkit

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from xgboost import XGBRegressor

from tasks.lipo import Lipophilicity

def prep_instances(instance_dict):
    idxs, x, y = [], [], []
    for idx, instance in instance_dict.items():
        idxs.append(idx)
        x.append(np.array(instance["fp"]))
        y.append(instance["y_true"])
    return idxs, x, y

def write_data(path, data): 
    with open(path, 'w') as f:
        json.dump(data, f)

dataset = Lipophilicity("data/lipo/")

fp_sizes = [300, 512, 1024, 2048] 
fp_methods = ["mfp", "rdkfp", "apfp", "ttfp"]


for fp_size in fp_sizes:
    for fp_method in fp_methods:
        # Load respective method
        train = dataset.get_fingerprints_dataset(f"{fp_method}_train_{fp_size}")
        val = dataset.get_fingerprints_dataset(f"{fp_method}_val_{fp_size}")
        test = dataset.get_fingerprints_dataset(f"{fp_method}_test_{fp_size}")
        
        idx_train, x_train, y_train = prep_instances(train)
        idx_val, x_val, y_val = prep_instances(val)
        idx_test, x_test, y_test = prep_instances(test)
        
        lr = LinearRegression().fit(np.stack(x_train), np.array(y_train))
        lr_results = lr.predict(np.stack(x_test))
        
        xgb_reg = XGBRegressor()
        xgb_reg.fit(np.stack(x_train), y_train)
        xgb_results = xgb_reg.predict(np.stack(x_test))

        # Tune c
        best_c = 0
        best_mse = 10000.0
        
        for c in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0]:
            svr = SVR(C=c)
            svr.fit(np.stack(x_train), np.array(y_train))
            val_results = svr.predict(np.stack(x_val))
            if root_mean_squared_error(val_results, y_val) <= best_mse:
                best_mse = root_mean_squared_error(val_results, y_val)
                best_c = c
        best_svr = SVR(C=best_c)
        best_svr.fit(np.stack(x_train), np.array(y_train))
        svr_results = best_svr.predict(np.stack(x_test))
        
        svr_rmse = root_mean_squared_error(svr_results, y_test)
        lr_rmse = root_mean_squared_error(lr_results, y_test)
        xgb_rmse = root_mean_squared_error(xgb_results, y_test)
        logging.info(f"Fingerprints size: {fp_size} --- Fingerprint method: {fp_method} --- Test RMSE LR: {lr_rmse} --- Test RMSE SVR (c={best_c}): {svr_rmse} --- Test RMSE XGB: {xgb_rmse}")
       
        # save
        joblib.dump(lr, f"models/{fp_method}_lr_{fp_size}.pkl") 
        joblib.dump(best_svr, f"models/{fp_method}_svr_{fp_size}.pkl") 

        
        
