# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
from sklearn.externals import joblib


if __name__ == '__main__':
    
    lightgmb_weight = 0.7
    xgboost_weight = 0.3

    model1 = pd.read_pickle("data/prediction_lightgbm.pkl")
    model2 = pd.read_pickle("data/prediction_xgboost.pkl")
    
    result = pd.merge(model1, model2, on = ['order_id', 'product_id'], suffixes=('_x', '_y'))
    
    result['prediction'] = result['prediction_x']*lightgmb_weight + result['prediction_y']*xgboost_weight
    result.drop(['prediction_x', 'prediction_y'], axis = 1, inplace = True)
    
    result['prediction'] = result['prediction'].astype(np.float32)
    result['order_id'] = result['order_id'].astype(np.int32)
    result['product_id'] = result['product_id'].astype(np.int32)
    
    result.to_pickle('data/prediction_two_model.pkl')