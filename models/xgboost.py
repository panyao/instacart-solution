# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 23:34:14 2017

@author: py
"""

import os
import pandas as pd
import numpy as np
import xgboost
from sklearn.externals import joblib
    
from util import save_to_hdfs
from util import load_from_hdfs
from maximizeF1 import applyParallel
from maximizeF1 import create_products

if __name__ == '__main__':
    df_train = load_from_hdfs('df_train')
    
    #user related
    features_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
       'user_average_days_between_orders', 'user_average_basket', 'user_median_basket',
       'user_reorder_ratio', 'user_period', 'user_nb_aisles', 'user_nb_departments',
       #order related
       'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio', 
       'aisle_id', 'department_id', 'product_orders', 'product_reorders',
       #product related
       'product_reorder_rate', 'product_reorder_probability', 
       'vec1', 'vec2', 
       'product_nb_users',  'product_mean_add_to_cart_position',
       'prod_buy_second_time_total_cnt', 
       'frequency_mean', 'frequency_std', 
       # user*product
       'UP_orders', 
       'UP_orders_ratio', 'UP_average_pos_in_cart', 
       'UP_orders_since_last','UP_delta_hour_vs_last',
       'UP_first_order', 'UP_last_order', 'UP_order_rate_since_first_order', 
       'UP_recency', 'UP_recency_day', 'UP_recency_exp',
       'UP_interval_distance', 'UP_interval_mean', 'UP_interval_std', 'UP_order_number_skew',
       'UP_order_streak'] 
    
    categories = ['product_id', 'aisle_id', 'department_id']
    
    # train on entire database
    d_train = xgboost.DMatrix(X_train[features_to_use], y_train)
    d_test = xgboost.DMatrix(X_test[features_to_use])
    
    xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.05
    ,"max_depth"        : 6
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}
    ROUNDS = 151   
    
    print('xgboost train:')
    xgboost = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=ROUNDS,  verbose_eval=10)
    
    #lgb.plot_importance(lightGBM, figsize=(9,40))
    #plt.show()
    

    # save model to file
    joblib.dump(xgboost, "xgboost.model")
    
    df_test = load_from_hdfs('df_test')
    final_preds = xgboost.predict(df_test[features_to_use])
    df_final_pred = pd.DataFrame(df_test[['order_id', 'product_id']])
    df_final_pred['prediction'] = final_preds
    
    df_final_pred = df_final_pred.loc[df_final_pred.prediction > 0.01, ['order_id', 'prediction', 'product_id']]
    df_order = applyParallel(df_final_pred.groupby(df_final_pred.order_id), create_products).reset_index()

    df_order[['order_id', 'products']].to_csv('../submission/submission.csv', index=False)
        
    