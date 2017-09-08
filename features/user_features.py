# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from util import save_to_hdfs

def load_data():
    IDIR = os.path.join('..', 'data', 'raw')
    print('loading prior')
    priors = pd.read_csv(os.path.join(IDIR, 'order_products__prior.csv'), dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})
    
    print('loading products')
    products = pd.read_csv(os.path.join(IDIR, 'products.csv'), dtype={
        'product_id': np.uint16,
        'product_name': object,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
        usecols=['product_id', 'product_name', 'aisle_id', 'department_id'])

    print('loading orders')
    orders = pd.read_csv(os.path.join(IDIR, 'orders.csv'), dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})
    
    return priors, products, orders
    
if __name__ == '__main__':
    
    priors, products, orders = load_data()

    ### user features
    print('computing user features')
    users = pd.DataFrame()
    users['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
    users['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)
    users['user_period'] = orders.groupby('user_id')['days_since_prior_order'].sum()  
    
    users['total_items'] = priors.groupby('user_id').size().astype(np.int16)
    users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
    users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)
    users['reorder_ratio'] =  (priors.groupby('user_id')['reordered'].sum()/priors[priors.order_number > 1].groupby('user_id').size()).astype(np.float32)
    
    users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)   
    users['median_basket'] = orders.groupby('user_id')['basket_size'].median().astype(np.float32)
    
    priors = pd.merge(priors, products[['product_id','aisle_id','department_id']], on ='product_id', how = 'left')

    users['nb_aisles'] = priors.groupby('user_id')['aisle_id'].apply(set).map(len).astype(np.int16)
    users['nb_departments'] = priors.groupby('user_id')['department_id'].apply(set).map(len).astype(np.int16)

    save_to_hdfs(users, 'users')
    
