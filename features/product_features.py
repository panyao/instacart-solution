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

    print('add order info to priors')
    orders.set_index('order_id', inplace=True, drop=False)
    priors = priors.join(orders, on='order_id', rsuffix='_')
    priors.drop('order_id_', inplace=True, axis=1)


    print('computing product features')
    prods = pd.DataFrame()
    prods['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
    prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
    prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
    
    prods['nb_users'] = priors.groupby(priors.product_id)['user_id'].apply(set).apply(len)
    grouped = priors.groupby(priors.product_id)['add_to_cart_order']
    prods['mean_add_to_cart'] = grouped.apply(sum)/grouped.apply(len).astype(np.float32)
    
    products = products.join(prods, on='product_id')
    
    grouped = priors.groupby('product_id')
    products['prod_buy_first_time_total_cnt'] = grouped['user_buy_product_times'].apply(lambda x: sum(x==1))
    products['prod_buy_second_time_total_cnt'] = grouped['user_buy_product_times'].apply(lambda x: sum(x==2))
    
    products['prod_reorder_probability'] = products.prod_buy_second_time_total_cnt / products.prod_buy_first_time_total_cnt

    del prods
    
    save_to_hdfs(products, 'products')
    
