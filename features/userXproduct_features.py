# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
from util import save_to_hdfs
from util import load_from_hdfs

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

    users = load_from_hdfs("users")
    
    return priors, products, orders, users


if __name__ == '__main__':
    
    priors, products, orders, users = load_data()

    ### userXproduct features
    #Generate unique userXproduct ID
    priors['user_product'] = priors.product_id + priors.user_id.astype(np.int64) * 100000

    # precomputer exp value for faster speed
    MAX_DAY = 366
    alpha = 0.02   # decay parameter
    expValue = np.zeros(MAX_DAY)
    for i in range(MAX_DAY):
        expValue[i] = np.exp(-alpha*i)
    
    d= dict()
    i = 0
    for row in priors.itertuples():
        if i % 1000000 == 0:
            print(i)
        i = i + 1
        z = row.user_product
        if z not in d:        
            d[z] = (1,
                    (row.order_number, row.order_id),
                    row.add_to_cart_order,
                    row.order_number,
                    row.order_number,
                    users.loc[row.user_id, 'nb_orders'],
                    1+ row.order_number/users.loc[row.user_id, 'nb_orders'],
                    1/(row.days_ago_since_last+1),
                    expValue[row.days_ago_since_last])
        else:
            d[z] = (d[z][0] + 1,
                    max(d[z][1], (row.order_number, row.order_id)),
                    d[z][2] + row.add_to_cart_order,
                    min(d[z][3], row.order_number),
                    max(d[z][4], row.order_number),
                    d[z][5],
                    d[z][6] + 1+ row.order_number/d[z][5],
                    d[z][7] + 1/(row.days_ago_since_last+1),
                    d[z][8] + expValue[row.days_ago_since_last])
        
    userXproduct = pd.DataFrame.from_dict(d, orient='index')
    
    userXproduct.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart', 'first_order', 'last_order', 
                            'rate_since_first_order', 'weight', 'weight_day', 'weight_exp']
    userXproduct.nb_orders = userXproduct.nb_orders.astype(np.int16)
    userXproduct.last_order_id = userXproduct.last_order_id.map(lambda x: x[1]).astype(np.int32)
    userXproduct.sum_pos_in_cart = userXproduct.sum_pos_in_cart.astype(np.int16)
    userXproduct.rate_since_first_order = userXproduct.nb_orders/(userXproduct.rate_since_first_order-userXproduct.first_order+1).astype(np.float32)
    
    userXproduct.first_order = userXproduct.first_order.astype(np.int16)
    userXproduct.last_order = userXproduct.last_order.astype(np.int16)
    userXproduct.weight = userXproduct.weight.astype(np.float32)
    userXproduct.weight_day = userXproduct.weight_day.astype(np.float32)
    userXproduct.weight_exp = userXproduct.weight_exp.astype(np.float32)

    save_to_hdfs(userXproduct, 'userXproduct')
    
