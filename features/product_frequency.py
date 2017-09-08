# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew

from util import save_to_hdfs
from util import load_from_hdfs

if __name__ == '__main__':
    
    IDIR = '../data/raw' 
    priors = pd.read_csv(os.path.join(IDIR, 'order_products__prior.csv'), dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})
    orders = load_from_hdfs('orders')
    
    priors = priors.join(orders, on='order_id', rsuffix='_')
    priors.drop('order_id_', inplace=True, axis=1)
    priors.head()

    priors_sorted = priors.sort_values(['product_id', 'user_id', 'order_number'])
    
    # computing order frequency for each product
    d= dict()
    i = 0
    prev_user_id = -1
    curDay = 0
    for row in priors_sorted.itertuples():
        if i%1000000 == 0:
            print(i)
        i = i + 1
        pid = row.product_id
    
        # (#of orders, array of interval, last purchase day)
        if pid not in d:
            d[pid] = (0, [], row.days_ago_since_last)
            prev_user_id = row.user_id
        else:
            if row.user_id != prev_user_id:
                d[pid] = (d[pid][0],
                          d[pid][1],
                          row.days_ago_since_last)
                prev_user_id = row.user_id
            # same user
            else:
                inter_list = d[pid][1]
                inter_list.append(d[pid][2] - row.days_ago_since_last)  # append does not return value!!
                d[pid] = (d[pid][0] + 1,
                          inter_list,
                          row.days_ago_since_last)
                
    interval = dict()
    # calculate average days between same product purchase
    i = 0
    for key in d:
        if i%10000 == 0:
            print(i)
        i= i+ 1
        if d[key][0] != 0:
            interval[key] = {'mean': np.mean(d[key][1]), 'std': np.std(d[key][1]), 'kurtosis': kurtosis(d[key][1]), 'skew': skew(d[key][1])}
        
    
    frequency = pd.DataFrame.from_dict(interval, orient='index')
    frequency.columns = ['frequency_mean', 'frequency_std','frequency_skew','frequency_kurtosis']
    frequency['product_id'] = frequency.index
    
    save_to_hdfs(frequency, 'frequency')