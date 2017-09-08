# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

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
    
    # user*product reorder frequency(interval) feature

    up = dict()
    skew = dict()
    i = 0
    
    for row in priors_sorted.itertuples():
        if i%1000000 == 0:
            print(i)
        i = i + 1    
        z = row.user_id.astype(np.int64)*100000 + row.product_id
    
        if z not in up:
            up[z] = [row.days_ago_since_last]
            skew[z] = [row.order_number]
        else:
            up[z].append(row.days_ago_since_last)
            skew[z].append(row.order_number)
     
    up_frequency = dict()
    i = 0
    for key, value in up.items():
        if i%100000 == 0:
            print(i)
        i = i + 1    
        dif = [(value[i]-value[i+1]) for i in range(len(value)-1)]
        # distance: last purchase day - average purchase interval
        avg = np.mean(dif)
        if len(value) == 1:
            dist = value[0]
        else:
            dist = np.abs(value[-1]- avg)
        up_frequency[key] = {'mean': avg, 'std': np.std(dif), 'distance': dist}
        
    df_up_frequency = pd.DataFrame.from_dict(up_frequency, orient='index')
    df_up_frequency.columns = {'up_frequency_distance', 'up_frequency_std', 'up_frequency_mean'}
    df_up_frequency['up_frequency_distance'] = df_up_frequency['up_frequency_distance'].astype(np.float32)
    df_up_frequency['up_frequency_std'] = df_up_frequency['up_frequency_std'].astype(np.float32)
    df_up_frequency['up_frequency_mean'] = df_up_frequency['up_frequency_mean'].astype(np.float32)
    save_to_hdfs(df_up_frequency, "df_up_frequency")
    
    i = 0
    for key, value in skew.items():
        if i%1000000 == 0:
            print(i)
        i = i + 1
        skew[key] = np.mean(value)/value[-1]
        
    df_skew = pd.DataFrame.from_dict(skew, orient='index')
    df_skew.columns = {'order_number_skew'}
    df_skew['order_number_skew'] = df_skew['order_number_skew'].astype(np.float32)
    save_to_hdfs(df_skew, "df_skew")