# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from util import save_to_hdfs

if __name__ == '__main__':
    
    # Loading data
    IDIR = os.path.join('..', 'data', 'raw') 

    print('loading orders')
    orders = pd.read_csv(os.path.join(IDIR, 'orders.csv'), dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

    ### order features
    orders['days_since_ratio'] = orders.days_since_prior_order / orders.user_average_days_between_orders

    orders['days_ago_since_last'] = orders.groupby('user_id')['days_since_prior_order'].cumsum()
    orders['days_ago_since_last'].fillna(0, inplace = True)
    orders['days_ago_since_last'] = orders.groupby('user_id')['days_ago_since_last'].transform(max) - orders['days_ago_since_last']
    orders['days_ago_since_last'] = orders['days_ago_since_last'].astype(np.int16)

    save_to_hdfs(orders, 'orders')
    
