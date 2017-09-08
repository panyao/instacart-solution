# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from util import save_to_hdfs

'''
    Prepare ground truth for validation
'''

def products_concat(series):
    out = ''
    if np.isnan(series).any():
        return 'None'
    else:
        for product in series:
            out = out + str(product) + ' '
    
        return out.rstrip()
    
if __name__ == '__main__':
    IDIR = os.path.join('..', 'data', 'raw') 
    
    print('loading train')
    train = pd.read_csv(os.path.join(IDIR,'order_products__train.csv'), dtype={
                'order_id': np.int32,
                'product_id': np.uint16,
                'add_to_cart_order': np.int16,
                'reordered': np.int8})
    
    train.set_index(['order_id', 'product_id'], inplace=True, drop=False)
    
    # Prepare ground truth for validation
    train_reorder_products = pd.DataFrame(train[train.reordered == 1].groupby(
        'order_id')['product_id'].apply(list))
    train_reorder_products.reset_index(inplace=True)
    train_reorder_products.columns = ['order_id','products_list']
    
    train_products = pd.DataFrame(train.groupby(
        'order_id')['product_id'].apply(list))
    train_products.reset_index(inplace=True)
    train_products.columns = ['order_id','products_list_all']
    
    temp = pd.merge(train_products, train_reorder_products, on = 'order_id', how = 'left')
    train_products = temp.drop('products_list_all', axis = 1)
    
    train_products['products_list'] = train_products['products_list'].apply(products_concat)
    save_to_hdfs('train_products', 'train_products')