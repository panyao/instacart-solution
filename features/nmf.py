# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
    
from util import save_to_hdfs
from util import load_from_hdfs


if __name__ == '__main__':
    IDIR = os.path.join('..', 'data', 'raw')
    order_prior = pd.read_csv(os.path.join(IDIR, 'order_products__prior.csv'))
    orders = pd.read_csv(os.path.join(IDIR, 'orders.csv'))
    train = pd.read_csv(os.path.join(IDIR, 'order_products__train.csv'))
    products = pd.read_csv(os.path.join(IDIR, 'products.csv'))
    
    prior = orders[orders['eval_set']=='prior']
    
    df_order = pd.merge(orders[orders['eval_set'] != 'prior'], prior[['user_id', 'order_id', 'order_number']], on = 'user_id')
    
    
    df_order = pd.merge(df_order, order_prior, left_on = 'order_id_y', right_on = 'order_id')
    df_order['new_order_id'] = df_order['order_id_x']
    df_order['prior_order_id'] = df_order['order_id_y']
    df_order = df_order.drop(['order_id_x', 'order_id_y'], axis = 1)
    del [orders, order_prior, train]
    
    product_list = df_order[df_order['reordered']==1].groupby(['user_id', 'order_number_x', 'new_order_id'])['product_id'].apply(list)
    product_list = pd.DataFrame(product_list.reset_index())
    product_list['num_products_reordered'] = product_list.product_id.apply(len)
    
    indptr = [0]
    indices = []
    data = []
    column_position = {}
    # input must be a list of lists
    for order in product_list['product_id']:
        for product in order:
            index = column_position.setdefault(product, len(column_position))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
        
    prod_matrix = csr_matrix((data, indices, indptr), dtype=int)
    del(df_order)
    
    print('Done constructing prod_matrix')
    
    
    
    nmf = NMF(n_components = 100, random_state = 42)
    model = nmf.fit(prod_matrix)
    H = model.components_
    model.components_.shape
    
    W = model.transform(prod_matrix)
    user_data = pd.DataFrame(normalize(W), index = product_list['user_id'])
    user_data.to_csv('data/nmf.csv')