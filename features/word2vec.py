# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import gensim
from sklearn import decomposition
    
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

    print('loading train')
    train = pd.read_csv(IDIR + 'order_products__train.csv', dtype={
                'order_id': np.int32,
                'product_id': np.uint16,
                'add_to_cart_order': np.int16,
                'reordered': np.int8})
    
    return priors, products, train
    

if __name__ == '__main__':
    
    priors, products, train = load_data()

    train["product_id"] = train["product_id"].astype(str)
    priors["product_id"] = priors["product_id"].astype(str)
    
    train_products = train.groupby("order_id").apply(lambda order: order['product_id'].tolist())
    prior_products = priors.groupby("order_id").apply(lambda order: order['product_id'].tolist())
    
    sentences = prior_products.append(train_products).values
    
    # size: the dimensionality of the feature vectors.
    # window: the maximum distance between the current and predicted word within a sentence.
    # min_count: ignore all words with total frequency lower than this.
    # worker: number of threads to train the model 
    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=5, workers=4)
    model.save('word2vec.model')
    
    id = []
    matrix = []
    for i in range(50000):
        if str(i) in model.wv.vocab:
            id.append(i)
            matrix.append(model.wv[str(i)])
    mat = np.array(matrix)
    
    embed_dim = 2
    pca = decomposition.PCA(n_components = embed_dim)
    reduced = pca.fit_transform(mat)
    
    array = np.zeros([mat.shape[0], embed_dim])
    for i in range(len(id)):
        array[id[i]-1] = reduced[i]
    
    products['vec1']= array[:,0]
    products['vec2']= array[:,1]
    
    save_to_hdfs(products, 'products')