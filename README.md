# instacart-solution
My solution for the Instacart Market Basket Analysis competition on Kaggle.

## Objective
Given anonymized data on customer orders over time, predict which previously purchased products will be in a user’s next order.
F1 score is used evaluate the prediction accuracy.

## Dataset
The dataeset contains a sample of over 3 million grocery orders from more than 200,000 Instacart users. For each user, between 4 and 100 of their orders are provided.

[Data Exploration Analysis Notebook](https://github.com/panyao/instacart-solution/blob/master/exploration/Data%20Exploratory%20for%20Instacart%20Market%20Basket%20Analysis.ipynb)

## Solution approach
The problem was reformulated as a binary classification problem: given a user u, a product p, and the user's previous n orders, predict if the product will be reordered in user's n+1 order. 


### Feature engineering
Exploratory data analysis has been performed to understand what features might be related to users' reorder behaviors.
4 groups of features are used as input, which include word embedding learned for product_id using word2vec.
#### Product features
 - product_reorder_rate', 'product_reorder_probability', 
   'vec1', 'vec2', 'product_nb_users',  'product_mean_add_to_cart_position',
   'prod_buy_second_time_total_cnt', 'frequency_mean', 'frequency_std
#### User features
 - 'user_total_orders', 'user_total_items', 'total_distinct_items',
       'user_average_days_between_orders', 'user_average_basket', 'user_median_basket',
       'user_reorder_ratio', 'user_period', 'user_nb_aisles', 'user_nb_departments',
#### Order features
 - 'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio', 
       'aisle_id', 'department_id', 'product_orders', 'product_reorders',
#### User product pair features
 - 'UP_orders', 'UP_orders_ratio', 'UP_average_pos_in_cart', 
       'UP_orders_since_last','UP_delta_hour_vs_last',
       'UP_first_order', 'UP_last_order', 'UP_order_rate_since_first_order', 
       'UP_recency', 'UP_recency_day', 'UP_recency_exp',
       'UP_interval_distance', 'UP_interval_mean', 'UP_interval_std', 'UP_order_number_skew',
       'UP_order_streak'

### Model
Gradient boosting algorithms are used. Both lightgbm and xgboost are used to train the model.
The prediction result is a value in the range of [0,1] for each (user,product) pair. A weighted average is used to average over multiple models to generate prediction. 

### F1 Score maximization
The numerical probability needs to be converted to binary decisions.
A threshold h is used to determine whether a product will be reordered. A product will be predicted as reorder is its predicted reorder probability is larger than threshold.

Different threshold value are used in diferent order in order to maximize F1 score. An algorithm proposed in this paper [1] is implemented. The idea is maximize the F1 score expectation given the predicted posteriors P = {p_1, p_2, ... , P_n} for all the products a user has purchased.

<a href="https://www.codecogs.com/eqnedit.php?latex=argmax_{k\in&space;[0,&space;n]}&space;E[F1(P,&space;k))]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?argmax_{k\in&space;[0,&space;n]}&space;E[F1(P,&space;k))]" title="argmax_{k\in [0, n]} E[F1(P, k))]" /></a>

The goal is to find the value k so that by reordering the products with largest k reorder probability, the expected F1 score for this order is the largest.

[1] Ye, N., Chai, K., Lee, W., and Chieu, H. "Optimizing F-measure: A Tale of Two Approaches", ICML, 2012. 

## Running instructions

### Requirements
 - lightgbm==2.0
 - xgboost==0.6
 - gensim==2.3
 - numpy==1.13
 - pandas== 0.19
 - scikit-learn==0.19
 
### How to run
 - Put raw csv files into data/raw directory
 - Preprocessing: order_features.py order
 - Model Training:
 - maximizeF1.py
