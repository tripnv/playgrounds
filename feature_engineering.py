#%%
import pandas as pd
import numpy as np
import os 
import math

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib as mpl
import seaborn as sns 

from tqdm import tqdm
from sklearn.model_selection import KFold
#%%

RANDOM_SEED = 912
plot_params = {
    'font.family': 'Times',
    'font.weight': 'light',
    
    'figure.figsize': (15,15),
    'figure.frameon': False, 
    'figure.titlesize': 'xx-large',
    'figure.titleweight': 'normal',
    
    'axes.titlesize': 'x-large',
    'axes.titlecolor': 'black',
    'axes.titleweight': 'normal',
    'axes.titlelocation': 'center',
    'axes.labelsize': 'x-large',

    'grid.alpha': .25, 
    'legend.frameon':False,
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large',
}

pylab.rcParams.update(plot_params)
sns.set_palette('mako')
#%%
data_folder = 'data'
train_data_path = 'train.csv'
test_data_path = 'test.csv'
sample_submission_path = 'sample_submission.csv'
#%%
train_path = os.path.join(data_folder, train_data_path)
test_path = os.path.join(data_folder, test_data_path)
sample_path = os.path.join(data_folder, sample_submission_path)
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)
#%%

feature_columns = train.drop('cost', axis = 1).columns.tolist()
target_column = 'cost'
assert train['id'].nunique() == train.__len__()
feature_columns.remove('id')
binary_features = train.columns[train.isin([0,1]).all()].tolist()
binary_features
non_binary_features = list(set(feature_columns).difference(set(binary_features)))
continuous_features = ['store_sqft', 'store_sales(in millions)', 'gross_weight']

discrete_features = list(set(non_binary_features).difference(set(continuous_features)))
assert len(feature_columns) == len(binary_features) \
                                + len(non_binary_features) 
assert len(feature_columns) == len(discrete_features) \
                                + len(continuous_features) \
                                + len(binary_features)
train.loc[:, binary_features] = train.loc[:, binary_features].astype('int16')
#%%
discrete_features
#%%
continuous_features
# %%
print(train.shape)
print('Binary: \t', train.loc[:, binary_features].shape[1])
print('Cont: \t\t', train.loc[:, continuous_features].shape[1])
print('Discrete: \t', train.loc[:, discrete_features].shape[1])
print('+ index + target')
# %%
feature_subset = train.loc[:, continuous_features + discrete_features]
feature_subset
# %%

# Handcrafted features 

feature_subset['store_sqft'] / feature_subset['store_sales(in millions)']
feature_subset['store_sales(in millions)'] / feature_subset['store_sqft'] 
feature_subset['gross_weight'] / feature_subset['store_sales(in millions)']
feature_subset['store_sales(in millions)'] / feature_subset['gross_weight'] 

feature_subset['unit_sales(in millions)'] / feature_subset['gross_weight'] 
feature_subset['unit_sales(in millions)'] / feature_subset['store_sqft'] 
feature_subset['unit_sales(in millions)'] / feature_subset['avg_cars_at_home'] 
feature_subset['unit_sales(in millions)'] / feature_subset['gross_weight'] 

feature_subset['avg_cars_at_home'] / feature_subset['num_children_at_home']
# %%
