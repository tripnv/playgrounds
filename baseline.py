#%%
import pandas as pd
import numpy as np
import os 
import math

from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, mean_squared_error
from xgboost import XGBRegressor

RANDOM_SEED = 999

#%%
def read_source_files(
        data_folder = 'data',
        train_data_path = None,
        valid_data_path = None,
        test_data_path = None,
        sample_submission_path = None):
    
    train_df = None
    valid_df = None
    test_df = None
    sample_submission_df = None
    
    if train_data_path:
        train_full_path = os.path.join(data_folder, train_data_path)
        train_df = pd.read_csv(train_full_path)
        
    if valid_data_path:
        valid_full_path = os.path.join(data_folder, valid_data_path)
        valid_df = pd.read_csv(valid_full_path)
        
    if test_data_path:
        test_full_path = os.path.join(data_folder, test_data_path)
        test_df = pd.read_csv(test_full_path)
        
    if sample_submission_path:
        sample_submission_full_path = os.path.join(data_folder, sample_submission_path)
        sample_submission_df = pd.read_csv(sample_submission_full_path)
        
    return train_df, valid_df, test_df, sample_submission_df

train_df, _, test_df, sample_submission_df = read_source_files(
    data_folder='data',
    train_data_path='train.csv',
    test_data_path='test.csv',
    sample_submission_path='sample_submission.csv'
)

#%%

# feature_columns = train.drop('cost', axis = 1).columns.tolist()
# target_column = 'cost'
# assert train['id'].nunique() == train.__len__()
# feature_columns.remove('id')
# binary_features = train.columns[train.isin([0,1]).all()].tolist()
# binary_features
# non_binary_features = list(set(feature_columns).difference(set(binary_features)))
# continuous_features = ['store_sqft', 'store_sales(in millions)', 'gross_weight']

# discrete_features = list(set(non_binary_features).difference(set(continuous_features)))
# assert len(feature_columns) == len(binary_features) \
#                                 + len(non_binary_features) 
# assert len(feature_columns) == len(discrete_features) \
#                                 + len(continuous_features) \
#                                 + len(binary_features)
# train.loc[:, binary_features] = train.loc[:, binary_features].astype('int16')
#%%

def fit_cv(x, y, params = None):
    cv = KFold(n_splits=3, random_state=RANDOM_SEED, shuffle=True)

    metrics = {
        "rmsle":[],
        "mse":[],
        "mae":[]
    }


    for fold_idx, (train_idxs, test_idxs) in enumerate(cv.split(x, y)):
        x_train, x_valid = x.loc[train_idxs], x.loc[test_idxs]
        y_train, y_valid = y.loc[train_idxs], y.loc[test_idxs]

        model = XGBRegressor()
        model.fit(x_train, y_train)

        y_preds = model.predict(x_valid)
        rmsle = math.sqrt(mean_squared_log_error(y_valid, y_preds))
        mse = mean_squared_error(y_valid, y_preds)
        mae = mean_absolute_error(y_valid, y_preds)

        metrics['mae'].append(mae)
        metrics['rmsle'].append(rmsle)
        metrics['mse'].append(mse)

        # print(f"\tFold {fold_idx + 1}: \n\t MSLE: {rmsle}\n\t MAE: {mae}\n\t MSE: {mse}")


    mean_mae = np.mean(metrics['mae'])
    mean_rmsle = np.mean(metrics['rmsle'])
    mean_mse = np.mean(metrics['mse'])

    return mean_mae, mean_rmsle, mean_mse


x, y = train_df.drop('cost', axis = 1), train_df.cost
mae, rmsle, mse = fit_cv(x, y)

print(f"\tMean metrics: \n\t MSLE: {rmsle}\n\t MAE: {mae}\n\t MSE: {mse}")
# %%
