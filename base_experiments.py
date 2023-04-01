import os
import math

import pandas as pd 
import numpy as np



from tqdm import tqdm

# from sklearnex import patch_sklearn
# patch_sklearn()


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, mean_squared_error


from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from baseline import read_source_files
RANDOM_SEED = 999
N_SPLITS = 5


if __name__ == '__main__':

    models = [
        LinearRegression(), 
        Ridge(),
        Lasso(),
        ElasticNet()
    ]

    train_df, _, test_df, sample_submission_df = read_source_files(
        data_folder='data',
        train_data_path='train.csv',
        test_data_path='test.csv',
        sample_submission_path='sample_submission.csv'
    )

    x, y = train_df.drop('cost', axis = 1), train_df.cost

    cv = KFold(n_splits=N_SPLITS, random_state=RANDOM_SEED, shuffle=True)

    for model in models:

        print(model.__repr__())
        metrics = {
            "rmsle":[],
            "mse":[],
            "mae":[]
        }


        for fold_idx, (train_idxs, test_idxs) in enumerate(cv.split(x, y)):
            x_train, x_valid = x.loc[train_idxs], x.loc[test_idxs]
            y_train, y_valid = y.loc[train_idxs], y.loc[test_idxs]

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


        print(f"\t{model.__repr__()}")
        print(f"\tMean metrics: \n\t MSLE: {mean_rmsle}\n\t MAE: {mean_mae}\n\t MSE: {mean_mse}")
