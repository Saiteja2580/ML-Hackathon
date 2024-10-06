import lightgbm as lgb
import pandas as pd


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge


def lightgbm_impute(data):
    for column in data.columns:
        if data[column].isnull().sum() > 0:

            x_gbm = data.drop(columns=[column])
            y_gbm = data[column]

            x_train_gbm = x_gbm[~y_gbm.isnull()]
            y_train_gbm = y_gbm[~y_gbm.isnull()]


            x_missing_gbm = x_gbm[y_gbm.isnull()]

            train_data_gbm = lgb.Dataset(x_train_gbm,label=y_train_gbm)

            params = {
                'boosting_type': 'gbdt',  # Gradient Boosting Decision Trees
                'objective': 'regression',  # Regression task
                'metric': 'l2',  # Mean Squared Error loss
                'learning_rate': 0.1,  # Learning rate
                'num_leaves': 31,  # Number of leaves in the tree
                'verbose': -1  # Suppress LightGBM output
            }

            model_gbm = lgb.train(params,train_data_gbm,num_boost_round=100)

            y_pred_miss_gbm = model_gbm.predict(x_missing_gbm)

            data.loc[y_gbm.isnull(),column] = y_pred_miss_gbm

    return data 





def iterative_impute(data):
    
    ridge_estimator = Ridge(alpha=0.1)
    imputer = IterativeImputer(estimator=ridge_estimator,max_iter=10,random_state=42)

    data_imputed = imputer.fit_transform(data)
    data_imputed = pd.DataFrame(data_imputed,columns=data.columns)
    return data_imputed
