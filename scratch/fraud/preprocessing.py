import argparse
import os
import requests
import tempfile

import numpy as np
import pandas as pd
import sys

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import logging
import logging.handlers

def _get_logger():
    '''
    로깅을 위해 파이썬 로거를 사용
    # https://stackoverflow.com/questions/17745914/python-logging-module-is-printing-lines-multiple-times
    '''
    loglevel = logging.DEBUG
    l = logging.getLogger(__name__)
    if not l.hasHandlers():
        l.setLevel(loglevel)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))        
        l.handler_set = True
    return l  

logger = _get_logger()

# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
label_column = "rings"

feature_columns_dtype = {
    "sex": str,
    "length": np.float64,
    "diameter": np.float64,
    "height": np.float64,
    "whole_weight": np.float64,
    "shucked_weight": np.float64,
    "viscera_weight": np.float64,
    "shell_weight": np.float64
}
label_column_dtype = {"rings": np.float64}


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, default="/opt/ml/processing")
    parser.add_argument('--dataset_file_path', type=str, default="input/untitled.csv")   
    parser.add_argument('--label_column', type=str, default="fraud")       
    # parse arguments
    args = parser.parse_args()     
    
    logger.info("#############################################")
    logger.info(f"args.base_dir: {args.base_dir}")
    logger.info(f"args.dataset_file_path: {args.dataset_file_path}")    
    logger.info(f"args.label_column: {args.label_column}")        
    
    ##############################################

    base_dir = args.base_dir
    dataset_file_path = args.dataset_file_path
    label_column = args.label_column    

    df = pd.read_csv(
        f"{base_dir}/{dataset_file_path}",
#        header=None, 
#         names=feature_columns_names + [label_column],
#         dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype
#                              )
    )
    logger.info(f"dataset sample \n {df.head(2)}")
    
    logger.info(f"df columns \n {df.columns}")    
    
    y = df.pop(label_column)
    


    

    float_cols = df.select_dtypes(include=['float']).columns.values
    int_cols = df.select_dtypes(include=['int']).columns.values
    numeric_features = np.concatenate((float_cols, int_cols), axis=0).tolist()

    
    
#     numeric_features = list(feature_columns_names)
#     numeric_features.remove("sex")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_features = df.select_dtypes(include=['object']).columns.values.tolist()    
#     categorical_features = ["sex"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    


    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    
    
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)
    
    X = np.concatenate((y_pre, X_pre), axis=1)

    
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(.7*len(X)), int(.85*len(X))])

    
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    
    logger.info(f"preprocessed train sample \n {pd.DataFrame(train).head(2)}")
    logger.info(f"All files are preprocessed")
