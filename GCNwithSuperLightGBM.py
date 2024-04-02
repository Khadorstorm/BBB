#import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from LightGBM import lightgbm
from graphgpt import GCN
from utils import *

if __name__ == "__main__":
    dataset = myDataset('train/train_feature.csv','test/test_feature.csv')
    train_raw, test_raw = dataset.get_raw()
    X_train, y_train, X_test, y_test = get_dataset('train/train_feature.csv','test/test_feature.csv')
    y_pred_prob, y_pred, train_pred_prob, train_pred = lightgbm(X_train, y_train, X_test, y_test)
    print(type(train_pred_prob))
    print(train_pred_prob.shape)
    zeros_array = np.zeros(train_pred_prob.shape[0])
    new_array = np.column_stack((train_pred_prob, train_pred_prob))
    zeros_array2 = np.zeros(y_pred_prob.shape[0])
    new_array2 = np.column_stack((y_pred_prob, y_pred_prob))
    #predictions = GCN(train_raw,test_raw,train_pred_prob,y_pred_prob)
    predictions = GCN(train_raw, test_raw, new_array,new_array2)

