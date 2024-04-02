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
    dataset = Dataset()
    train_raw, test_raw = dataset.get_raw()
    X_train, y_train, X_test, y_test = get_dataset()
    y_pred_prob, y_pred, train_pred_prob, train_pred = lightgbm(X_train, y_train, X_test, y_test)
    predictions = GCN(train_raw,test_raw,train_pred_prob,y_pred_prob)

