from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
#from utils import *

class Dataset:
    def __init__(self, train_path='~/PycharmProjects/BBB5/train/train_feature.csv',
                test_path = '~/PycharmProjects/BBB5/test/test_feature.csv'):
        self.train_data = pd.read_csv(train_path)
        self.train_data.fillna(0, inplace=True)
        self.X_train_tabular = self.train_data.drop(['id', 'label', 'smi'], axis=1)  # Drop the 'id' and 'label' columns for features
        self.X_train_smi = self.train_data['smi']
        self.y_train = self.train_data['label']  # Use the 'label' column as the label

        # Load test data
        self.test_data = pd.read_csv(test_path)
        self.test_data.fillna(0, inplace=True)
        self.X_test_tabular = self.test_data.drop(['id', 'label', 'smi'], axis=1)
        self.X_test_smi = self.test_data['smi']
        self.y_test = self.test_data['label']

    def get_tabular(self):
        return self.X_train_tabular,self.y_train,self.X_test_tabular,self.y_test

    def get_smi(self):
        return self.X_train_smi,self.y_train,self.X_test_smi,self.y_test

    def get_raw(self):
        return self.train_data, self.test_data

def get_dataset(train_path='~/PycharmProjects/BBB5/train/train_feature.csv',
                test_path = '~/PycharmProjects/BBB5/test/test_feature.csv'):
    train_data = pd.read_csv(train_path)
    train_data.fillna(0, inplace=True)
    X_train = train_data.drop(['id', 'label', 'smi'], axis=1)  # Drop the 'id' and 'label' columns for features
    y_train = train_data['label']  # Use the 'label' column as the label

    # Load test data
    test_data = pd.read_csv(test_path)
    test_data.fillna(0, inplace=True)
    X_test = test_data.drop(['id', 'label', 'smi'], axis=1)
    y_test = test_data['label']

    return X_train, y_train, X_test, y_test

def evaluate(y_test, y_pred, y_pred_prob):
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f'Accuracy: {accuracy}, ROC AUC: {roc_auc}')
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()