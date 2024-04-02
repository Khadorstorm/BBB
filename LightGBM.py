import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

train_data = pd.read_csv('train/train_feature.csv')
train_data.fillna(0,inplace=True)
X_train = train_data.drop(['id', 'label','smi'], axis=1)  # Drop the 'id' and 'label' columns for features
y_train = train_data['label']  # Use the 'label' column as the label

# Load test data
test_data = pd.read_csv('test/test_feature.csv')
test_data.fillna(0,inplace=True)
X_test = test_data.drop(['id', 'label', 'smi'], axis=1)
y_test = test_data['label']

# Prepare the LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Set the parameters for the model
params = {
    'objective': 'binary',  # Use 'multiclass' for multi-class classification
    'metric': 'binary_logloss',  # Use 'multi_logloss' for multi-class classification
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}

# Train the model
gbm = lgb.train(params, train_data, num_boost_round=100)

# Predict on test set
y_pred_prob = gbm.predict(X_test)
y_pred = np.round(y_pred_prob)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f'Accuracy: {accuracy}, ROC AUC: {roc_auc}')


# Prepare the LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Set the parameters for the model
params = {
    'objective': 'binary',  # Use 'multiclass' for multi-class classification
    'metric': 'binary_logloss',  # Use 'multi_logloss' for multi-class classification
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}

# Train the model
gbm = lgb.train(params, train_data, num_boost_round=100)

# Predict on test set
y_pred_prob = gbm.predict(X_test)
y_pred = np.round(y_pred_prob)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f'Accuracy: {accuracy}, ROC AUC: {roc_auc}')


# Compute ROC curve and ROC area
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