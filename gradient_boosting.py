import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import os

script_dir = os.path.dirname(__file__)
abs_file_path = os.path.join(script_dir, "Colony Count.xlsx")
os.path.normpath(abs_file_path)

df = pd.read_excel(abs_file_path, sheet_name='Sheet1')
# print(df.values)
X = df.values[:,range(2,4)]
y = df.values[:,1]
X_train, X_test, y_train, y_test = X,X,y,y

params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)


test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = reg.loss_(y_test, y_pred)

feature_importance = reg.feature_importances_
print("the feature importance is", feature_importance)
pred = X[:,0]*feature_importance[0] + X[:,1]*feature_importance[1]
mse = mean_squared_error(y, pred)
print("The boosting prediction is", pred)
print("The mean squared error (MSE) on test set is: {:.4f}".format(mse))

sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, ['kmeans', 'CNN'])
plt.title('Feature Importance (MDI)')
plt.show()

#Write result to array
df = pd.DataFrame(pred)
abs_file_path = os.path.join(script_dir, "BoostingResult.xlsx")
os.path.normpath(abs_file_path)
df.to_excel(excel_writer = abs_file_path)
