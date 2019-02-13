"""Colin Warn - House Prices Project Kaggle Competition"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

DATA = pd.read_csv("D:/Machine Learning/Kaggle/house-prices-advanced-regression-techniques/train.csv")

correlationMatrix = DATA.corr()
print(correlationMatrix["SalePrice"].sort_values(ascending=False))

DATA = DATA.dropna(axis=1)

X = DATA.drop(["SalePrice", "Id"], axis=1)
y = DATA["SalePrice"]


X = DATA[["OverallQual","GrLivArea", "BedroomAbvGr","Fireplaces","YearBuilt", "YearRemodAdd","FullBath"]]

#TEST DATA
TEST = pd.read_csv("D:/Machine Learning/Kaggle/house-prices-advanced-regression-techniques/test.csv")
TEST = TEST.dropna(axis=1)


testX = TEST[["OverallQual","GrLivArea", "BedroomAbvGr","Fireplaces","YearBuilt", "YearRemodAdd","FullBath"]]

# Linear Regression Fit Data
linReg = LinearRegression()
linReg.fit(X,y)

prediction = linReg.predict(testX)
# Tree Regression

treeReg = DecisionTreeRegressor()
treeReg.fit(X,y)

treePrediction = treeReg.predict(testX)

#Y has one more example than predictions, drop last example
y = y.iloc[1:]


def RMSE(y, prediction):
    accuracy = mean_squared_error(prediction, y)
    accuracy = np.sqrt(accuracy)
    return accuracy


#print(y)
#print(prediction)
print(RMSE(y, prediction))
print(RMSE(y, treePrediction))


submission = pd.DataFrame(prediction)
ysub = pd.DataFrame(y)
#print(submission)
submission.to_csv("D:/Machine Learning/Kaggle/house-prices-advanced-regression-techniques/submission.csv", index=False)
ysub.to_csv("D:/Machine Learning/Kaggle/house-prices-advanced-regression-techniques/ysubmission.csv", index=False)


#PRINTS FIRST SECOND COLUMN OF DATA
#print(X.values[:,1])






