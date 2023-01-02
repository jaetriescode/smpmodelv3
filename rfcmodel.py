# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 06:31:13 2022

@author: Jae Lee
"""

# In[1]:

## Pre-import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pickle


warnings.filterwarnings("ignore")

dataset = pd.read_excel('smpdata.xlsx')
data = dataset.iloc[ : ,1:9].values
target = dataset.iloc[ : ,11].values


from sklearn.model_selection import train_test_split
Xtrain_o, Xtest_o, Ytrain, Ytest = train_test_split(data,target,test_size=0.2)
test_number = np.array(Ytest.shape)

from sklearn.preprocessing import MinMaxScaler


# Instantiation
scaler = MinMaxScaler()  
Xtrain = scaler.fit_transform(Xtrain_o)  
Xtest = scaler.transform(Xtest_o)  


# In[10]:

## Random Forest Regressor
# Determination of n_estimators
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
Vtr = []
tr = []
te = []
for i in range(20):
    rfr = RandomForestRegressor(criterion="mse"
                                ,random_state=0
                                ,n_estimators=10*i+1
                                ,max_depth=8)
    rfr = rfr.fit(Xtrain, Ytrain)
    rfr_s = cross_val_score(rfr,Xtrain,Ytrain,cv=10).mean()
    score_tr = rfr.score(Xtrain,Ytrain)
    score_te = rfr.score(Xtest,Ytest)
    Vtr.append(rfr_s)
    tr.append(score_tr)
    te.append(score_te)
print(Vtr)
print(max(Vtr))
print(tr)
print(max(tr))
print(te)
print(max(te))
plt.plot(range(1,21),Vtr,color="green",label="Validication Score",linewidth=3)
plt.plot(range(1,21),tr,color="cornflowerblue",label="Train Score",linewidth=3)
plt.plot(range(1,21),te,color="Red",label="Test Score",linewidth=3)
plt.xlabel("n_estimators",fontsize=18)
plt.ylabel("Score",fontsize=18)
plt.title("Random Forest Regressor",fontsize=18)
plt.xticks(range(0,21,2))
plt.yticks(np.linspace(0.8,1,5))
plt.legend(fontsize=18)
plt.show()


# In[11]:

# Optimised model
# from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor(criterion="mse"
                                ,random_state=0
                                ,n_estimators=102
                                ,max_depth=8)
rfc = rfc.fit(Xtrain, Ytrain)
rfc_score = cross_val_score(rfc,Xtrain,Ytrain,cv=10)
rfc_s = cross_val_score(rfc,Xtrain,Ytrain,cv=10).mean()
Ytrain_RF = rfc.predict(Xtrain)
Ytest_RF = rfc.predict(Xtest)
plt.scatter(Ytest,Ytest_RF)
plt.plot(Ytest,Ytest,color='r')
plt.xlabel("Failure load (kN)",fontsize=18)
plt.ylabel("Predicted failure load (kN)",fontsize=18)
plt.title("Random Forest Regressor",fontsize=18)
plt.show()

# Assessment
from sklearn.metrics import r2_score #Coefficient of determination R2
from sklearn.metrics import mean_squared_error #Mean squared error MSE
from sklearn.metrics import mean_absolute_error #Mean absolute error MAE
import statistics #COV
RF_train_acc_R2 = r2_score(Ytrain, Ytrain_RF) #Coefficient of determination R2
RF_test_acc_R2 = r2_score(Ytest, Ytest_RF)
print('R2_RF on train set:{},R2_RF on test set:{}'.format(RF_train_acc_R2,RF_test_acc_R2))
RF_test_acc_MSE = mean_squared_error(Ytest, Ytest_RF) #Mean squared error MSE
RF_train_acc_MSE = mean_squared_error(Ytrain, Ytrain_RF)
print('MSE_RF on test set:{}'.format(RF_test_acc_MSE))
RF_test_acc_MAE = mean_absolute_error(Ytest, Ytest_RF) #Mean absolute error MAE
RF_train_acc_MAE = mean_absolute_error(Ytrain, Ytrain_RF)
print('MAE_RF on test set:{}'.format(RF_test_acc_MAE))
RF_test_acc_mean=np.sum(Ytest/Ytest_RF)/len(Ytest) #Mean ratio
RF_train_acc_mean=np.sum(Ytrain/Ytrain_RF)/len(Ytrain)
print('Mean_RF on test set:{}'.format(RF_test_acc_mean))
RF_test_acc_cov=statistics.stdev(Ytest/Ytest_RF)/RF_test_acc_mean #COV
RF_train_acc_cov=statistics.stdev(Ytrain/Ytrain_RF)/RF_train_acc_mean
print('COV_RF on test set:{}'.format(RF_test_acc_cov))

rfcmodel=rfc.fit(Xtrain, Ytrain)
pickle.dump(rfcmodel,open('rfc_model.pkl','wb'))

