import pandas as pd 
import numpy as np
import os
from sklearn.linear_model import LinearRegression

run=100 # number of experiment repetitions

y1_pred=np.zeros(run)
y2_pred=np.zeros(run)

for runi in np.arange(0,run):
    
    np.random.seed(runi)
    
    df1 = pd.read_csv('./data/ECtraindata1/ECtraindata1' + '_{}.csv'.format(runi+1),header=None)
    X1_train=df1.iloc[:,0:15]
    y1_train=df1.iloc[:,15]
    df2 = pd.read_csv('./data/ECtraindata2/ECtraindata2' + '_{}.csv'.format(runi+1),header=None)
    X2_train=df2.iloc[:,0:15]
    y2_train=df2.iloc[:,15]   
    X_test=pd.read_csv('./data/ECPX/ECPX.csv',header=None)
                                  
    ############################### Linear Regression Metamodel ############################################# 

    # train Y1 | X
    lr1 = LinearRegression()
    lr1.fit(X1_train, y1_train)
    
    # predict mean
    y1_pred[runi] = lr1.predict(X_test)[0]

    # train Y2 | X
    lr2 = LinearRegression()
    lr2.fit(X2_train, y2_train)
    
    # predict mean
    y2_pred[runi] = lr2.predict(X_test)[0]

    
os.makedirs("./data/LinearRegression", exist_ok=True)    
y1_pred=pd.DataFrame(y1_pred) 
y1_pred.to_csv("./data/LinearRegression/y1_pred.csv",index=0)

y2_pred=pd.DataFrame(y2_pred) 
y2_pred.to_csv("./data/LinearRegression/y2_pred.csv",index=0)