import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


n_set=(np.arange(2,17)*10)**2
m_set=(np.arange(2,17)*10)

run=100 # number of experiment repetitions

d = 2 # dimensions of covariates
dd = 10 # dimensions of covariates after transformation
p = 3 # degree of polynomial transformation

x1lb=0;
x1ub=10;
x2lb=-5;
x2ub=5;
# range of covariates


for i in np.arange(0,15):
    m=m_set[i] # fedelity parameter in QRGMM
    n=n_set[i] # number of training data
    
    le=1/m
    ue=1-le
    quantiles=np.linspace(le, ue, m-1) # quantile levels in QRGMM
    
    for runi in np.arange(0,run):
        np.random.seed(i*run+runi)
        ############################### generate data ###############################
        u1=np.random.rand(n)
        x1=x1lb+(x1ub-x1lb)*u1
        u2=np.random.rand(n)
        x2=x2lb+(x2ub-x2lb)*u2

        g1=0.05*x1*x2
        g2=(5*(np.sin(x1+x2)**2)+5)
    
        X=np.zeros((n,2))
        for j in np.arange(0,n):
            X[j,0]=x1[j]
            X[j,1]=x2[j]
        
        poly = PolynomialFeatures(degree=p,include_bias=True)
        PX=poly.fit_transform(X)
    
        F=np.zeros((n,dd+1))
        for j in np.arange(0,n):
            F[j,0:dd]=PX[j]
            F[j,dd]=np.random.laplace(g1[j],g2[j])
        df = pd.DataFrame(F) # training data
        os.makedirs("./data_convergence/traindata", exist_ok=True)
        df.to_csv("./data_convergence/traindata/traindata" + '_{}_{}.csv'.format(i+1,runi+1),index=0)