import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import wasserstein_distance


run=100 # number of experiment repetitions

d = 2 # dimensions of covariates
dd = 10 # dimensions of covariates after transformation
p = 3 # degree of polynomial transformation
n = 10000 # number of training data
m = 300 # number of quantile levels in QRGMM

le=1/m
ue=1-le
quantiles = np.linspace(le, ue, m-1) # quantile levels in QRGMM

x1lb=0;
x1ub=10;
x2lb=-5;
x2ub=5;
# range of covariates



# Generate data for experiments
for runi in np.arange(0,run):
    
    np.random.seed(runi)
    
    u1=np.random.rand(n)
    x1=x1lb+(x1ub-x1lb)*u1
    u2=np.random.rand(n)
    x2=x2lb+(x2ub-x2lb)*u2

    g1=0.05*x1*x2
    g2=(5*(np.sin(x1+x2)**2)+5)
    
    X=np.zeros((n,2))
    for i in np.arange(0,n):
        X[i,0]=x1[i]
        X[i,1]=x2[i]
        
    poly = PolynomialFeatures(degree=p,include_bias=True)
    PX=poly.fit_transform(X)
    
    F=np.zeros((n,dd+1))
    for i in np.arange(0,n):
        F[i,0:dd]=PX[i]
        F[i,dd]=np.random.laplace(g1[i],g2[i])
    df = pd.DataFrame(F) # training data
    os.makedirs("./data/traindata", exist_ok=True)
    df.to_csv("./data/traindata/traindata" + "_{}.csv".format(runi+1),index=0,header=None)
    
    testu1=np.random.rand(n)
    testx1=x1lb+(x1ub-x1lb)*testu1
    testu2=np.random.rand(n)
    testx2=x2lb+(x2ub-x2lb)*testu2

    testg1=0.05*testx1*testx2
    testg2=(5*(np.sin(testx1+testx2)**2)+5)
    
    testX=np.zeros((n,2))
    for i in np.arange(0,n):
        testX[i,0]=testx1[i]
        testX[i,1]=testx2[i]
        
    poly = PolynomialFeatures(degree=p,include_bias=True)
    testPX=poly.fit_transform(testX)
    
    testF=np.zeros((n,dd+1))
    for i in np.arange(0,n):
        testF[i,0:dd]=testPX[i]
        testF[i,dd]=np.random.laplace(testg1[i],testg2[i])
    testdf = pd.DataFrame(testF) # training data
    os.makedirs("./data/testdata", exist_ok=True)
    testdf.to_csv("./data/testdata/testdata" + "_{}.csv".format(runi+1),index=0,header=None)
