import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LinearRegression

n_set=(np.arange(2,17)*10)**2
m_set=(np.arange(2,17)*10)

run=100 # number of experiment repetitions

K=100000 # number of online generation data

d = 2 # dimensions of covariates
dd = 10 # dimensions of covariates after transformation
p = 3 # degree of polynomial transformation

x1lb=0;
x1ub=10;
x2lb=-5;
x2ub=5;
# range of covariates

testx1=7
testx2=2 

ctestx=np.zeros((1,2))
ctestx[:,0]=testx1
ctestx[:,1]=testx2
poly=PolynomialFeatures(degree=p,include_bias=True)
ctestpx=poly.fit_transform(ctestx)

def laplacefun(x, mu, lambda_):
    pdf = (1/(2*lambda_)) * np.e**(-1*(np.abs(x-mu)/lambda_))
    return pdf

def g1fun(x1,x2):
    g1=0.05*x1*x2
    return g1 

def g2fun(x1,x2):
    g2=(5*(np.sin(x1+x2)**2)+5)
    return g2 


def QRGMM_xstar(px,k): # QRGMM online algorithm: input specified covariates (1*dd), output sample vector (k*1)
    quantile_curve=np.reshape(np.dot(nmodels[:,0:dd],px.T),-1)
    quantile_curve_augmented=np.zeros(m+1)
    quantile_curve_augmented[0]=quantile_curve[0]
    quantile_curve_augmented[1:m]=quantile_curve
    quantile_curve_augmented[m]=quantile_curve[-1]
    u=np.random.rand(k)
    order=u//le
    order=order.astype(np.uint64)
    alpha=u-order*le
    q1=quantile_curve_augmented[order]
    q2=quantile_curve_augmented[order+1]
    q=q1*(1-alpha/le)+q2*(alpha/le)
    return q

KS_QRGMM=np.zeros((15,run))


for i in np.arange(0,15):
    m=m_set[i] # fedelity parameter in QRGMM
    n=n_set[i] # number of training data
    
    le=1/m
    ue=1-le
    quantiles=np.linspace(le, ue, m-1) # quantile levels in QRGMM
    
    for runi in np.arange(0,run):
        np.random.seed(i*run+runi)

        ################################ QRGMM ###################################
        models = pd.read_csv("./data_convergence/QRGMMcoeff/QRGMMcoeff" + '_{}_{}.csv'.format(i+1,runi+1),index_col=False)# offline training of QRGMM 
        nmodels=models.to_numpy()
        # A trick to accelerate online generating speed: expand the quantile regression matrix with beta_1 and beta_m-1, 
        # thus no need to use "if else" operation in online interpolation, 
        # then we can entirely use matrix computation to fast generate data onpyline.
        fastmodels=np.zeros((np.shape(nmodels)[0]+2,np.shape(nmodels)[1]))
        fastmodels[0,:]=nmodels[0,:]
        fastmodels[1:np.shape(nmodels)[0]+1,:]=nmodels[:,:]
        fastmodels[np.shape(nmodels)[0]+1,:]=nmodels[-1,:]  
                     
                     
        test_data_QRGMM=QRGMM_xstar(ctestpx,K) # online generating conditional test samples using QRGMM
        KS_QRGMM[i,runi],_=stats.kstest(test_data_QRGMM, 'laplace',args=(g1fun(testx1,testx2),g2fun(testx1,testx2)))
                                              
KS_QRGMM = pd.DataFrame(KS_QRGMM)   
os.makedirs("./data_convergence", exist_ok=True)                                                          
KS_QRGMM.to_csv("./data_convergence/KS_QRGMM_TestProblem_2.csv",index=0)    

