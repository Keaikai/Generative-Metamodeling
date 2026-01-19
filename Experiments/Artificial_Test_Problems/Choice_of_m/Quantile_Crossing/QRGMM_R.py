import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import wasserstein_distance
import time

run=100 # number of experiment repetitions

n = 10000 # number of training data
m = 300 # number of quantile levels in QRGMM
d = 2 # dimensions of covariates
dd = 10 # dimensions of covariates after transformation
p = 3 # degree of polynomial transformation

le=1/m
ue=1-le
quantiles = np.linspace(le, ue, m-1) # quantile levels in QRGMM

x1lb=0;
x1ub=10;
x2lb=-5;
x2ub=5;
# range of covariates

K=100000

testx1=4
testx2=4

def laplacefun(x, mu, lambda_):  # pdf of the laplace distribution 
    pdf = (1/(2*lambda_)) * np.e**(-1*(np.abs(x-mu)/lambda_))
    return pdf

def g1fun(x1,x2): # mean function of Scenario 2
    g1=0.05*x1*x2
    return g1 

def g2fun(x1,x2): # standard deviation function of Scenario 2
    g2=(5*(np.sin(x1+x2)**2)+5)
    return g2 

def QRGMM(PX,K): # QRGMM online algorithm: input specified covariates (1*dd), output sample vector (K*1)
    quantile_curve=np.reshape(np.dot(nmodels[:,0:dd],PX.T),-1)
    quantile_curve_augmented=np.zeros(m+1)
    quantile_curve_augmented[0]=quantile_curve[0]
    quantile_curve_augmented[1:m]=quantile_curve
    quantile_curve_augmented[m]=quantile_curve[-1]
    u=np.random.rand(K)
    order=u//le
    order=order.astype(np.uint64)
    alpha=u-order*le
    q1=quantile_curve_augmented[order]
    q2=quantile_curve_augmented[order+1]
    q=q1*(1-alpha/le)+q2*(alpha/le)
    return q

def QRGMM_R(PX,K): # QRGMM-R online algorithm: input specified covariates (1*dd), output sample vector (K*1)
    quantile_curve=np.reshape(np.dot(nmodels[:,0:dd],PX.T),-1)
    sorted_quantile_curve=np.sort(quantile_curve)
    quantile_curve_augmented=np.zeros(m+1)
    quantile_curve_augmented[0]=sorted_quantile_curve[0]
    quantile_curve_augmented[1:m]=sorted_quantile_curve
    quantile_curve_augmented[m]=sorted_quantile_curve[-1]
    u=np.random.rand(K)
    order=u//le
    order=order.astype(np.uint64)
    alpha=u-order*le
    q1=quantile_curve_augmented[order]
    q2=quantile_curve_augmented[order+1]
    q=q1*(1-alpha/le)+q2*(alpha/le)
    return q

onlinetime=np.zeros((run,2))
crossing_rate=np.zeros(run)

for runi in np.arange(0,run):
    
    np.random.seed(runi)
    
    testg1=0.05*testx1*testx2
    testg2=(5*(np.sin(testx1+testx2)**2)+5)
    
    testX=np.zeros((K,2))
    testX[:,0]=testx1
    testX[:,1]=testx2
     
    poly = PolynomialFeatures(degree=p,include_bias=True)
    testPX=poly.fit_transform(testX)
    
    testF=np.zeros((K,dd+1))
    testF[:,0:dd]=testPX
    testF[:,dd]=np.random.laplace(testg1,testg2,K)
    testdf = pd.DataFrame(testF) # test data
    os.makedirs("./data_QC/test_data", exist_ok=True)
    testdf.to_csv("./data_QC/test_data/test_data" + "_{}.csv".format(runi+1),index=0,header=None)


for runi in np.arange(0,run):
    

    ############################### prepare data ###############################
    testX=np.zeros((1,2))
    testX[:,0]=testx1
    testX[:,1]=testx2
    poly=PolynomialFeatures(degree=p,include_bias=True)
    testPX=poly.fit_transform(testX)

    models = pd.read_csv(
    "../../Test_Problem_2_Performance/data/QRGMMcoeff/QRGMMcoeff_{}.csv".format(runi + 1),
    index_col=False) # offline training of QRGMM 
    nmodels=models.to_numpy()

    ################################ QRGMM-R ############################################# 
    np.random.seed(0)
    _=QRGMM_R(testPX,1000) # warm up

    np.random.seed(runi)
    time_start=time.perf_counter() 
    test_data_QRGMM_R=QRGMM_R(testPX,K) # online generating conditional test samples using QRGMM-R
    time_end=time.perf_counter() 
    onlinetime[runi,0]=time_end - time_start 
    test_data_QRGMM_R = pd.DataFrame(test_data_QRGMM_R)
    os.makedirs("./data_QC/test_data_QRGMM_R", exist_ok=True)
    test_data_QRGMM_R.to_csv("./data_QC/test_data_QRGMM_R/test_data_QRGMM_R" + '_{}.csv'.format(runi+1),index=0)
    
    quantile_curve=np.reshape(np.dot(nmodels[:,0:dd],testPX.T),-1)
    crossing_rate[runi]=sum(quantile_curve!=np.sort(quantile_curve))/(m-1) 

    ################################ QRGMM #############################################
    np.random.seed(0)
    _=QRGMM(testPX,1000) # warm up

    np.random.seed(runi)
    time_start=time.perf_counter() 
    test_data_QRGMM=QRGMM(testPX,K) # online generating conditional test samples using QRGMM
    time_end=time.perf_counter() 
    onlinetime[runi,1]=time_end - time_start 
    test_data_QRGMM = pd.DataFrame(test_data_QRGMM)
    os.makedirs("./data_QC/test_data_QRGMM", exist_ok=True)
    test_data_QRGMM.to_csv("./data_QC/test_data_QRGMM/test_data_QRGMM" + '_{}.csv'.format(runi+1),index=0)


onlinetime_df=pd.DataFrame(onlinetime)
onlinetime_df.to_csv("./data_QC/onlinetime.csv",index=0)
crossing_rate_df=pd.DataFrame(crossing_rate)
crossing_rate_df.to_csv("./data_QC/crossing_rate.csv",index=0)


# ============================ summary time ============================

mean_QRGMM_R = np.mean(onlinetime[:, 0])
std_QRGMM_R  = np.std(onlinetime[:, 0], ddof=1)

mean_QRGMM = np.mean(onlinetime[:, 1])
std_QRGMM  = np.std(onlinetime[:, 1], ddof=1)


print("========== Online Time Summary ({} runs) ==========".format(run))
print("QRGMM-R : mean = {:.4e} s, std = {:.4e} s".format(mean_QRGMM_R, std_QRGMM_R))
print("QRGMM   : mean = {:.4e} s, std = {:.4e} s".format(mean_QRGMM, std_QRGMM))
print("===================================================")
