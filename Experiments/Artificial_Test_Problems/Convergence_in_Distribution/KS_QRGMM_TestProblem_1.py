import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from joblib import Parallel, delayed
import os
import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning

n_set=(np.arange(2,17)*10)**2
m_set=(np.arange(2,17)*10)

run=100 # number of experiment repetitions
d = 3 # dimensions of covariates

K=100000

x1lb=0;
x1ub=10;
x2lb=-5;
x2ub=5;
x3lb=0;
x3ub=5;
# range of covariates

a0=5;
a1=1;
a2=2;
a3=0.5;
r0=1;
r1=0.1;
r2=0.2;
r3=0.05;
# example coefficients

def fit_model(q): # quantile regression
    warnings.filterwarnings("ignore", category=IterationLimitWarning)
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['A'],res.params['B'],res.params['C']] 

def QRGMM_xstar(x,k): # QRGMM online algorithm: input specified covariates (1*(d+1)), output sample vector (k*1)
    quantile_curve=np.reshape(np.dot(nmodels[:,1:(d+2)],x.T),-1)
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

def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf
def g1fun(x_1,x_2,x_3):
    g1=a0+a1*x_1+a2*x_2+a3*x_3
    return g1 
def g2fun(x_1,x_2,x_3):
    g2=r0+r1*x_1+r2*x_2+r3*x_3
    return g2 

# covariates of test data, x0=(6,1,2)
x0_1=6
x0_2=1
x0_3=2 

ctestx = np.zeros((1, d+1))
ctestx[:, 0] = 1
ctestx[:, 1] = x0_1
ctestx[:, 2] = x0_2
ctestx[:, 3] = x0_3

KS_QRGMM=np.zeros((15,run))

for i in np.arange(0,15):
    m=m_set[i] # fedelity parameter in QRGMM
    n=n_set[i] # number of training data
    print(f"\n▶ Data size {i+1}/15 | n = {n}")
    
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
        u3=np.random.rand(n)
        x3=x3lb+(x3ub-x3lb)*u3
        g1=a0+a1*x1+a2*x2+a3*x3
        g2=r0+r1*x1+r2*x2+r3*x3
        
        F=np.zeros((n,4))
        for j in np.arange(0,n):
            F[j,0]=x1[j]
            F[j,1]=x2[j]
            F[j,2]=x3[j]
            F[j,3]=np.random.normal(g1[j],g2[j])
        df = pd.DataFrame(F, columns=list('A''B''C''F')) # training data
        ################################ QRGMM ###################################

        mod = smf.quantreg('F ~ A + B + C', df) # offline training of QRGMM 
        n_jobs = min(len(quantiles), os.cpu_count()//2 or 1) # Quantile regression can be computed in parallel very naturally.
        models_list = Parallel(n_jobs=n_jobs, backend="loky")(delayed(fit_model)(q) for q in quantiles)
        models = pd.DataFrame(models_list, columns=['q', 'b0', 'b1','b2','b3'])
        nmodels = models.to_numpy()
    

        test_data_QRGMM=QRGMM_xstar(ctestx,K) # online application(generating conditional test samples) of QRGMM
        KS_QRGMM[i,runi],_=stats.kstest(test_data_QRGMM, 'norm',args=(g1fun(x0_1,x0_2,x0_3),g2fun(x0_1,x0_2,x0_3)))
 
                                                           
KS_QRGMM = pd.DataFrame(KS_QRGMM)      
os.makedirs("./data_convergence", exist_ok=True)                                                      
KS_QRGMM.to_csv("./data_convergence/KS_QRGMM_TestProblem_1.csv",index=0)    