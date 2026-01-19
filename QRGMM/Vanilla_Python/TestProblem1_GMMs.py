import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Prevent kernel suddenly interrupted caused by some unknown problem caused by CWGAN
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import time
from scipy.stats import wasserstein_distance
import random
import torch
from utils import *
from joblib import Parallel, delayed
import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning
import wgan  # Load the wgan python file in the current directory (recommended, as it is more convenient, 
             # facilitates reproducibility, and avoids potential issues that may arise during installation), 
             # or install the package if needed.


run=100 # number of experiment repetitions

d = 3 # dimensions of covariates
n = 10000 # number of training data
m = 300 # number of quantile levels in QRGMM
le=1/m
ue=1-le
quantiles = np.linspace(le, ue, m-1) # quantile levels in QRGMM

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

# covariates of conditionanl test data, x0=(1, 4,-1,3)
x0_1=4
x0_2=-1
x0_3=3

k=100000

def fit_model(q): # quantile regression
    warnings.filterwarnings("ignore", category=IterationLimitWarning)
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['A'],res.params['B'],res.params['C']] 


def QRGMM(X): # QRGMM online algorithm: input covariates matrix (k*d), output sample vector (k*1)
    output_size=np.shape(X)[0]
    u=np.random.rand(output_size)
    order=u//le
    order=order.astype(np.uint64)
    alpha=u-order*le
    b1=fastmodels[order,1:(d+2)]
    b2=fastmodels[order+1,1:(d+2)]
    b=b1*(1-np.tile(alpha.reshape(len(alpha),1),d+1)/le)+b2*np.tile(alpha.reshape(len(alpha),1),d+1)/le          
    sample=np.sum(b*X,1)
    return sample

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

class DiffRectDatasetFromDF:
    """
    Adapt the DataFrame df (columns A, B, C, F) to the format required by the Dataset class, specifically:
    x_train: a standardized torch tensor of shape (n, 3)
    y_train: a standardized torch tensor of shape (n, 1)
    Additionally, save x_mean, x_std, y_mean, and y_std to apply consistent standardization during testing and for denormalization.
    """
    def __init__(self, df, device="cpu"):
        x = df[["A","B","C"]].to_numpy(dtype=np.float32)
        y = df[["F"]].to_numpy(dtype=np.float32)  # (n,1)

        self.x_mean = x.mean(axis=0)
        self.x_std  = x.std(axis=0)
        self.x_std[self.x_std == 0.0] = 1.0
        x_norm = (x - self.x_mean) / self.x_std

        self.y_mean = float(y.mean())
        self.y_std  = float(y.std())
        if self.y_std == 0.0:
            self.y_std = 1.0
        y_norm = (y - self.y_mean) / self.y_std

        self.x_train = torch.as_tensor(x_norm, dtype=torch.float32)
        self.y_train = torch.as_tensor(y_norm, dtype=torch.float32)  # (n,1)
        self.device = device


time_records = []   # list of dicts -> DataFrame
for runi in np.arange(0,run):
    
    random.seed(int(runi))
    np.random.seed(int(runi))
    torch.manual_seed(int(runi))
    torch.cuda.manual_seed(int(runi))
    torch.cuda.manual_seed_all(int(runi))  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
    for i in np.arange(0,n):
        F[i,0]=x1[i]
        F[i,1]=x2[i]
        F[i,2]=x3[i]
        F[i,3]=np.random.normal(g1[i],g2[i])
    df = pd.DataFrame(F, columns=list('A''B''C''F')) # training data
    os.makedirs("./data/traindata", exist_ok=True)
    df.to_csv("./data/traindata/traindata" + '_{}.csv'.format(runi+1),index=0) 
    
    testu1=np.random.rand(n)
    testx1=x1lb+(x1ub-x1lb)*testu1
    testu2=np.random.rand(n)
    testx2=x2lb+(x2ub-x2lb)*testu2
    testu3=np.random.rand(n)
    testx3=x3lb+(x3ub-x3lb)*testu3
    testg1=a0+a1*testx1+a2*testx2+a3*testx3
    testg2=r0+r1*testx1+r2*testx2+r3*testx3
    
    testF=np.zeros((n,4))
    for i in np.arange(0,n):
        testF[i,0]=testx1[i]
        testF[i,1]=testx2[i]
        testF[i,2]=testx3[i]
        testF[i,3]=np.random.normal(testg1[i],testg2[i])
    testdf = pd.DataFrame(testF, columns=list('A''B''C''F')) # test data
    os.makedirs("./data/testdata", exist_ok=True)
    testdf.to_csv("./data/testdata/testdata" + '_{}.csv'.format(runi+1),index=0) 

    ctestX=np.zeros((100000,d+1))
    ctestX[:,0]=1
    ctestX[:,1]=x0_1
    ctestX[:,2]=x0_2
    ctestX[:,3]=x0_3
    ctestx = ctestX[0,:]
    
    testdataX=np.zeros((n,d+1))
    testdataX[:,0]=1
    testdataX[:,1]=testx1
    testdataX[:,2]=testx2
    testdataX[:,3]=testx3

    ctestF=np.zeros((100000,4))
    ctestF[:,0:3]=ctestX[:,1:4]
    ctestdf = pd.DataFrame(ctestF, columns=list('A''B''C''F')) 

    X_ctest_raw = ctestdf[["A","B","C"]].to_numpy(dtype=np.float32)
    X_test_raw = testdf[["A","B","C"]].to_numpy(dtype=np.float32)

    ################################ QRGMM ###################################

    t0 = time.perf_counter()
    # offline training of QRGMM 

    mod = smf.quantreg('F ~ A + B + C', df)

    n_jobs = min(len(quantiles), os.cpu_count()//2 or 1) # Quantile regression can be computed in parallel very naturally.
    models_list = Parallel(n_jobs=n_jobs, backend="loky")(delayed(fit_model)(q) for q in quantiles)

    # models_list = [fit_model(x) for x in quantiles] # non-parallel version

    models = pd.DataFrame(models_list, columns=['q', 'b0', 'b1','b2','b3'])
    nmodels = models.to_numpy()

    # A trick to accelerate online generating speed: expand the quantile regression matrix with beta_1 and beta_m-1, 
    # thus no need to use "if else" operation in online interpolation, 
    # then we can entirely use matrix computation to fast generate data online.
    fastmodels=np.zeros((np.shape(nmodels)[0]+2,np.shape(nmodels)[1]))
    fastmodels[0,:]=nmodels[0,:]
    fastmodels[1:np.shape(nmodels)[0]+1,:]=nmodels[:,:]
    fastmodels[np.shape(nmodels)[0]+1,:]=nmodels[-1,:] 
    
    t1 = time.perf_counter() 
    ctest_data_QRGMM = QRGMM_xstar(ctestx,k) # online generating conditional test samples using QRGMM
    t2 = time.perf_counter()
    
    ctest_data_QRGMM = pd.DataFrame(ctest_data_QRGMM) 
    os.makedirs("./data/ctestdata_QRGMM", exist_ok=True)
    ctest_data_QRGMM.to_csv("./data/ctestdata_QRGMM/ctestdata_QRGMM" + '_{}.csv'.format(runi+1),index=0)
    
    
    test_data_QRGMM = QRGMM(testdataX) # online generating unconditional test samples using QRGMM
    test_data_QRGMM = pd.DataFrame(test_data_QRGMM) 
    os.makedirs("./data/testdata_QRGMM", exist_ok=True)
    test_data_QRGMM.to_csv("./data/testdata_QRGMM/testdata_QRGMM" + '_{}.csv'.format(runi+1),index=0)

    qrgmm_train = t1 - t0
    qrgmm_gen   = t2 - t1
    time_records.append({
        "run": int(runi+1),
        "model": "QRGMM",
        "train_time": qrgmm_train,
        "gen_time": qrgmm_gen,
    })
    print(f"[run {runi+1}] QRGMM train time: {qrgmm_train:.4f}s, QRGMM gen time: {qrgmm_gen:.4f}s")
    
        
    ################################ CWGAN-GP #############################################    
    random.seed(int(runi) + 100000)
    np.random.seed(int(runi) + 100000)
    torch.manual_seed(int(runi) + 100000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(runi) + 100000)

    t0 = time.perf_counter()    
    # Y | X
    continuous_vars = ["F"]
    categorical_vars = []
    context_vars = ["A", "B", "C"]
    
    # Initialize objects
    data_wrapper = wgan.DataWrapper(df, continuous_vars, categorical_vars, 
                                  context_vars)
    spec = wgan.Specifications(data_wrapper, batch_size=4096, max_epochs=1000, critic_lr=1e-3, generator_lr=1e-3,
                             print_every=100, device = "cuda")
    generator = wgan.Generator(spec)
    critic = wgan.Critic(spec)
    
    # train Y | X
    y, context = data_wrapper.preprocess(df)
    wgan.train(generator, critic, y, context, spec)
    t1= time.perf_counter()

    # generate conditional test samples using CWGAN
    ctest_data_CWGAN = data_wrapper.apply_generator(generator, ctestdf)
    t2 = time.perf_counter()
    os.makedirs("./data/ctestdata_CWGAN", exist_ok=True)
    ctest_data_CWGAN.to_csv("./data/ctestdata_CWGAN/ctestdata_CWGAN" + '_{}.csv'.format(runi+1),index=0)
    
    # generate unconditional test samples using CWGAN
    test_data_CWGAN = data_wrapper.apply_generator(generator, testdf)
    os.makedirs("./data/testdata_CWGAN", exist_ok=True)
    test_data_CWGAN.to_csv("./data/testdata_CWGAN/testdata_CWGAN" + '_{}.csv'.format(runi+1),index=0)

    cwgan_train = t1 - t0
    cwgan_gen   = t2 - t1
    time_records.append({
        "run": int(runi+1),
        "model": "CWGAN",
        "train_time": cwgan_train,
        "gen_time": cwgan_gen,
    })
    print(f"[run {runi+1}] CWGAN train time: {cwgan_train:.4f}s, CWGAN gen time: {cwgan_gen:.4f}s")


    ################################ Diffusion ###################################
    random.seed(int(runi) + 200000)
    np.random.seed(int(runi) + 200000)
    torch.manual_seed(int(runi) + 200000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(runi) + 200000)

    diffrect_device = "cuda" if torch.cuda.is_available() else "cpu"


    t0 = time.perf_counter()
    data_dr = DiffRectDatasetFromDF(df, device=diffrect_device)
    args_dr = get_args(data_type='TestProblem1')
    args_dr['data_size'] = n
    args_dr['instance'] = f"TestProblem1_run{runi+1}_n{n}"
    ensure_instance_dirs(args_dr['instance'])

    train_all(data_dr, args_dr, model_type='diffusion')
    t1 = time.perf_counter()

    
    y_ctest_diff = generate(X_ctest_raw, data_dr, args_dr, model_name='diffusion')
    t2 = time.perf_counter()

    ctestdf_diff = ctestdf.copy()
    ctestdf_diff["F"] = y_ctest_diff
    os.makedirs("./data/ctestdata_Diffusion", exist_ok=True)
    ctestdf_diff.to_csv(f"./data/ctestdata_Diffusion/ctestdata_Diffusion_{runi+1}.csv", index=0)
    
    
    y_diff = generate(X_test_raw, data_dr, args_dr, model_name='diffusion')
    testdf_diff = testdf.copy()
    testdf_diff["F"] = y_diff
    os.makedirs("./data/testdata_Diffusion", exist_ok=True)
    testdf_diff.to_csv(f"./data/testdata_Diffusion/testdata_Diffusion_{runi+1}.csv", index=0)

    diff_train = t1 - t0
    diff_gen   = t2 - t1
    time_records.append({
        "run": int(runi+1),
        "model": "Diffusion",
        "train_time": diff_train,
        "gen_time": diff_gen,
    })
    print(f"[run {runi+1}] diffusion train time: {diff_train:.4f}s, diffusion gen time: {diff_gen:.4f}s")

    ################################  Rectified Flow ###################################
    random.seed(int(runi) + 300000)
    np.random.seed(int(runi) + 300000)
    torch.manual_seed(int(runi) + 300000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(runi) + 300000)


    diffrect_device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    data_dr = DiffRectDatasetFromDF(df, device=diffrect_device)
    args_dr = get_args(data_type='TestProblem1')
    args_dr['data_size'] = n
    args_dr['instance'] = f"TestProblem1_run{runi+1}_n{n}"
    ensure_instance_dirs(args_dr['instance'])

    train_all(data_dr, args_dr, model_type='rectified')
    t1 = time.perf_counter()

    y_ctest_rect = generate(X_ctest_raw, data_dr, args_dr, model_name='rectified')
    t2 = time.perf_counter()

    ctestdf_rect = ctestdf.copy()
    ctestdf_rect["F"] = y_ctest_rect
    os.makedirs("./data/ctestdata_RectFlow", exist_ok=True)
    ctestdf_rect.to_csv(f"./data/ctestdata_RectFlow/ctestdata_RectFlow_{runi+1}.csv", index=0)

    y_rect = generate(X_test_raw, data_dr, args_dr, model_name='rectified')
    testdf_rect = testdf.copy()
    testdf_rect["F"] = y_rect
    os.makedirs("./data/testdata_RectFlow", exist_ok=True)
    testdf_rect.to_csv(f"./data/testdata_RectFlow/testdata_RectFlow_{runi+1}.csv", index=0)

    rect_train = t1 - t0
    rect_gen   = t2 - t1
    time_records.append({
        "run": int(runi+1),
        "model": "RectFlow",
        "train_time": rect_train,
        "gen_time": rect_gen,
    })
    print(f"[run {runi+1}] rectflow train time: {rect_train:.4f}s, rectflow gen time: {rect_gen:.4f}s")

df_time = pd.DataFrame(time_records)
os.makedirs("./results", exist_ok=True)
df_time.to_csv("./results/timing_all_models.csv", index=False)


summary = df_time.groupby("model")[["train_time", "gen_time"]].agg(
    ['mean', 'std']
)
print("\nSummary time:\n", summary)

  