import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Prevent kernel suddenly interrupted caused by some unknown problem caused by CWGAN
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import time
from scipy.stats import wasserstein_distance
import random
import torch
from utils import *
import wgan  # Load the wgan python file in the current directory (recommended, as it is more convenient, 
             # facilitates reproducibility, and avoids potential issues that may arise during installation), 
             # or install the package if needed.

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

ctestx1=4
ctestx2=4 

k=100000

def QRGMM(PX): # QRGMM online algorithm: input covariates matrix (k*dd), output sample vector (k*1)
    output_size=np.shape(PX)[0]
    u=np.random.rand(output_size)
    order=u//le
    order=order.astype(np.uint64)
    alpha=u-order*le
    b1=fastmodels[order,0:dd]
    b2=fastmodels[order+1,0:dd]
    b=b1*(1-np.tile(alpha.reshape(len(alpha),1),dd)/le)+b2*np.tile(alpha.reshape(len(alpha),1),dd)/le          
    sample=np.sum(b*PX,1)
    return sample

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

class DiffRectDatasetFromDF:
    """
    Adapt the DataFrame df (columns 1,2,10) to the format required by the Dataset class, specifically:
    x_train: a standardized torch tensor of shape (n, 2)
    y_train: a standardized torch tensor of shape (n, 1)
    Additionally, save x_mean, x_std, y_mean, and y_std to apply consistent standardization during testing and for denormalization.
    """
    def __init__(self, df, device="cpu"):
        x = df[["1","2"]].to_numpy(dtype=np.float32)
        y = df[["10"]].to_numpy(dtype=np.float32)  # (n,1)

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
    ############################### load data ###############################
    df = pd.read_csv("./data/traindata/traindata" + "_{}.csv".format(runi+1),index_col=False,header=None)
    df2=df[[1,2,10]]
    df2.columns=["1","2","10"]
    
    testdf = pd.read_csv("./data/testdata/testdata" + "_{}.csv".format(runi+1),index_col=False,header=None)
    testdf2=testdf[[1,2,10]]
    testdf2.columns=["1","2","10"]

    testX=testdf.iloc[:,1:3]
    testX=testX.to_numpy() 
    poly=PolynomialFeatures(degree=p,include_bias=True)
    testPX=poly.fit_transform(testX)

    ctestX=np.zeros((100000,2))
    ctestX[:,0]=ctestx1
    ctestX[:,1]=ctestx2
    poly=PolynomialFeatures(degree=p,include_bias=True)
    ctestPX=poly.fit_transform(ctestX)
    ctestpx=ctestPX[0,:]

    ctestF=np.zeros((100000,dd+1))
    ctestF[:,0:dd]=ctestPX[:,:]
    ctestdf=pd.DataFrame(ctestF)
    ctestdf2=ctestdf[[1,2,10]]
    ctestdf2.columns=["1","2","10"]

    X_test_raw = testdf[[1,2]].to_numpy(dtype=np.float32)
    X_ctest_raw = ctestdf[[1,2]].to_numpy(dtype=np.float32)


    ################################ QRGMM ###################################

    models = pd.read_csv("./data/QRGMMcoeff/QRGMMcoeff" + "_{}.csv".format(runi+1),index_col=False)# offline training of QRGMM 
    nmodels=models.to_numpy()
    # A trick to accelerate online generating speed: expand the quantile regression matrix with beta_1 and beta_m-1, 
    # thus no need to use "if else" operation in online interpolation, 
    # then we can entirely use matrix computation to fast generate data onpyline.
    fastmodels=np.zeros((np.shape(nmodels)[0]+2,np.shape(nmodels)[1]))
    fastmodels[0,:]=nmodels[0,:]
    fastmodels[1:np.shape(nmodels)[0]+1,:]=nmodels[:,:]
    fastmodels[np.shape(nmodels)[0]+1,:]=nmodels[-1,:]   

    t1 = time.perf_counter()     
    ctest_data_QRGMM=QRGMM_xstar(ctestpx,k) # online generating conditional test samples using QRGMM
    t2 = time.perf_counter()
    ctest_data_QRGMM = pd.DataFrame(ctest_data_QRGMM)
    os.makedirs("./data/ctestdata_QRGMM", exist_ok=True)
    ctest_data_QRGMM.to_csv("./data/ctestdata_QRGMM/ctestdata_QRGMM" + '_{}.csv'.format(runi+1),index=0)

    test_data_QRGMM=QRGMM(testPX) # online generating unconditional test samples using QRGMM
    test_data_QRGMM=pd.DataFrame(test_data_QRGMM)
    os.makedirs("./data/testdata_QRGMM", exist_ok=True)
    test_data_QRGMM.to_csv("./data/testdata_QRGMM/testdata_QRGMM" + '_{}.csv'.format(runi+1),index=0)

    qrgmm_gen   = t2 - t1
    time_records.append({
        "run": int(runi+1),
        "model": "QRGMM",
        "gen_time": qrgmm_gen,
    })
    print(f"[run {runi+1}] QRGMM gen time: {qrgmm_gen:.4f}s")
        
    ################################ CWGAN-GP #############################################    
    random.seed(int(runi) + 100000)
    np.random.seed(int(runi) + 100000)
    torch.manual_seed(int(runi) + 100000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(runi) + 100000)

    # Y | X
    continuous_vars = ["10"]
    categorical_vars = []
    context_vars = ["1", "2"]
    
    # Initialize objects
    data_wrapper = wgan.DataWrapper(df2, continuous_vars, categorical_vars, 
                                    context_vars)
    spec = wgan.Specifications(data_wrapper, batch_size=4096, max_epochs=1000, critic_lr=1e-3, generator_lr=1e-3,
                                print_every=100, device = "cuda")
    generator = wgan.Generator(spec)
    critic = wgan.Critic(spec)
    
    # train Y | X
    y, context = data_wrapper.preprocess(df2)
    wgan.train(generator, critic, y, context, spec)

    # generate conditional test samples using CWGAN
    t1= time.perf_counter()
    ctest_data_CWGAN = data_wrapper.apply_generator(generator, ctestdf2)
    t2= time.perf_counter()
    os.makedirs("./data/ctestdata_CWGAN", exist_ok=True)
    ctest_data_CWGAN.to_csv("./data/ctestdata_CWGAN/ctestdata_CWGAN" + '_{}.csv'.format(runi+1),index=0)

    # generate unconditional test samples using CWGAN
    test_data_CWGAN = data_wrapper.apply_generator(generator, testdf2)
    os.makedirs("./data/testdata_CWGAN", exist_ok=True)
    test_data_CWGAN.to_csv("./data/testdata_CWGAN/testdata_CWGAN" + '_{}.csv'.format(runi+1),index=0)


    cwgan_gen   = t2 - t1
    time_records.append({
        "run": int(runi+1),
        "model": "CWGAN",
        "gen_time": cwgan_gen,
    })
    print(f"[run {runi+1}] CWGAN gen time: {cwgan_gen:.4f}s")


    ################################ Diffusion ###################################
    random.seed(int(runi) + 200000)
    np.random.seed(int(runi) + 200000)
    torch.manual_seed(int(runi) + 200000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(runi) + 200000)

    diffrect_device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    data_dr = DiffRectDatasetFromDF(df2, device=diffrect_device)
    args_dr = get_args(data_type='TestProblem2')
    args_dr['data_size'] = n
    args_dr['instance'] = f"TestProblem2_run{runi+1}_n{n}"
    ensure_instance_dirs(args_dr['instance'])


    train_all(data_dr, args_dr, model_type='diffusion')

    t1 = time.perf_counter()
    y_ctest_diff = generate(X_ctest_raw, data_dr, args_dr, model_name='diffusion')
    t2 = time.perf_counter()

    ctestdf_diff = ctestdf2.copy()
    ctestdf_diff["10"] = y_ctest_diff
    os.makedirs("./data/ctestdata_Diffusion", exist_ok=True)
    ctestdf_diff.to_csv(f"./data/ctestdata_Diffusion/ctestdata_Diffusion_{runi+1}.csv", index=0)
    

    y_diff = generate(X_test_raw, data_dr, args_dr, model_name='diffusion')
    testdf_diff = testdf2.copy()
    testdf_diff["10"] = y_diff
    os.makedirs("./data/testdata_Diffusion", exist_ok=True)
    testdf_diff.to_csv(f"./data/testdata_Diffusion/testdata_Diffusion_{runi+1}.csv", index=0)

    diff_gen   = t2 - t1
    time_records.append({
        "run": int(runi+1),
        "model": "Diffusion",
        "gen_time": diff_gen,
    })
    print(f"[run {runi+1}] diffusion gen time: {diff_gen:.4f}s")

    ################################  Rectified Flow ###################################
    random.seed(int(runi) + 300000)
    np.random.seed(int(runi) + 300000)
    torch.manual_seed(int(runi) + 300000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(runi) + 300000)


    diffrect_device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dr = DiffRectDatasetFromDF(df2, device=diffrect_device)
    args_dr = get_args(data_type='TestProblem2')
    args_dr['data_size'] = n
    args_dr['instance'] = f"TestProblem2_run{runi+1}_n{n}"
    ensure_instance_dirs(args_dr['instance'])

    train_all(data_dr, args_dr, model_type='rectified')
    
    t1 = time.perf_counter()
    y_ctest_rect = generate(X_ctest_raw, data_dr, args_dr, model_name='rectified')
    t2 = time.perf_counter()

    ctestdf_rect = ctestdf2.copy()
    ctestdf_rect["10"] = y_ctest_rect
    os.makedirs("./data/ctestdata_RectFlow", exist_ok=True)
    ctestdf_rect.to_csv(f"./data/ctestdata_RectFlow/ctestdata_RectFlow_{runi+1}.csv", index=0)

    y_rect = generate(X_test_raw, data_dr, args_dr, model_name='rectified')
    testdf_rect = testdf2.copy()
    testdf_rect["10"] = y_rect
    os.makedirs("./data/testdata_RectFlow", exist_ok=True)
    testdf_rect.to_csv(f"./data/testdata_RectFlow/testdata_RectFlow_{runi+1}.csv", index=0)

    rect_gen   = t2 - t1
    time_records.append({
        "run": int(runi+1),
        "model": "RectFlow",
        "gen_time": rect_gen,
    })
    print(f"[run {runi+1}] rectflow gen time: {rect_gen:.4f}s")

df_time = pd.DataFrame(time_records)
os.makedirs("./results", exist_ok=True)
df_time.to_csv("./results/timing_all_models.csv", index=False)


summary = df_time.groupby("model")[["gen_time"]].agg(
    ['mean', 'std']
)
print("\nSummary time:\n", summary)

  