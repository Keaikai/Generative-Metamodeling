import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random
import time
import ot
import sys

import wgan # Load the wgan function in the current directory
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#Prevent kernel suddenly interrupted caused by some unknown problem caused by CWGAN

def set_all_seeds(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Train CWGAN and Unconditional Test

run=100
# for runi in np.arange(0,run):

#     set_all_seeds(int(runi))

#     ############################### load data ###############################
#     df = pd.read_csv("./data/traindata/train_data_rep" + "{}.csv".format(runi+1),index_col=False,header=0)
#     testdf = pd.read_csv("./data/testdata/test_data_rep" + "{}.csv".format(runi+1),index_col=False,header=0)
#     ################################ CWGAN-GP ################################
    
#     # Y | X
#     continuous_vars = ["V11","V12","V13","V14","V15"]
#     categorical_vars = []
#     context_vars = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10"]
    
#     # Initialize objects
#     data_wrapper = wgan.DataWrapper(df, continuous_vars, categorical_vars, 
#                                   context_vars)
#     spec = wgan.Specifications(data_wrapper, batch_size=4096, max_epochs=1000, critic_lr=1e-3, generator_lr=1e-3,
#                              print_every=100, device = "cuda")
#     generator = wgan.Generator(spec)
#     critic = wgan.Critic(spec)
    
#     # train Y | X

#     y, context = data_wrapper.preprocess(df)
#     wgan.train(generator, critic, y, context, spec)

#     # simulate data with conditional WGANs
    
#     testdf_generated = data_wrapper.apply_generator(generator, testdf)
#     os.makedirs("./data/testdata_CWGAN", exist_ok=True)
#     testdf_generated.to_csv("./data/testdata_CWGAN/testdata_CWGAN_rep" + '{}.csv'.format(runi+1),index=0)
#     print(f"[rep {runi+1}] Saved CWGAN unconditional test outputs.")
# Online Generation Timing (Conditional Test)
seed=2024
set_all_seeds(seed)
    
############################### load data ###############################
df = pd.read_csv("./data/traindata/train_data_rep1.csv",index_col=False,header=0)
testdf = pd.read_csv("./data/onlinetest/test_data_x0.csv",index_col=False,header=0)
################################ CWGAN-GP ################################
    
# Y | X
continuous_vars = ["V11","V12","V13","V14","V15"]
categorical_vars = []
context_vars = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10"]
    
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


run=100
OnlineTime_CWGAN=np.zeros((run,1))
for runi in np.arange(0,run):
    set_all_seeds(int(runi))
    # simulate data with conditional WGANs
    time_start = time.time() 
    testdf_generated = data_wrapper.apply_generator(generator, testdf)
    time_end = time.time()
    OnlineTime_CWGAN[runi,0]= time_end - time_start
    os.makedirs("./data/onlinetest/testdata_CWGAN_online", exist_ok=True)
    testdf_generated.to_csv("./data/onlinetest/testdata_CWGAN_online/testdata_CWGAN_online_rep" + '{}.csv'.format(runi+1),index=0)
OnlineTime_CWGAN =pd.DataFrame(OnlineTime_CWGAN)
os.makedirs("./data/onlinetest", exist_ok=True)
OnlineTime_CWGAN.to_csv('./data/onlinetest/onlinetime_CWGAN.csv',index=0)  
    