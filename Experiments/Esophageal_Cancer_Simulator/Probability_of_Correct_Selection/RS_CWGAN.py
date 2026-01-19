import pandas as pd 
from scipy.stats import wasserstein_distance
import numpy as np
import random
import torch
import os

import wgan

run=100 # number of experiment repetitions

testdf = pd.read_csv('./data/RS/Xsample/Xsample.csv',header=None)
testdf.columns=["A","B","C","D"]# test data1   
testdf["F"]=0

os.makedirs("./data/RS/RS_CWGAN", exist_ok=True)

def set_all_seeds(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


for runi in np.arange(0,run):
    
    choice=np.zeros(100)
    samplesize=np.zeros(100)
    
    set_all_seeds(int(runi))
    
    df1 = pd.read_csv('./data/RS/traindata1/traindata1' + '_{}.csv'.format(runi+1),header=None)
    df1.columns=["I","A","B","C","D","F"] # training data1
    df1=df1.drop(columns="I")  

    df2 = pd.read_csv('./data/RS/traindata2/traindata2' + '_{}.csv'.format(runi+1),header=None)
    df2.columns=["I","A","B","C","D","F"] # training data2
    df2=df2.drop(columns="I")  
                       
    
    ################################ CWGAN-GP ############################################# 
    
    # Y1| X
    continuous_vars = ["F"]
    categorical_vars = []
    context_vars = ["A","B","C","D"]
    
    # Initialize objects for Y1
    data_wrapper = wgan.DataWrapper(df1, continuous_vars, categorical_vars, 
                                  context_vars)
    spec = wgan.Specifications(data_wrapper, batch_size=32768, max_epochs=500, critic_lr=1e-3, generator_lr=1e-3,
                             print_every=100, device = "cuda")
    generator = wgan.Generator(spec)
    critic = wgan.Critic(spec)
    
    # train Y1 | X

    y, context = data_wrapper.preprocess(df1)
    wgan.train(generator, critic, y, context, spec)
    
    # Initialize objects for Y2
    data_wrapper2 = wgan.DataWrapper(df2, continuous_vars, categorical_vars, 
                                  context_vars)
    spec2 = wgan.Specifications(data_wrapper2, batch_size=32768, max_epochs=500, critic_lr=1e-3, generator_lr=1e-3,
                             print_every=100, device = "cuda")
    generator2 = wgan.Generator(spec2)
    critic2 = wgan.Critic(spec2)
    
    # train Y | X
    y2, context2 = data_wrapper2.preprocess(df2)
    wgan.train(generator2, critic2, y2, context2, spec2)


    for row in range(100):
        testdf_temp=testdf.loc[[row],:]
        testdfrow = pd.DataFrame(np.repeat(testdf_temp.values,100000,axis=0))
        testdfrow.columns = testdf_temp.columns
        cwgan_generated_row1 = data_wrapper.apply_generator(generator, testdfrow)
        cwgan_generated_row2 = data_wrapper2.apply_generator(generator2, testdfrow)
  
        choice[row] = (cwgan_generated_row1["F"].mean()<=cwgan_generated_row2["F"].mean())+1
    choice = pd.DataFrame(choice)
    choice.to_csv('./data/RS/RS_CWGAN/choice' + '_{}.csv'.format(runi+1),index=None,header=None) 
    print(f"[run {runi+1}] CWGAN choices saved.")                                                   