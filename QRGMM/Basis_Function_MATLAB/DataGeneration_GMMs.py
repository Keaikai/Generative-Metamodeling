import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import random
import numpy as np
import pandas as pd
import torch
from utils import *
import wgan  # Load the wgan python file in the current directory (recommended, as it is more convenient,
             # facilitates reproducibility, and avoids potential issues that may arise during installation),
             # or install the package if needed.

# ---------------------------
# Basic configuration
# ---------------------------
run = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Dataset wrapper for Diffusion / Rectified Flow
# X columns: A,B,C,D ; Y column: F
# ---------------------------
class DiffRectDatasetFromDF:
    """
    Adapt a DataFrame with columns ["A","B","C","D","F"] to the format required by your train_all/generate interface:
      - x_train: standardized torch tensor of shape (n, 4)
      - y_train: standardized torch tensor of shape (n, 1)

    Also stores:
      - x_mean, x_std, y_mean, y_std
    so that generate() can normalize X consistently and denormalize Y.
    """
    def __init__(self, df: pd.DataFrame, device: str = "cpu"):
        x = df[["A", "B", "C", "D"]].to_numpy(dtype=np.float32)
        y = df[["F"]].to_numpy(dtype=np.float32)  # shape (n,1)

        # Standardize X
        self.x_mean = x.mean(axis=0)
        self.x_std  = x.std(axis=0)
        self.x_std[self.x_std == 0.0] = 1.0
        x_norm = (x - self.x_mean) / self.x_std

        # Standardize Y
        self.y_mean = float(y.mean())
        self.y_std  = float(y.std())
        if self.y_std == 0.0:
            self.y_std = 1.0
        y_norm = (y - self.y_mean) / self.y_std

        # Store torch tensors on the specified device
        self.x_train = torch.as_tensor(x_norm, dtype=torch.float32, device=device)
        self.y_train = torch.as_tensor(y_norm, dtype=torch.float32, device=device)
        self.device = device


def set_all_seeds(seed: int):
    """
    Set seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_output_dirs():
    """
    Create the folders needed for saving the required generated outputs.
    """
    # CWGAN outputs
    os.makedirs("./data/CWGANoutput1_1", exist_ok=True)
    os.makedirs("./data/CWGANoutput1_2", exist_ok=True)
    os.makedirs("./data/CWGANoutputtest1", exist_ok=True)
    os.makedirs("./data/CWGANoutputtest2", exist_ok=True)

    # Diffusion outputs
    os.makedirs("./data/Diffusionoutput1_1", exist_ok=True)
    os.makedirs("./data/Diffusionoutput1_2", exist_ok=True)
    os.makedirs("./data/Diffusionoutputtest1", exist_ok=True)
    os.makedirs("./data/Diffusionoutputtest2", exist_ok=True)

    # Rectified Flow outputs
    os.makedirs("./data/RectFlowoutput1_1", exist_ok=True)
    os.makedirs("./data/RectFlowoutput1_2", exist_ok=True)
    os.makedirs("./data/RectFlowoutputtest1", exist_ok=True)
    os.makedirs("./data/RectFlowoutputtest2", exist_ok=True)

    # Timing summary
    os.makedirs("./results", exist_ok=True)


def train_cwgan_and_generate(df_train, df_test_list_timed, df_test_list_untimed, max_epochs, device="cuda"):
    """
    Train CWGAN and generate data.

    Returns
    -------
    gen_out : dict[str, DataFrame]
        Generated DataFrames for all requested splits.
    gen_time : dict[str, float]
        Online generation time (seconds) for TIMED splits.
    """
    continuous_vars = ["F"]
    categorical_vars = []
    context_vars = ["A", "B", "C", "D"]

    data_wrapper = wgan.DataWrapper(df_train, continuous_vars, categorical_vars, context_vars)
    spec = wgan.Specifications(
        data_wrapper,
        batch_size=4096,
        max_epochs=max_epochs,
        critic_lr=1e-3,
        generator_lr=1e-3,
        print_every=100,
        device=device
    )
    generator = wgan.Generator(spec)
    critic = wgan.Critic(spec)

    # Train 
    y, context = data_wrapper.preprocess(df_train)
    wgan.train(generator, critic, y, context, spec)

    gen_out = {}
    gen_time = {}

    # Timed generation 
    for split_name, df_test in df_test_list_timed:
        t1 = time.perf_counter()
        out = data_wrapper.apply_generator(generator, df_test)
        t2 = time.perf_counter()
        gen_out[split_name] = out
        gen_time[split_name] = t2 - t1

    for split_name, df_test in df_test_list_untimed:
        out = data_wrapper.apply_generator(generator, df_test)
        gen_out[split_name] = out

    return gen_out, gen_time


def train_diffrect_and_generate(df_train, df_test_list_timed, df_test_list_untimed, model_type, runi, tag, device):
    """
    Train Diffusion/Rectified Flow and generate data.
    """
    data_dr = DiffRectDatasetFromDF(df_train, device=device)

    args_dr = get_args(data_type='ECsimulator')
    args_dr['data_size'] = len(df_train)
    args_dr['instance'] = f"ECsimulator_{tag}_run{runi+1}_n{len(df_train)}"
    ensure_instance_dirs(args_dr['instance'])

    # Train 
    train_all(data_dr, args_dr, model_type=model_type)

    gen_y = {}
    gen_time = {}

    def _gen_once(X_raw):
        if model_type == "diffusion":
            return generate(X_raw, data_dr, args_dr, model_name='diffusion')
        elif model_type == "rectified":
            return generate(X_raw, data_dr, args_dr, model_name='rectified')
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # Timed generation 
    for split_name, df_test in df_test_list_timed:
        X_raw = df_test[["A", "B", "C", "D"]].to_numpy(dtype=np.float32)
        t1 = time.perf_counter()
        y_hat = _gen_once(X_raw)
        t2 = time.perf_counter()
        gen_y[split_name] = y_hat
        gen_time[split_name] = t2 - t1

    for split_name, df_test in df_test_list_untimed:
        X_raw = df_test[["A", "B", "C", "D"]].to_numpy(dtype=np.float32)
        y_hat = _gen_once(X_raw)
        gen_y[split_name] = y_hat

    return gen_y, gen_time


# ---------------------------
# Main experiment loop
# ---------------------------
ensure_output_dirs()

# Only record timing for output1_1 and output1_2
time_records = []  # records: run, model, split, gen_time

for runi in np.arange(0, run):
    set_all_seeds(int(runi))

    # ---------------------------
    # Load data
    # ---------------------------
    df1 = pd.read_csv(f'./data/traindata1/traindata1_{runi+1}.csv', header=None)
    df1.columns = ["A", "B", "C", "D", "F"]

    df2 = pd.read_csv(f'./data/traindata2/traindata2_{runi+1}.csv', header=None)
    df2.columns = ["A", "B", "C", "D", "F"]

    testdf1_1 = pd.read_csv(f'./data/testdf1_1/testdf1_1_{runi+1}.csv', header=None)
    testdf1_1.columns = ["A", "B", "C", "D", "F"]

    testdf1_2 = pd.read_csv(f'./data/testdf1_2/testdf1_2_{runi+1}.csv', header=None)
    testdf1_2.columns = ["A", "B", "C", "D", "F"]

    testdf1 = pd.read_csv(f'./data/testdata1/testdata1_{runi+1}.csv', header=None)
    testdf1.columns = ["A", "B", "C", "D", "F"]

    testdf2 = pd.read_csv(f'./data/testdata2/testdata2_{runi+1}.csv', header=None)
    testdf2.columns = ["A", "B", "C", "D", "F"]

    # ==========================================================
    # CWGAN-GP
    # ==========================================================
    cwgan_out_1, cwgan_time_1 = train_cwgan_and_generate(
        df_train=df1,
        df_test_list_timed=[("1_1", testdf1_1)],
        df_test_list_untimed=[("test1", testdf1)],
        max_epochs=1000,
        device=DEVICE
    )
    cwgan_out_2, cwgan_time_2 = train_cwgan_and_generate(
        df_train=df2,
        df_test_list_timed=[("1_2", testdf1_2)],
        df_test_list_untimed=[("test2", testdf2)],
        max_epochs=1000,
        device=DEVICE
    )

    # Save CWGAN outputs (all required outputs are still saved)
    cwgan_out_1["1_1"].to_csv(f'./data/CWGANoutput1_1/CWGANoutput1_1_{runi+1}.csv', index=0)
    cwgan_out_2["1_2"].to_csv(f'./data/CWGANoutput1_2/CWGANoutput1_2_{runi+1}.csv', index=0)
    cwgan_out_1["test1"].to_csv(f'./data/CWGANoutputtest1/CWGANoutputtest1_{runi+1}.csv', index=0)
    cwgan_out_2["test2"].to_csv(f'./data/CWGANoutputtest2/CWGANoutputtest2_{runi+1}.csv', index=0)

    # Record CWGAN timing ONLY for 1_1 and 1_2
    time_records.append({"run": int(runi+1), "model": "CWGAN", "split": "Y1", "gen_time": cwgan_time_1["1_1"]})
    time_records.append({"run": int(runi+1), "model": "CWGAN", "split": "Y2", "gen_time": cwgan_time_2["1_2"]})

    print(
        f"[run {runi+1}] CWGAN timed gen: "
        f"Y1={cwgan_time_1['1_1']:.4f}s, Y2={cwgan_time_2['1_2']:.4f}s"
    )

    # ==========================================================
    # Diffusion
    # ==========================================================
    diff_y_1, diff_t_1 = train_diffrect_and_generate(
        df_train=df1,
        df_test_list_timed=[("1_1", testdf1_1)],
        df_test_list_untimed=[("test1", testdf1)],
        model_type="diffusion",
        runi=runi,
        tag="train1",
        device=DEVICE
    )
    diff_y_2, diff_t_2 = train_diffrect_and_generate(
        df_train=df2,
        df_test_list_timed=[("1_2", testdf1_2)],
        df_test_list_untimed=[("test2", testdf2)],
        model_type="diffusion",
        runi=runi,
        tag="train2",
        device=DEVICE
    )

    # Save Diffusion outputs
    out_1_1 = testdf1_1.copy()
    out_1_1["F"] = diff_y_1["1_1"]
    out_1_1.to_csv(f'./data/Diffusionoutput1_1/Diffusionoutput1_1_{runi+1}.csv', index=0)

    out_1_2 = testdf1_2.copy()
    out_1_2["F"] = diff_y_2["1_2"]
    out_1_2.to_csv(f'./data/Diffusionoutput1_2/Diffusionoutput1_2_{runi+1}.csv', index=0)

    out_test1 = testdf1.copy()
    out_test1["F"] = diff_y_1["test1"]
    out_test1.to_csv(f'./data/Diffusionoutputtest1/Diffusionoutputtest1_{runi+1}.csv', index=0)

    out_test2 = testdf2.copy()
    out_test2["F"] = diff_y_2["test2"]
    out_test2.to_csv(f'./data/Diffusionoutputtest2/Diffusionoutputtest2_{runi+1}.csv', index=0)

    # Record Diffusion timing 
    time_records.append({"run": int(runi+1), "model": "Diffusion", "split": "Y1", "gen_time": diff_t_1["1_1"]})
    time_records.append({"run": int(runi+1), "model": "Diffusion", "split": "Y2", "gen_time": diff_t_2["1_2"]})

    print(
        f"[run {runi+1}] Diffusion timed gen: "
        f"Y1={diff_t_1['1_1']:.4f}s, Y2={diff_t_2['1_2']:.4f}s"
    )

    # ==========================================================
    # Rectified Flow
    # ==========================================================
    rect_y_1, rect_t_1 = train_diffrect_and_generate(
        df_train=df1,
        df_test_list_timed=[("1_1", testdf1_1)],
        df_test_list_untimed=[("test1", testdf1)],
        model_type="rectified",
        runi=runi,
        tag="train1",
        device=DEVICE
    )
    rect_y_2, rect_t_2 = train_diffrect_and_generate(
        df_train=df2,
        df_test_list_timed=[("1_2", testdf1_2)],
        df_test_list_untimed=[("test2", testdf2)],
        model_type="rectified",
        runi=runi,
        tag="train2",
        device=DEVICE
    )

    # Save Rectified Flow outputs
    out_1_1 = testdf1_1.copy()
    out_1_1["F"] = rect_y_1["1_1"]
    out_1_1.to_csv(f'./data/RectFlowoutput1_1/RectFlowoutput1_1_{runi+1}.csv', index=0)

    out_1_2 = testdf1_2.copy()
    out_1_2["F"] = rect_y_2["1_2"]
    out_1_2.to_csv(f'./data/RectFlowoutput1_2/RectFlowoutput1_2_{runi+1}.csv', index=0)

    out_test1 = testdf1.copy()
    out_test1["F"] = rect_y_1["test1"]
    out_test1.to_csv(f'./data/RectFlowoutputtest1/RectFlowoutputtest1_{runi+1}.csv', index=0)

    out_test2 = testdf2.copy()
    out_test2["F"] = rect_y_2["test2"]
    out_test2.to_csv(f'./data/RectFlowoutputtest2/RectFlowoutputtest2_{runi+1}.csv', index=0)

    # Record Rectified Flow timing
    time_records.append({"run": int(runi+1), "model": "RectFlow", "split": "Y1", "gen_time": rect_t_1["1_1"]})
    time_records.append({"run": int(runi+1), "model": "RectFlow", "split": "Y2", "gen_time": rect_t_2["1_2"]})

    print(
        f"[run {runi+1}] RectFlow timed gen: "
        f"Y1={rect_t_1['1_1']:.4f}s, Y2={rect_t_2['1_2']:.4f}s"
    )


# ---------------------------
# Save timing results 
# ---------------------------
df_time = pd.DataFrame(time_records)
df_time.to_csv("./results/timing_online_generation_GMMs.csv", index=False)

summary = df_time.groupby(["model", "split"])[["gen_time"]].agg(["mean", "std"])
print("\nSummary time:\n", summary)
