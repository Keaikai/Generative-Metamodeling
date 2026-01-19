import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import random
import numpy as np
import pandas as pd
import torch

from utils import *  


# ============================================================
# Basic configuration
# ============================================================
run = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# X columns: V1..V10 ; Y columns: V11..V15
X_COLS = [f"V{i}" for i in range(1, 11)]
Y_COLS = [f"V{i}" for i in range(11, 16)]


# ============================================================
# Dataset wrapper for Diffusion / Rectified Flow
# (Align with your train_all/generate interface)
# ============================================================
class DiffRectDatasetFromDF:
    """
    Adapt a DataFrame with columns X_COLS + Y_COLS to the format required by your train_all/generate interface:
      - x_train: standardized torch tensor of shape (n, 10)
      - y_train: standardized torch tensor of shape (n, 5)

    Also stores:
      - x_mean, x_std, y_mean, y_std
    so that generate() can normalize X consistently and denormalize Y.
    """
    def __init__(self, df: pd.DataFrame, device: str = "cpu"):
        x = df[X_COLS].to_numpy(dtype=np.float32)          # (n,10)
        y = df[Y_COLS].to_numpy(dtype=np.float32)          # (n,5)

        # Standardize X (per-feature)
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0)
        self.x_std[self.x_std == 0.0] = 1.0
        x_norm = (x - self.x_mean) / self.x_std

        # Standardize Y (per-feature)
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0)
        self.y_std[self.y_std == 0.0] = 1.0
        y_norm = (y - self.y_mean) / self.y_std

        self.x_train = torch.as_tensor(x_norm, dtype=torch.float32, device=device)
        self.y_train = torch.as_tensor(y_norm, dtype=torch.float32, device=device)
        self.device = device


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


def ensure_output_dirs_newdata():
    """Create folders for generated outputs and online timing logs."""
    os.makedirs("./data/testdata_Diffusion", exist_ok=True)
    os.makedirs("./data/testdata_RectFlow", exist_ok=True)

    os.makedirs("./data/onlinetest/testdata_Diffusion_online", exist_ok=True)
    os.makedirs("./data/onlinetest/testdata_RectFlow_online", exist_ok=True)
    os.makedirs("./data/onlinetest", exist_ok=True)


# ============================================================
# Train + generate helpers
# ============================================================
def train_diffrect_model(df_train: pd.DataFrame, model_type: str, instance: str, device: str):
    """
    Train one Diffusion / Rectified model on df_train.
    """
    data_dr = DiffRectDatasetFromDF(df_train, device=device)

    args_dr = get_args(data_type="BankSimulator")
    args_dr["data_size"] = len(df_train)
    args_dr["instance"] = instance
    args_dr["output_dim"] = len(Y_COLS)  # Y is 5-d now

    # Safety checks (prevents silent dimension mismatch)
    assert data_dr.y_train.shape[1] == args_dr["output_dim"], \
        f"y_train has {data_dr.y_train.shape[1]} dims but args['output_dim']={args_dr['output_dim']}"

    ensure_instance_dirs(args_dr["instance"])

    train_all(data_dr, args_dr, model_type=model_type)
    return data_dr, args_dr


def generate_diffrect(df_x: pd.DataFrame, data_dr, args_dr, model_name: str):
    """
    Generate Y for a given X DataFrame with columns X_COLS.
    Output: numpy array shape (n, output_dim) if you use the updated generate().
    """
    x_raw = df_x[X_COLS].to_numpy(dtype=np.float32)
    y_hat = generate(x_raw, data_dr, args_dr, model_name=model_name)
    return y_hat



# ============================================================
# Part 1: Train + unconditional test generation (rep-wise)
# ============================================================
ensure_output_dirs_newdata()

for runi in np.arange(0, run):
    set_all_seeds(int(runi))

    # ---------------------------
    # Load data
    # ---------------------------
    df = pd.read_csv(f"./data/traindata/train_data_rep{runi+1}.csv", header=0)
    testdf = pd.read_csv(f"./data/testdata/test_data_rep{runi+1}.csv", header=0)

    # ---------------------------
    # Diffusion: train + generate
    # ---------------------------
    instance_diff = f"BankSim_newdata_Diffusion_rep{runi+1}"
    data_diff, args_diff = train_diffrect_model(df, model_type="diffusion", instance=instance_diff, device=DEVICE)

    y_pred_diff = generate_diffrect(testdf, data_diff, args_diff, model_name="diffusion")
    out_diff = testdf.copy()
    out_diff.loc[:, Y_COLS] = y_pred_diff

    out_diff.to_csv(f"./data/testdata_Diffusion/testdata_Diffusion_rep{runi+1}.csv", index=0)

    # ---------------------------
    # Rectified Flow: train + generate
    # ---------------------------
    instance_rect = f"BankSim_newdata_RectFlow_rep{runi+1}"
    data_rect, args_rect = train_diffrect_model(df, model_type="rectified", instance=instance_rect, device=DEVICE)

    y_pred_rect = generate_diffrect(testdf, data_rect, args_rect, model_name="rectified")
    out_rect = testdf.copy()
    out_rect.loc[:, Y_COLS] = y_pred_rect

    out_rect.to_csv(f"./data/testdata_RectFlow/testdata_RectFlow_rep{runi+1}.csv", index=0)

    print(f"[rep {runi+1}] Saved Diffusion + RectFlow unconditional test outputs.")


# ============================================================
# Part 2: Online generation timing (conditional test)
# (Train ONCE on train_data_rep1, then time repeated generations on test_data_x0)
# ============================================================
seed = 2024
set_all_seeds(seed)

df = pd.read_csv("./data/traindata/train_data_rep1.csv", header=0)
testdf_online = pd.read_csv("./data/onlinetest/test_data_x0.csv", header=0)

# ---------------------------
# Train Diffusion once 
# ---------------------------
instance_diff_online = "BankSim_newdata_Diffusion_online_trainrep1"
data_diff_online, args_diff_online = train_diffrect_model(
    df, model_type="diffusion", instance=instance_diff_online, device=DEVICE
)

OnlineTime_Diffusion = np.zeros((run, 1))

for runi in np.arange(0, run):
    set_all_seeds(int(runi))

    t1 = time.perf_counter()
    y_hat = generate_diffrect(testdf_online, data_diff_online, args_diff_online, model_name="diffusion")
    t2 = time.perf_counter()

    OnlineTime_Diffusion[runi, 0] = t2 - t1

    out = testdf_online.copy()
    out.loc[:, Y_COLS] = y_hat
    out.to_csv(f"./data/onlinetest/testdata_Diffusion_online/testdata_Diffusion_online_rep{runi+1}.csv", index=0)

OnlineTime_Diffusion = pd.DataFrame(OnlineTime_Diffusion)
OnlineTime_Diffusion.to_csv("./data/onlinetest/onlinetime_Diffusion.csv", index=0)

print("Saved ./data/onlinetest/onlinetime_Diffusion.csv")


# ---------------------------
# Train RectFlow once 
# ---------------------------
set_all_seeds(seed)

instance_rect_online = "BankSim_newdata_RectFlow_online_trainrep1"
data_rect_online, args_rect_online = train_diffrect_model(
    df, model_type="rectified", instance=instance_rect_online, device=DEVICE
)

OnlineTime_RectFlow = np.zeros((run, 1))

for runi in np.arange(0, run):
    set_all_seeds(int(runi))

    t1 = time.perf_counter()
    y_hat = generate_diffrect(testdf_online, data_rect_online, args_rect_online, model_name="rectified")
    t2 = time.perf_counter()

    OnlineTime_RectFlow[runi, 0] = t2 - t1

    out = testdf_online.copy()
    out.loc[:, Y_COLS] = y_hat
    out.to_csv(f"./data/onlinetest/testdata_RectFlow_online/testdata_RectFlow_online_rep{runi+1}.csv", index=0)

OnlineTime_RectFlow = pd.DataFrame(OnlineTime_RectFlow)
OnlineTime_RectFlow.to_csv("./data/onlinetest/onlinetime_RectFlow.csv", index=0)

print("Saved ./data/onlinetest/onlinetime_RectFlow.csv")
