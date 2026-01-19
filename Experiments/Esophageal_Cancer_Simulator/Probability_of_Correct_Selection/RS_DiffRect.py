import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

N_ROWS = 100          # number of design points in Xsample
MC_SAMPLES = 100000   # Monte Carlo samples per design point

# ============================================================
# Dataset wrapper for Diffusion / Rectified Flow
# X columns: A,B,C,D ; Y column: F
# ============================================================
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
        self.x_std = x.std(axis=0)
        self.x_std[self.x_std == 0.0] = 1.0
        x_norm = (x - self.x_mean) / self.x_std

        # Standardize Y
        self.y_mean = float(y.mean())
        self.y_std = float(y.std())
        if self.y_std == 0.0:
            self.y_std = 1.0
        y_norm = (y - self.y_mean) / self.y_std

        # Store torch tensors on the specified device
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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_output_dirs():
    """Create folders for saving RS choice results."""
    os.makedirs("./data/RS/RS_Diffusion", exist_ok=True)
    os.makedirs("./data/RS/RS_RectFlow", exist_ok=True)



# ============================================================
# Train Diffusion / RectFlow 
# ============================================================
def train_diffrect(df_train: pd.DataFrame, model_type: str, runi: int, tag: str, device: str):
    """
    model_type:
      - "diffusion"  -> train_all(..., model_type="diffusion"), generate(..., model_name="diffusion")
      - "rectified"  -> train_all(..., model_type="rectified"), generate(..., model_name="rectified")
    """
    data_dr = DiffRectDatasetFromDF(df_train, device=device)

    args_dr = get_args(data_type="ECsimulator")
    args_dr["data_size"] = len(df_train)
    args_dr["instance"] = f"RS_{tag}_run{runi+1}_n{len(df_train)}"
    ensure_instance_dirs(args_dr["instance"])

    train_all(data_dr, args_dr, model_type=model_type)
    return data_dr, args_dr


def generate_diffrect_y(x_raw: np.ndarray, data_dr, args_dr, model_type: str):
    """Generate y for given x_raw (N,4) using your generate() interface."""
    if model_type == "diffusion":
        return generate(x_raw, data_dr, args_dr, model_name="diffusion")
    elif model_type == "rectified":
        return generate(x_raw, data_dr, args_dr, model_name="rectified")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================
# Main loop: RS choice task
# ============================================================
ensure_output_dirs()

# Load X design points once (100 rows, columns A,B,C,D)
Xsample = pd.read_csv("./data/RS/Xsample/Xsample.csv", header=None)
Xsample.columns = ["A", "B", "C", "D"]
testdf = Xsample.copy()
testdf["F"] = 0.0

for runi in range(run):
    set_all_seeds(int(runi))

    # ---------------------------
    # Load train data (drop "I")
    # ---------------------------
    df1 = pd.read_csv(f"./data/RS/traindata1/traindata1_{runi+1}.csv", header=None)
    df1.columns = ["I", "A", "B", "C", "D", "F"]
    df1 = df1.drop(columns="I")

    df2 = pd.read_csv(f"./data/RS/traindata2/traindata2_{runi+1}.csv", header=None)
    df2.columns = ["I", "A", "B", "C", "D", "F"]
    df2 = df2.drop(columns="I")

    # ============================================================
    # Diffusion: train two models (df1 as Y1, df2 as Y2), then do the choice task
    # ============================================================
    data_diff1, args_diff1 = train_diffrect(df1, model_type="diffusion", runi=runi, tag="Diff_Y1", device=DEVICE)
    data_diff2, args_diff2 = train_diffrect(df2, model_type="diffusion", runi=runi, tag="Diff_Y2", device=DEVICE)

    choice_diff = np.zeros(N_ROWS)
    for row in range(N_ROWS):
        x = Xsample.loc[[row], ["A", "B", "C", "D"]].to_numpy(dtype=np.float32)
        X_raw = np.repeat(x, MC_SAMPLES, axis=0)  # (100000, 4)

        y1 = generate_diffrect_y(X_raw, data_diff1, args_diff1, model_type="diffusion")
        y2 = generate_diffrect_y(X_raw, data_diff2, args_diff2, model_type="diffusion")

        choice_diff[row] = (np.mean(y1) <= np.mean(y2)) + 1

    pd.DataFrame(choice_diff).to_csv(
        f"./data/RS/RS_Diffusion/choice_{runi+1}.csv",
        index=None, header=None
    )
    print(f"[run {runi+1}] Diffusion choices saved.")

    # ============================================================
    # Rectified Flow: train two models (df1 as Y1, df2 as Y2), then do the same choice task
    # ============================================================
    data_rect1, args_rect1 = train_diffrect(df1, model_type="rectified", runi=runi, tag="Rect_Y1", device=DEVICE)
    data_rect2, args_rect2 = train_diffrect(df2, model_type="rectified", runi=runi, tag="Rect_Y2", device=DEVICE)

    choice_rect = np.zeros(N_ROWS)
    for row in range(N_ROWS):
        x = Xsample.loc[[row], ["A", "B", "C", "D"]].to_numpy(dtype=np.float32)
        X_raw = np.repeat(x, MC_SAMPLES, axis=0)

        y1 = generate_diffrect_y(X_raw, data_rect1, args_rect1, model_type="rectified")
        y2 = generate_diffrect_y(X_raw, data_rect2, args_rect2, model_type="rectified")

        choice_rect[row] = (np.mean(y1) <= np.mean(y2)) + 1

    pd.DataFrame(choice_rect).to_csv(
        f"./data/RS/RS_RectFlow/choice_{runi+1}.csv",
        index=None, header=None
    )
    print(f"[run {runi+1}] RectFlow choices saved.")
