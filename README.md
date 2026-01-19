# QRGMM: Quantile-Regression-Based Generative Metamodeling

This repository provides an **accessible, well-documented, and research-ready implementation of QRGMM (Quantile-Regression-Based Generative Metamodeling)**. It is designed both as a **reproducibility package** for the paper *“Learning to Simulate: Generative Metamodeling via Quantile Regression”*  , available at https://arxiv.org/abs/2311.17797 and as a **general-purpose codebase** that researchers and practitioners can easily adapt to new simulation-based learning problems.

The primary goal of this codebase is to make QRGMM easy to use, easy to adapt, and computationally efficient for simulation-based learning and real-time decision-making problems, we expose the full modeling pipeline, implementation details, and practical tricks required for fast and reliable conditional sample generation.

## 1. What is QRGMM?

QRGMM is a generative metamodeling framework for learning **conditional output distributions** of complex stochastic simulators using **quantile regression**, with the goal of building a fast **“simulator of a simulator.”** Given a dataset $(x_i, y_i)_{i=1}^{n}$, QRGMM approximates the conditional distribution $Y \mid X = x$ by:

1. **Fitting conditional quantile models** on a grid of quantile levels $(\tau_j = j/m: j=1,\ldots,m-1)$ via quantile regression;

2. **Constructing an approximate conditional quantile function** $\widehat{Q}(\tau \mid x)$ by **linearly interpolating** between the fitted quantile values across adjacent grid points;

3. **Generating samples via inverse transform sampling** by **plugging in uniform random variables** $u \sim \mathrm{Unif}(0,1)$ and outputting $\hat{y} = \widehat{Q}(u \mid x)$, which enables fast online conditional sample generation.

Key characteristics:

* Fully **nonparametric** conditional distribution learning, without imposing specific distribution type for the underlying conditional distribution

* Provides **more accurate and stable distributional learning** than GAN-, diffusion-, and rectified-flow-based methods in the **one-dimensional and low-dimensional output regimes** that are most common in stochastic simulation and operations research

* Extremely **fast online sample generation**, requiring less than $0.01$ seconds to generate $10^5$ conditional observations

* Naturally compatible with **simulation-based learning and real-time decision-making problems**

## 2. Implemented QRGMM Variants

This codebase includes multiple QRGMM variants discussed in the paper:

* **Standard QRGMM**&#x20;

  Classical linear quantile-regression-based generative metamodel.

* **Basis-funtion-based QRGMM**

  Linear quantile-regression-based generative metamodel with basis function.

* **QRGMM-R (Rearranged QRGMM)**
  Enforces monotonicity across quantiles to eliminate quantile crossing.

* **Neural-network-based QRGMM (QRNN-GMM)**
  Uses quantile regression neural networks for high-dimensional settings.

* **Multi-output QRGMM**
  Supports vector-valued simulator outputs via joint or marginal modeling.

## 3. Repository Structure

The repository is organized into two complementary layers:

### (a) Experiment & Replication Pipelines

End-to-end pipelines and the full source code used to reproduce all numerical experiments in the paper:

```
Experiments/
├── Artificial_Test_Problems/
├── Esophageal_Cancer_Simulator/
└── Bank_Simulator/
```

Each pipeline includes:

* data generation from simulators;

* quantile regression fitting;

* QRGMM training and sampling;

* comparisons with CWGAN, Diffusion, and Rectified Flow;

* Jupyter notebooks for tables and figures.

For a detailed, experiment-by-experiment description of the replication pipelines, execution order, environment setup and implementation details, please refer to **README.pdf** for more detailed instructions.

### (b) QRGMM Core Codebase

Reusable implementations of QRGMM and its variants:

```
QRGMM/
├── Vanilla_Python/
├── Basis_Function_Python/
├── Basis_Function_MATLAB/
├── QRGMM_R/ # Rearranged QRGMM (QRGMM-R) 
└── MutiOutput_NeuralNetwork/
```

The above structure clearly organizes the QRGMM-related implementations used in the numerical experiments of the paper in a unified and modular manner. Each directory corresponds to a specific QRGMM variant or implementation setting, and provides a **complete end-to-end pipeline**, including data preparation, QRGMM fitting, and online conditional sample generation. For details of the underlying datasets, experimental settings, and application-specific configurations, readers are referred to the paper and README.pdf for more information.

## 4. Environment and Dependencies

We recommend using **Anaconda** to ensure reproducibility.

```
conda env create -n QRGMM -f environment_QRGMM_history.yml
conda activate QRGMM
```

Main dependencies:

* Python ≥ 3.9

* numpy, scipy, pandas, matplotlib

* statsmodels (quantile regression)

* torch (for CWGAN, Diffusion, Rectified Flow)

* joblib (parallel quantile fitting)

R and MATLAB are used for specific components including quantile regression estimation and simulation procedures.

## 5. Typical Workflow

A minimal QRGMM workflow consists of:

1. **Generate training data** from a simulator

2. **Fit quantile regressions** on a predefined grid

3. **Store quantile coefficients**

4. **Generate conditional samples** using QRGMM

Conceptually:

```
Simulator → Quantile Regression → Quantile Grid → Fast Sampling
```

Both Python and MATLAB implementations follow this same logic, making the framework easy to understand and modify.

## 6. Practical Acceleration Techniques

A key focus of this codebase is **fast online sample generation**, which is critical when QRGMM is used for real-time decision-making problems. The implementations therefore emphasize **vectorized computation, parallelism, and avoidance of loop and branch flow**.

### 6.1 Fully Vectorized Online Sampling

The online QRGMM sampling algorithm is implemented using **pure matrix operations**, avoiding explicit `for` loops and `if–else` branching. Given a batch of covariates ($X \in \mathbb{R}^{k \times d}$), all conditional observations are generated simultaneously.

A key trick is to map uniform random variables directly to quantile indices via integer division. For a quantile grid with spacing $\Delta\tau$, the enclosing quantile index is computed as $j = \left\lfloor u / \Delta\tau \right\rfloor$, which corresponds to $j = \lfloor u m \rfloor$ in the QRGMM algorithm. Linear interpolation between adjacent quantiles is then carried out entirely via matrix operations.

An illustrative implementation is:

```python
def QRGMM(X):
    output_size = X.shape[0]
    u = np.random.rand(output_size)

    order = (u // le).astype(np.uint64)
    alpha = u - order * le

    b1 = fastmodels[order, 1:(d+2)]
    b2 = fastmodels[order+1, 1:(d+2)]

    w = alpha.reshape(-1, 1) / le
    b = b1 * (1 - w) + b2 * w

    sample = np.sum(b * X, axis=1)
    return sample
```

This design avoids explicit `for` loops and `if–else` branching and enables efficient batch sampling.

### 6.2 Tail Handling via Quantile Coefficient Expansion

&#x20;

To avoid special-case handling at the boundary quantiles, the quantile regression coefficient matrix is expanded at both ends by duplicating the first and last quantile coefficients. This allows interpolation to be handled uniformly for all samples without conditional checks, and enables fully vectorized computation throughout the online sampling stage.

```
fastmodels = np.zeros((nmodels.shape[0] + 2, nmodels.shape[1]))
fastmodels[0, :] = nmodels[0, :]
fastmodels[1:-1, :] = nmodels[:, :]
fastmodels[-1, :] = nmodels[-1, :]
```

### 6.3 Parallel Quantile Regression Training

Quantile regression models at different quantile levels are mutually independent. As a result, the offline training stage naturally supports parallel computation. In the Python implementation, this is achieved using `joblib`, while in the R implementation it is handled via `future.apply`. Parallelization substantially reduces training time when a fine quantile grid is used.

### 6.4 Large-Scale Sampling for Fixed Covariates

In applications where a large number of observations (e.g., $10^5$ or more) are required for the same covariate vector $x$, an additional acceleration strategy is applied. The entire conditional quantile curve is first computed via matrix multiplication between $x$ and the quantile regression coefficient matrix. All uniform random variables are then mapped simultaneously to target observations using vectorized interpolation for conditional quantile curve instead of  quantile regression coefficient.

This approach avoids processing each random draw of the uniform random variable individually and further improves computational efficiency.

```python
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
```

## 7. Reproducibility

All experimental results reported in the paper can be reproduced using this repository. Random seeds are fixed throughout, and all figures and tables are generated via documented Jupyter notebooks.

## 8. Reference

If you use this codebase, please cite:

> Hong L J, Hou Y, Zhang Q, et al. Learning to simulate: Generative metamodeling via quantile regression[J]. arXiv preprint arXiv:2311.17797, 2023.

## 9. Contact

For questions, suggestions, or bug reports, please contact:

**Qingkai Zhang**
Email: 22110690021@m.fudan.edu.cn
