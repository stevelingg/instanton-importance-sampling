# Instanton-Guided Importance Sampling for Rare Financial Events

A Python implementation of **physics-inspired rare-event simulation** for path-dependent financial drawdowns. The project combines Freidlin–Wentzell large-deviation theory, minimum-action optimisation and importance sampling under a regularised Heston-type stochastic-volatility model.

Developed as the computational component of a third-year group research project in theoretical physics at Imperial College London (2025–2026).

> [!NOTE]
> This is a numerical study of rare-event sampling, not a validated market crash forecasting system. The historical SPY results illustrate the algorithm; the predictive validity of the fitted market model has not been established.

## Sampling mechanism

![Ordinary Monte Carlo versus instanton-guided importance sampling](docs/figures/instanton_sampling.png)

Both panels use the same synthetic stochastic-volatility model and initial state. The left samples under the original measure $P$; the right uses a controlled proposal measure $Q$. The dashed black line is the numerically optimised instanton; the proposal uses a tempered, capped version of its control.

| Synthetic experiment | Ordinary Monte Carlo | Instanton-guided IS |
|---|---:|---:|
| Simulated paths | 50,000 | 50,000 |
| Raw event hit rate | 1.988% | 19.456% |
| Estimated target probability | 1.988% | 1.902% |
| Monte Carlo standard error (percentage points) | 0.0624 | 0.0319 |
| Weight ESS | 50,000 | 7,061 |

For this 40-trading-day, 20% drawdown example, the proposal increases the hit rate approximately 9.8-fold while the likelihood-ratio correction estimates the same target probability. The estimated variance is approximately 3.8 times smaller at equal path count. This is one synthetic experiment, not a general efficiency claim or a comparison at equal runtime.

Reproduce the figure without market data:

```bash
python -m pip install numpy pandas scipy matplotlib yfinance
python scripts/05_plot_sampling_demo.py
```

## Mathematical framework

### 1. Stochastic-volatility dynamics

Let $Z_k=(X_k,V_k)$, where $X_k$ is the logarithm of the adjusted close and $V_k$ is the variance state. The implementation uses a daily Euler discretisation, with $\Delta t=1/252$ years:

```math
Z_{k+1}=Z_k+b(Z_k)\Delta t+\sigma(Z_k)\Delta B_k,\qquad \Delta B_k\sim\mathcal{N}(0,\Delta t I_2).
```

The regularised Heston/Cox–Ingersoll–Ross (CIR)-type coefficients are

```math
b(z)=\begin{pmatrix}\mu\\\kappa(\bar v-g(v))\end{pmatrix},\qquad
\sigma(z)=\sqrt{g(v)}\begin{pmatrix}1&0\\\xi\rho&\xi\sqrt{1-\rho^2}\end{pmatrix}.
```

The effective variance is smoothly bounded below using

```math
g(v)=v_{\min}+\frac{1}{\alpha_v}\log(1+\exp(\alpha_v(v-v_{\min}))).
```

Here $\mu$ is log-price drift, $\kappa$ and $\bar v$ are variance mean-reversion parameters, $\xi$ is volatility of variance and $\rho$ is shock correlation. Unlike standard Heston dynamics, the model uses a constant log-price drift and regularised variance coefficients.

The five parameters are fitted via Gaussian quasi-maximum likelihood (QMLE) on daily log returns and changes in a 20-trading-day annualised realised-variance proxy. This is a proxy-based approximation, not a full latent-state likelihood fit.

### 2. Path-dependent drawdown event

Define the running log-price peak, log-drawdown and fractional drawdown:

```math
M_k=\max_{j\le k}X_j,\qquad D_k=M_k-X_k,\qquad \delta_k=1-e^{-D_k}.
```

For a fractional threshold $\delta$ and horizon $n$ trading days, let $d=-\log(1-\delta)$. The target is the conditional model probability

```math
p_k(n,d)=P\left(\max_{0\le i\le n}D_{k+i}\ge d\mid X_k,V_k,D_k\right).
```

The drawdown is updated deterministically from each simulated log-price path, including its initial historical peak. It is not a third SDE. If $D_k\ge d$, the event is already satisfied and $p_k=1$ by construction.

### 3. Discrete minimum-action path (instanton)

For a diffusion with noise scaled by $\sqrt{\varepsilon}$, Freidlin–Wentzell theory associates rare paths with the small-noise rate functional

```math
S[z]=\frac{1}{2}\int_0^T\|\sigma(z_t)^{-1}(\dot z_t-b(z_t))\|^2\,dt.
```

The exponential large-deviation scaling is governed by $S[z]/\varepsilon$. This motivates the proposal construction; it is not used as a direct finite-noise crash probability formula.

Matching the action to the daily Euler scheme gives the step control and discrete action

```math
u_i(z)=\sigma(z_i)^{-1}\left(\frac{z_{i+1}-z_i}{\Delta t}-b(z_i)\right),
```

```math
S^{(\tau)}[z]=\frac{\Delta t}{2}\sum_{i=0}^{\tau-1}\|u_i(z)\|^2.
```

For each candidate hitting day $\tau=1,\ldots,n$, a fixed-time minimum-action problem is solved numerically subject to the initial state and first-hitting constraints. If $\mathcal{A}_\tau$ denotes the relevant constrained path space, the objective is

```math
z^{\star,\tau}\approx\underset{z\in\mathcal{A}_\tau}{\arg\min}\;S^{(\tau)}[z].
```

The implementation uses L-BFGS with analytic gradients, a smooth running-maximum approximation and a pre-hit penalty. It searches every candidate hitting day and selects the lowest-action admissible solution after checking the exact drawdown and constraint tolerances. Global optimality is not guaranteed.

### 4. Controlled proposal and exact change of measure

Extract $u_i^*=u_i(z^*)$ from the selected instanton, temper it by $\lambda$ and cap its Euclidean norm at $u_{\max}$. For steps before the selected hitting time, use

```math
u_i=\lambda u_i^*\min\left(1,\frac{u_{\max}}{\|\lambda u_i^*\|}\right),
```

with zero control when the raw control is zero. The control is set to zero from the selected hitting time $\tau^*$ onwards, rather than after each simulated path's own hitting time.

Under $Q$, the Brownian increments are shifted:

```math
\Delta B_i\sim\mathcal{N}(u_i\Delta t,\Delta t I_2).
```

For these shifted increments, the exact Gaussian-shift path likelihood ratio is

```math
L=\frac{dP}{dQ}=\exp\left(\sum_{i=0}^{n-1}\left[-u_i^\top\Delta B_i+\frac{1}{2}\|u_i\|^2\Delta t\right]\right).
```

For the event $A$ defined in Section 2, the reweighted estimator is

```math
P(A)=\mathbb{E}_Q[\mathbf{1}_A L],\qquad \hat p_{\mathrm{IS}}=\frac{1}{N}\sum_{j=1}^{N}\mathbf{1}_A^{(j)}L^{(j)}.
```

This changes the sampling distribution, not the model probability being estimated. The likelihood-ratio identity is exact for the specified discretisation; the finite-sample estimate remains noisy.

### 5. Numerical diagnostics

The implementation records Monte Carlo standard errors (computed from weighted event indicators), proposal hit rates, likelihood-weight statistics and weight effective sample size:

```math
\mathrm{ESS}=\frac{(\sum_{j=1}^{N}L^{(j)})^2}{\sum_{j=1}^{N}(L^{(j)})^2}.
```

The mean weight is compared against $\mathbb{E}_Q[L]=1$. Neither this check nor ESS alone establishes rare-event estimator accuracy.

## Historical SPY demonstration

The original investigation fits the model using pre-2007 observations and evaluates it from 1 January 2007 to 31 October 2008. The four configurations combine 20- and 60-trading-day horizons with 20% and 30% drawdown thresholds. Non-trivial queries use 5,000 paths, $\lambda=0.5$ and $u_{\max}=10$.

| Horizon | Drawdown | Median weight ESS | Mean weight |
|---:|---:|---:|---:|
| 20 days | 20% | 710 | 0.999 |
| 20 days | 30% | 81 | 0.997 |
| 60 days | 20% | 2,247 | 1.002 |
| 60 days | 30% | 601 | 1.007 |

The original outputs include probability time series, calibration and scoring diagnostics, and full-sample and pre-hit evaluations. The low ESS in the 20-day/30% configuration indicates substantial weight concentration.

> [!WARNING]
> The historical figures are not representative evidence of prospective crash prediction. The event includes the simulation origin and uses the historical running peak: an already-breached threshold produces probability one, and proximity to the threshold can inflate apparent forecasting performance. Pre-hit evaluation removes already-breached cases but does not resolve the remaining target-definition or model-validity issues.

## Reproducing the historical experiment

From the repository root (install dependencies as above):

```bash
python scripts/00_fetch_data.py
python scripts/01_build_state.py --ticker SPY --price_col "Adj Close" --rv_window_days 20
python scripts/02_fit_model.py --ticker SPY --train_end 2006-12-31
python scripts/03_daily_forecast.py
python scripts/04_run_evaluation.py --run_dir outputs/runs/YOUR_RUN_FOLDER
```

Replace `YOUR_RUN_FOLDER` with the generated run directory. For a moved or archived run, pass the matching input paths explicitly with `--state_csv`, `--params_json` and, where applicable, `--instantons_dir`; recorded fingerprints must match. The historical script names retain `forecast` for compatibility.

Optional comparison against ordinary Monte Carlo:

```bash
python scripts/04_run_evaluation.py --run_dir outputs/runs/YOUR_RUN_FOLDER --run_naive_checks --naive_n 20000 --is_n 5000
```

Use `--no-make_plots` to disable evaluation plots. Parallel execution of the daily run requires explicit `--allow_parallel_nondeterminism` because block-local warm starts can change the selected instanton. Each pipeline stage records configuration and provenance in metadata.

Run implementation tests:

```bash
python -m pip install pytest
pytest -q
```

## Repository structure

```text
src/
  data_state.py          # Price, drawdown and variance-proxy construction
  fit.py                 # Gaussian QMLE
  model_sv.py            # Effective variance regularisation
  instanton.py           # Discrete action, gradients and hitting-time search
  importance_sampling.py # Controlled simulation, weights and estimates
  evaluation.py          # Numerical and exploratory historical diagnostics
  plotting.py            # Visualisation
  constants.py, io.py    # Configuration and I/O
scripts/                 # Pipeline entry points and synthetic demonstration
  05_plot_sampling_demo.py
docs/figures/            # Sampling comparison figure
data/                    # Generated raw and processed data
outputs/                 # Parameter snapshots and simulation results
tests/                   # Implementation tests
```

## Limitations

- **Model risk:** The diffusion has no explicit jumps or heavy-tailed innovations, and the proxy-based QMLE does not fully account for latent-variance uncertainty or market regime changes.
- **Historical validity:** The selected crisis period and original event definition do not establish generalisable predictive performance.
- **Sampling efficiency:** Increasing the raw hit rate does not guarantee variance reduction; repeated experiments at matched runtime, including optimiser cost, are needed to establish efficiency.
- **Numerical reliability:** Optimisation is local; weights can degenerate and zero-hit runs can report zero standard error without establishing that the event is impossible. The implementation also rejects materially out-of-range finite-sample probability estimates, which can occur in ordinary importance sampling.

Let $p_\theta$ be the event probability under the fitted model and $p_{\mathrm{true}}$ the unknown real-world conditional probability of the same event. The central distinction is between Monte Carlo error and model error:

```math
\hat p_{\mathrm{IS}}-p_{\mathrm{true}}=(\hat p_{\mathrm{IS}}-p_\theta)+(p_\theta-p_{\mathrm{true}}).
```

The instanton proposal addresses the first term through a change of measure. The second depends on the stochastic model and its estimated parameters. A more precise estimate of a misspecified model is not necessarily a better financial prediction.

## Selected references

- Freidlin, M. I. and Wentzell, A. D. *Random Perturbations of Dynamical Systems*. 3rd ed., Springer, 2012.
- Grafke, T. and Vanden-Eijnden, E. “Numerical computation of rare events via large deviation theory.” *Chaos* 29, 063118 (2019). [DOI](https://doi.org/10.1063/1.5084025).
- E, W., Ren, W. and Vanden-Eijnden, E. “Minimum action method for the study of rare events.” *Communications on Pure and Applied Mathematics* 57, 637–656 (2004).
- Heston, S. L. “A closed-form solution for options with stochastic volatility with applications to bond and currency options.” *Review of Financial Studies* 6, 327–343 (1993).
- Cox, J. C., Ingersoll, J. E. and Ross, S. A. “A theory of the term structure of interest rates.” *Econometrica* 53, 385–407 (1985).
- Glasserman, P. *Monte Carlo Methods in Financial Engineering*. Springer, 2004.
- Bucklew, J. A. *Introduction to Rare Event Simulation*. Springer, 2004.
