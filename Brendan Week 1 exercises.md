# HCT Microbiome Analysis — Exercises

---

## Dataset Overview

**Liao et al. (2021)**, *Scientific Data* 8:71. DOI: 10.1038/s41597-021-00860-8

Over 10,000 longitudinal fecal 16S rRNA samples from >1,000 allogeneic hematopoietic cell
transplantation (allo-HCT) patients at MSKCC, paired with clinical metadata
("hospitalome"): drug administrations, blood cultures, temperatures.

**Key tables:**

| Table | Contents |
|---|---|
| `samples/tblASVsamples.csv` | Per-sample metadata: `SampleID`, `PatientID`, `DayRelativeToNearestHCT` |
| `counts/tblcounts_asv_melt.csv` | Long-format ASV counts: `SampleID`, `ASV`, `Count` |
| `counts/tblcounts_class_wide.csv` | Class-level abundance table |
| `counts/tblqpcr.csv` | Total 16S gene copies per sample (see Exercise 4) |
| `meta_data/tbldrug.csv` | Drug administrations: `PatientID`, drug name/category, start/stop day relative to HCT |
| `taxonomy/tblASVtaxonomy_silva132_v4v5_filter.csv` | ASV → Kingdom … Genus mapping |

**Linking keys:** `SampleID` connects counts ↔ sample metadata; `PatientID` connects
samples ↔ drug records across time.

**Temporal anchor:** All time is indexed by `DayRelativeToNearestHCT`, with day 0 = transplant date.

---

## Exercise 1 — Alpha Diversity vs. Day Relative to HCT

Alpha diversity quantifies within-sample taxonomic richness. We use **inverse Simpson**:

$$D = \frac{1}{\sum_i p_i^2}$$

where $p_i$ is the relative abundance of taxon $i$ ($\sum_i p_i = 1$). $D$ ranges from 1
(complete domination by a single taxon) to $S$ (perfectly even distribution across $S$ taxa).

Compute $D$ per sample from `tblcounts_asv_melt.csv`. Merge with `tblASVsamples.csv` on
`SampleID` to attach `DayRelativeToNearestHCT`. Restrict to the window $[-15, +35]$ days,
bin by integer day, and plot mean $\pm$ SEM vs. day with a vertical line at day 0.

---

## Exercise 2 — In Silico Microbial Community Simulation

### Model definition

The generalized Lotka–Volterra (gLV) model with external perturbations:

$$\dot{x}_i = x_i \left( r_i + \sum_{j=1}^n A_{ij}\, x_j + \sum_{k=1}^p B_{ik}\, u_k(t) \right), \quad i = 1, \ldots, n$$

**Matrix-vector form.** Let $X = \operatorname{diag}(\mathbf{x}) \in \mathbb{R}^{n \times n}$:

$$\dot{\mathbf{x}} = X\bigl(\mathbf{r} + A\mathbf{x} + B\mathbf{u}(t)\bigr)$$

**Parameters:**
- $\mathbf{r} \in \mathbb{R}^n$ — intrinsic growth rates
- $A \in \mathbb{R}^{n \times n}$ — interaction matrix ($A_{ii}$ self-limitation; $A_{ij}$ inter-taxon effect of $j$ on $i$)
- $B \in \mathbb{R}^{n \times p}$ — perturbation susceptibility
- $\mathbf{u}(t) \in \{0,1\}^p$ — binary perturbation signal (is antibiotic class $k$ active at time $t$?)

### Synthetic parameters

Use $n = 4$ taxa and $p = 1$ antibiotic perturbation:

$$\mathbf{r} = \begin{pmatrix} 0.8 \\ 0.6 \\ 0.4 \\ 1.2 \end{pmatrix}, \qquad
A = \begin{pmatrix}
-1.0 & -0.2 & 0    & 0    \\
-0.1 & -0.8 & -0.3 & 0    \\
 0   & -0.1 & -0.9 & -0.2 \\
 0   &  0   & -0.1 & -1.5
\end{pmatrix}, \qquad
B = \begin{pmatrix} -2.0 \\ -1.5 \\ -1.8 \\ 0.2 \end{pmatrix}$$

**Initial condition:** $\mathbf{x}(0) = (0.3,\ 0.25,\ 0.25,\ 0.2)^\top$

**Perturbation schedule:** $u(t) = 1$ for $t \in [5, 12]$, else $0$.

Numerically integrate the ODE over $t \in [0, 30]$ using `scipy.integrate.solve_ivp`. Plot
all four taxon trajectories with the antibiotic window shaded. Does the community return to
its pre-perturbation equilibrium after $u$ is removed? Repeat with $B_{4,1} = -2.0$ and
compare the post-perturbation recovery.

---

## Exercise 3 — Parameter Recovery from Synthetic Data

Given the trajectory simulated in Exercise 2, recover $\mathbf{r}$, $A$, $B$ from the
time-series alone — without access to the true parameters — as a validation of the
inference pipeline prior to application on clinical data.

### Log-linearization

Dividing the gLV equation by $x_i$:

$$\frac{d \ln x_i}{dt} = r_i + \sum_j A_{ij}\, x_j(t) + \sum_k B_{ik}\, u_k(t)$$

The right-hand side is linear in $(r_i,\, A_{i\cdot},\, B_{i\cdot})$. Approximating the
derivative between consecutive timepoints $t_m$, $t_{m+1}$:

$$\frac{\ln x_i(t_{m+1}) - \ln x_i(t_m)}{t_{m+1} - t_m} \approx r_i + \sum_j A_{ij}\, x_j(t_m) + \sum_k B_{ik}\, u_k(t_m)$$

Stacking over all timestep pairs gives, for each taxon $i$, the linear system:

$$\mathbf{z}_i = \Phi\,\boldsymbol{\theta}_i + \boldsymbol{\varepsilon}$$

where $\mathbf{z}_i \in \mathbb{R}^M$ is the vector of log-derivatives,
$\Phi = [\mathbf{1},\ X_0,\ U_0] \in \mathbb{R}^{M \times (1 + n + p)}$ is the design
matrix evaluated at $t_m$, and $\boldsymbol{\theta}_i = (r_i,\, A_{i\cdot},\, B_{i\cdot})^\top$.

Each row of $\Phi$ corresponds to one consecutive sample pair $(t_m, t_{m+1})$: the first
entry is 1 (intercept, absorbing $r_i$), the next $n$ entries are the taxon abundances
$\mathbf{x}(t_m)$, and the final $p$ entries are the perturbation indicators $\mathbf{u}(t_m)$.
The corresponding entry of $\mathbf{z}_i$ is the finite-difference log-derivative for taxon
$i$ across that interval. The system is overdetermined — many more sample pairs $M$ than
parameters $1 + n + p$ — so we cannot solve it exactly, and instead minimize the residual
$\|\mathbf{z}_i - \Phi\,\boldsymbol{\theta}_i\|_2^2$ over $\boldsymbol{\theta}_i$.

The ordinary least-squares solution $\hat{\boldsymbol{\theta}}_i = (\Phi^\top \Phi)^{-1} \Phi^\top \mathbf{z}_i$
is unstable when columns of $\Phi$ are collinear (which is common when taxa co-vary) and
tends to overfit when $M$ is not large relative to the number of parameters. Ridge
regression addresses this by adding an $\ell_2$ penalty on parameter magnitude:

$$\hat{\boldsymbol{\theta}}_i = \underset{\boldsymbol{\theta}}{\arg\min}\ \|\mathbf{z}_i - \Phi\,\boldsymbol{\theta}\|_2^2 + \lambda\|\boldsymbol{\theta}\|_2^2$$

The closed-form solution is $\hat{\boldsymbol{\theta}}_i = (\Phi^\top \Phi + \lambda I)^{-1} \Phi^\top \mathbf{z}_i$.
The regularization term $\lambda \|\boldsymbol{\theta}\|_2^2$ penalizes large parameter
values, shrinking the solution toward zero and stabilizing the inversion of $\Phi^\top\Phi$.
The hyperparameter $\lambda > 0$ controls the bias-variance tradeoff: larger $\lambda$
produces more regularized (smaller, more stable) estimates at the cost of bias away from the
true parameters. In `sklearn`, this is `Ridge(alpha=lambda, fit_intercept=True)`, where
`fit_intercept=True` means $r_i$ is handled separately from the columns of $\Phi$ (i.e.
drop the leading 1 column and let sklearn add the intercept).

Construct $\Phi$ and $\mathbf{z}_i$ from the Exercise 2 trajectory, fit with $\lambda = 1$
as a starting point, and report the elementwise error $\hat{\boldsymbol{\theta}}_i - \boldsymbol{\theta}_i^*$
against the known ground-truth parameters.

---

## Exercise 4 — Fitting the gLV Model to Clinical Data

### Absolute abundance via qPCR

16S sequencing yields only **relative** abundances $p_i(t)$ ($\sum_i p_i = 1$). Fitting
gLV directly on relative abundances introduces spurious negative correlations due to
compositionality — a unit increase in one taxon forces all others to decrease proportionally
regardless of their true dynamics.

`tblqpcr.csv` contains total 16S gene copy number per gram of stool $Q(t)$ per sample,
measured by quantitative PCR. Absolute abundance is then:

$$a_i(t) = p_i(t) \times Q(t)$$

Merge on `SampleID` and substitute $a_i$ for $x_i$ throughout; the log-derivative
approximation and regression formulation from Exercise 3 carry over without modification.

### Fitting procedure

Load `tblcounts_class_wide.csv`, normalize to relative abundance, scale by matched qPCR
values from `tblqpcr.csv`, and select the **top 10 classes** by mean absolute abundance
across all samples.

Construct the regression dataset by iterating over patients. For each patient, sort their
samples by `DayRelativeToNearestHCT` and form all consecutive pairs $(t_m, t_{m+1})$ with
gap $\leq 7$ days. For each such pair, record:
- $\mathbf{x}(t_m)$ — the $n$-vector of class absolute abundances at $t_m$ (add pseudocount
  $\epsilon = 10^{-4}$ before taking logs)
- $\mathbf{u}(t_m)$ — the $p$-vector of antibiotic indicators active at $t_m$, derived from
  `tbldrug.csv` by checking whether the sample day falls within any drug's start/stop window
  for that patient
- $\mathbf{z}(t_m)$ — the $n$-vector of finite-difference log-derivatives
  $[\ln x_i(t_{m+1}) - \ln x_i(t_m)] / (t_{m+1} - t_m)$, one entry per class

Stack these across all patients and all valid pairs to form the global design matrix
$\Phi \in \mathbb{R}^{M \times (n + p)}$ and response matrix
$\mathbf{Z} \in \mathbb{R}^{M \times n}$, where $M$ is the total number of sample pairs.
Note that $\Phi$ and $\mathbf{Z}$ are shared across taxa — the regression for each class $i$
uses the same $\Phi$ but a different response column $\mathbf{z}_i$. Fit one `Ridge` model
per class and tune $\lambda$ by cross-validation (e.g. `RidgeCV` in sklearn).

Using $\hat{\mathbf{r}},\, \hat{A},\, \hat{B}$, simulate the gLV ODE forward for one
held-out patient from their observed initial condition and antibiotic schedule as
$\mathbf{u}(t)$, and overlay the simulated trajectory against their observed class-level
time series.
