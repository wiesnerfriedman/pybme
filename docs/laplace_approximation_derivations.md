# Laplace Approximation for BME Soft-Data Integration

## Complete Mathematical Derivations

**Author:** Corinne Wiesner-Friedman  
**Date:** March 2026  
**Purpose:** Self-contained mathematical reference with every derivation step shown explicitly, suitable for review by a mathematician or statistician unfamiliar with the codebase.

---

## Table of Contents

1. [Notation and Preliminaries](#1-notation-and-preliminaries)
2. [Problem Statement](#2-problem-statement)
3. [The BME Integral](#3-the-bme-integral)
4. [Log-Target Function and Its Structure](#4-log-target-function-and-its-structure)
5. [The Laplace Approximation: Full Derivation](#5-the-laplace-approximation-full-derivation)
6. [Gradient of the Log-Target](#6-gradient-of-the-log-target)
7. [Hessian of the Log-Target](#7-hessian-of-the-log-target)
8. [Newton's Method for Mode-Finding](#8-newtons-method-for-mode-finding)
9. [Assembling the Final Laplace Formula](#9-assembling-the-final-laplace-formula)
10. [Analytic Derivatives for Specific PDFs](#10-analytic-derivatives-for-specific-pdfs)
    - 10.1 [Gaussian Soft PDF](#101-gaussian-soft-pdf)
    - 10.2 [Truncated Normal Soft PDF](#102-truncated-normal-soft-pdf)
    - 10.3 [Constant-Hessian Consequence](#103-constant-hessian-consequence)
11. [Finite-Difference Fallback for General PDFs](#11-finite-difference-fallback-for-general-pdfs)
12. [Vectorized Batch Laplace (Constant-Hessian Case)](#12-vectorized-batch-laplace-constant-hessian-case)
13. [Connection to BME Prediction: The Conditioning Chain](#13-connection-to-bme-prediction-the-conditioning-chain)
14. [Validation: Exact Analytic Test Cases](#14-validation-exact-analytic-test-cases)
15. [Error Analysis and Approximation Quality](#15-error-analysis-and-approximation-quality)
16. [Related Methods: EP and Laplace Importance Sampling](#16-related-methods-ep-and-laplace-importance-sampling)
17. [References](#17-references)

---

## 1. Notation and Preliminaries

We establish notation used throughout this document.

| Symbol | Meaning |
|--------|---------|
| $n_s$ | Number of soft-data locations |
| $n_h$ | Number of hard-data locations |
| $\mathbf{x} = (x_1, \ldots, x_{n_s})^\top$ | Unknown true values at soft-data locations |
| $\mathbf{z}_h = (z_1, \ldots, z_{n_h})^\top$ | Observed hard-data values |
| $z_k$ | Unknown value at a prediction location |
| $\boldsymbol{\mu} \in \mathbb{R}^{n_s}$ | Conditional mean of $\mathbf{x} \mid \mathbf{z}_h$ |
| $\boldsymbol{\Sigma} \in \mathbb{R}^{n_s \times n_s}$ | Conditional covariance of $\mathbf{x} \mid \mathbf{z}_h$ |
| $\mathbf{Q} = \boldsymbol{\Sigma}^{-1}$ | Precision matrix |
| $f_i(x_i)$ | Soft-data PDF for the $i$-th datum |
| $\phi(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$ | Multivariate Gaussian density |
| $\mathbf{x}^*$ | Mode (MAP point) of $\log g(\mathbf{x})$ |
| $\mathbf{H}(\mathbf{x})$ | Hessian matrix of $\log g$ at $\mathbf{x}$ |
| $|\mathbf{M}|$ | Determinant of a matrix $\mathbf{M}$ |

**Key background facts** used without proof:

**Fact 1 (Multivariate Gaussian density).** If $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ with $\boldsymbol{\Sigma}$ symmetric positive definite, then

$$
\phi(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = (2\pi)^{-n_s/2} |\boldsymbol{\Sigma}|^{-1/2} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
$$

Since $\mathbf{Q} = \boldsymbol{\Sigma}^{-1}$, we have $|\boldsymbol{\Sigma}|^{-1/2} = |\mathbf{Q}|^{1/2}$ (because $|\boldsymbol{\Sigma}^{-1}| = |\boldsymbol{\Sigma}|^{-1}$ for any invertible matrix, so $|\mathbf{Q}|^{1/2} = (|\boldsymbol{\Sigma}|^{-1})^{1/2} = |\boldsymbol{\Sigma}|^{-1/2}$). Thus:

$$
\phi(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{|\mathbf{Q}|^{1/2}}{(2\pi)^{n_s/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{Q} (\mathbf{x} - \boldsymbol{\mu})\right)
$$

**Fact 2 (Multivariate Gaussian integral).** For any symmetric positive-definite $\mathbf{A} \in \mathbb{R}^{n \times n}$ and any $\mathbf{c} \in \mathbb{R}^n$,

$$
\int_{\mathbb{R}^n} \exp\!\left(-\frac{1}{2} (\mathbf{x} - \mathbf{c})^\top \mathbf{A} (\mathbf{x} - \mathbf{c})\right) d\mathbf{x} = \frac{(2\pi)^{n/2}}{|\mathbf{A}|^{1/2}} \tag{GI}
$$

> *Proof of Fact 2.* The function $q(\mathbf{x}) \equiv \frac{|\mathbf{A}|^{1/2}}{(2\pi)^{n/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \mathbf{c})^\top \mathbf{A} (\mathbf{x} - \mathbf{c})\right)$ is the density of $\mathcal{N}(\mathbf{c}, \mathbf{A}^{-1})$. Since any probability density integrates to 1:
>
> $$
> 1 = \int_{\mathbb{R}^n} q(\mathbf{x})\,d\mathbf{x} = \frac{|\mathbf{A}|^{1/2}}{(2\pi)^{n/2}} \int_{\mathbb{R}^n} \exp\!\left(-\frac{1}{2}(\mathbf{x}-\mathbf{c})^\top \mathbf{A} (\mathbf{x}-\mathbf{c})\right) d\mathbf{x}
> $$
>
> Solving for the integral:
>
> $$
> \int_{\mathbb{R}^n} \exp\!\left(-\frac{1}{2}(\mathbf{x}-\mathbf{c})^\top \mathbf{A} (\mathbf{x}-\mathbf{c})\right) d\mathbf{x} = \frac{(2\pi)^{n/2}}{|\mathbf{A}|^{1/2}} \qquad \square
> $$

**Fact 3 (Derivative of a quadratic form).** For symmetric $\mathbf{Q}$ and constant $\boldsymbol{\mu}$,

$$
\frac{\partial}{\partial \mathbf{x}} \left[ (\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{Q} (\mathbf{x} - \boldsymbol{\mu}) \right] = 2\mathbf{Q}(\mathbf{x} - \boldsymbol{\mu})
$$

> *Proof of Fact 3.* Let $\mathbf{d} = \mathbf{x} - \boldsymbol{\mu}$. The quadratic form is $\mathbf{d}^\top \mathbf{Q} \mathbf{d} = \sum_{j,k} d_j Q_{jk} d_k$. Differentiating with respect to $d_m$ (equivalently $x_m$):
>
> $$
> \frac{\partial}{\partial d_m} \sum_{j,k} d_j Q_{jk} d_k = \sum_{k} Q_{mk} d_k + \sum_{j} d_j Q_{jm}
> $$
>
> The first sum is $[\mathbf{Q}\mathbf{d}]_m$ and the second is $[\mathbf{Q}^\top \mathbf{d}]_m = [\mathbf{Q}\mathbf{d}]_m$ (since $\mathbf{Q}$ is symmetric). So $\frac{\partial}{\partial d_m}(\mathbf{d}^\top \mathbf{Q} \mathbf{d}) = 2[\mathbf{Q}\mathbf{d}]_m$, and in vector form: $\nabla_{\mathbf{d}}(\mathbf{d}^\top \mathbf{Q} \mathbf{d}) = 2\mathbf{Q}\mathbf{d}$. $\square$

**Fact 4 (Second derivative of a quadratic form).** For symmetric $\mathbf{Q}$,

$$
\frac{\partial^2}{\partial \mathbf{x} \,\partial \mathbf{x}^\top} \left[ (\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{Q} (\mathbf{x} - \boldsymbol{\mu}) \right] = 2\mathbf{Q}
$$

> *Proof of Fact 4.* From Fact 3 the gradient is $2\mathbf{Q}(\mathbf{x} - \boldsymbol{\mu})$. Its $m$-th component is $2\sum_k Q_{mk}(x_k - \mu_k)$. Differentiating with respect to $x_l$: $\frac{\partial}{\partial x_l}[2\sum_k Q_{mk}(x_k-\mu_k)] = 2Q_{ml}$. So the $(m,l)$ entry of the Hessian is $2Q_{ml}$, which is the matrix $2\mathbf{Q}$. $\square$

---

## 2. Problem Statement

In Bayesian Maximum Entropy (BME) geostatistics, predictions at unobserved locations must incorporate **soft probabilistic data** — observations that are not exact values but probability distributions (e.g., detection limits, censored measurements, expert-elicited intervals, sensor uncertainty). This requires evaluating a high-dimensional integral that has no closed-form solution in general.

The Laplace approximation provides an $O(n_s^3)$ method for evaluating this integral, replacing the exponential-cost $O(n_q^{n_s})$ Gauss-Hermite quadrature when the number of soft data $n_s$ exceeds approximately 6.

---

## 3. The BME Integral

Let $\mathbf{x} = (x_1, \ldots, x_{n_s})^\top$ be the unknown true values at $n_s$ soft-data locations. Conditional on hard data $\mathbf{z}_h$, the BME framework models $\mathbf{x}$ as multivariate Gaussian:

$$
\mathbf{x} \mid \mathbf{z}_h \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

where $\boldsymbol{\mu} \in \mathbb{R}^{n_s}$ is the conditional (kriging) mean vector and $\boldsymbol{\Sigma} \in \mathbb{R}^{n_s \times n_s}$ is the conditional covariance matrix (both derived via kriging equations from the spatial covariance model and hard data).

Each soft datum $i$ contributes a **soft PDF** $f_i(x_i)$ expressing the probabilistic knowledge about the true value $x_i$. These may be:

- **Gaussian:** $f_i(x_i) = \mathcal{N}(x_i; \mu_i^s, \sigma_i^2)$ (measurement with known uncertainty)
- **Uniform / Interval:** $f_i(x_i) = \frac{1}{b_i - a_i} \mathbf{1}_{[a_i, b_i]}(x_i)$ (e.g., below-detection-limit censoring)
- **Truncated normal:** $f_i(x_i)$ (censored measurement with known uncertainty)
- **Lognormal, triangular, histogram, mixture, arbitrary callable**

The central integral that BME must evaluate is:

$$
I = \mathbb{E}_{\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})} \left[ \prod_{i=1}^{n_s} f_i(x_i) \right] \tag{1}
$$

Expanding the expectation using the multivariate Gaussian density $\phi$:

$$
I = \int_{\mathbb{R}^{n_s}} \left(\prod_{i=1}^{n_s} f_i(x_i)\right) \phi(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) \, d\mathbf{x} \tag{2}
$$

Substituting Fact 1:

$$
I = \int_{\mathbb{R}^{n_s}} \left(\prod_{i=1}^{n_s} f_i(x_i)\right) \frac{|\mathbf{Q}|^{1/2}}{(2\pi)^{n_s/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{Q}(\mathbf{x} - \boldsymbol{\mu})\right) d\mathbf{x} \tag{3}
$$

**Key structural property:** Each soft PDF $f_i$ depends only on $x_i$ (not on $x_j$, $j \neq i$). This **separability** is a fundamental property of BME soft data and will be exploited repeatedly below.

---

## 4. Log-Target Function and Its Structure

We now factor out the normalizing constant from the Gaussian density. From equation (3):

$$
I = \frac{|\mathbf{Q}|^{1/2}}{(2\pi)^{n_s/2}} \int_{\mathbb{R}^{n_s}} \left(\prod_{i=1}^{n_s} f_i(x_i)\right) \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{Q}(\mathbf{x} - \boldsymbol{\mu})\right) d\mathbf{x} \tag{4}
$$

Define the **unnormalized log-target** by combining the exponential and the product:

$$
g(\mathbf{x}) \;\stackrel{\text{def}}{=}\; \left(\prod_{i=1}^{n_s} f_i(x_i)\right) \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{Q}(\mathbf{x} - \boldsymbol{\mu})\right) \tag{5}
$$

Taking the logarithm (using $\log(\prod a_i) = \sum \log a_i$ and $\log(\exp(\cdot)) = \cdot$):

$$
\log g(\mathbf{x}) = \log\!\left(\prod_{i=1}^{n_s} f_i(x_i)\right) + \log\!\left(\exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \mathbf{Q}(\mathbf{x}-\boldsymbol{\mu})\right)\right)
$$

$$
= \sum_{i=1}^{n_s} \log f_i(x_i) + \left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \mathbf{Q}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

$$
\boxed{
\log g(\mathbf{x}) = -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{Q}(\mathbf{x} - \boldsymbol{\mu}) + \sum_{i=1}^{n_s} \log f_i(x_i)
} \tag{6}
$$

Now equation (4) becomes:

$$
I = \frac{|\mathbf{Q}|^{1/2}}{(2\pi)^{n_s/2}} \int_{\mathbb{R}^{n_s}} g(\mathbf{x}) \, d\mathbf{x} = \frac{|\mathbf{Q}|^{1/2}}{(2\pi)^{n_s/2}} \int_{\mathbb{R}^{n_s}} \exp\!\bigl(\log g(\mathbf{x})\bigr) \, d\mathbf{x} \tag{7}
$$

In (6), the two terms have distinct character:

- $-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{Q}(\mathbf{x} - \boldsymbol{\mu})$ is an **exactly quadratic** function of $\mathbf{x}$, coupling all dimensions through $\mathbf{Q}$.
- $\sum_{i=1}^{n_s} \log f_i(x_i)$ is a **separable** sum where each term depends on a single coordinate $x_i$ only.

This structure is what makes the Laplace approximation especially efficient for BME.

---

## 5. The Laplace Approximation: Full Derivation

We wish to approximate $\int \exp(\log g(\mathbf{x})) \, d\mathbf{x}$ in equation (7).

### Step 1: Multivariate Taylor expansion

Recall the **multivariate Taylor expansion** of a scalar function $h: \mathbb{R}^n \to \mathbb{R}$ about a point $\mathbf{a}$:

$$
h(\mathbf{x}) = h(\mathbf{a}) + \nabla h(\mathbf{a})^\top (\mathbf{x} - \mathbf{a}) + \frac{1}{2}(\mathbf{x} - \mathbf{a})^\top \nabla^2 h(\mathbf{a}) (\mathbf{x} - \mathbf{a}) + O(\|\mathbf{x} - \mathbf{a}\|^3)
$$

where $\nabla h(\mathbf{a}) \in \mathbb{R}^n$ is the gradient vector with components $[\nabla h]_j = \frac{\partial h}{\partial x_j}\big|_{\mathbf{a}}$, and $\nabla^2 h(\mathbf{a}) \in \mathbb{R}^{n \times n}$ is the Hessian matrix with entries $[\nabla^2 h]_{jk} = \frac{\partial^2 h}{\partial x_j \partial x_k}\big|_{\mathbf{a}}$.

### Step 2: Expand $\log g$ at the mode

Let $\mathbf{x}^* = \arg\max_{\mathbf{x}} \log g(\mathbf{x})$ be the mode. Applying the Taylor expansion with $h = \log g$ and $\mathbf{a} = \mathbf{x}^*$:

$$
\log g(\mathbf{x}) = \log g(\mathbf{x}^*) + \nabla \log g(\mathbf{x}^*)^\top (\mathbf{x} - \mathbf{x}^*) + \frac{1}{2}(\mathbf{x} - \mathbf{x}^*)^\top \mathbf{H}(\mathbf{x}^*) (\mathbf{x} - \mathbf{x}^*) + O(\|\mathbf{x} - \mathbf{x}^*\|^3)
\tag{8}
$$

where $\mathbf{H}(\mathbf{x}^*) = \nabla^2 \log g(\mathbf{x}^*)$ is the Hessian of $\log g$ evaluated at $\mathbf{x}^*$.

### Step 3: First-order term vanishes

Since $\mathbf{x}^*$ is a **local maximum** of $\log g$, the **necessary first-order condition** for an interior maximum gives:

$$
\nabla \log g(\mathbf{x}^*) = \mathbf{0} \tag{9}
$$

Therefore the linear term in (8) vanishes:

$$
\nabla \log g(\mathbf{x}^*)^\top (\mathbf{x} - \mathbf{x}^*) = \mathbf{0}^\top (\mathbf{x} - \mathbf{x}^*) = 0
$$

### Step 4: Negative semi-definiteness of the Hessian at the mode

The **necessary second-order condition** for $\mathbf{x}^*$ to be a local maximum is that the Hessian is **negative semi-definite** at the mode:

$$
\mathbf{H}(\mathbf{x}^*) \preceq \mathbf{0} \tag{10}
$$

That is, $\mathbf{v}^\top \mathbf{H}(\mathbf{x}^*) \mathbf{v} \leq 0$ for all $\mathbf{v} \in \mathbb{R}^{n_s}$. (If the maximum is strict and non-degenerate, then $\mathbf{H}(\mathbf{x}^*) \prec \mathbf{0}$, i.e., strictly negative definite.)

### Step 5: Truncated Taylor expansion

Dropping the $O(\|\mathbf{x} - \mathbf{x}^*\|^3)$ remainder — this is the **Laplace approximation** — we obtain:

$$
\log g(\mathbf{x}) \approx \log g(\mathbf{x}^*) + \frac{1}{2}(\mathbf{x} - \mathbf{x}^*)^\top \mathbf{H}(\mathbf{x}^*)(\mathbf{x} - \mathbf{x}^*) \tag{11}
$$

### Step 6: Exponentiate

Exponentiating both sides of (11):

$$
g(\mathbf{x}) = \exp(\log g(\mathbf{x})) \approx \exp\!\left(\log g(\mathbf{x}^*) + \frac{1}{2}(\mathbf{x} - \mathbf{x}^*)^\top \mathbf{H}(\mathbf{x}^*)(\mathbf{x} - \mathbf{x}^*)\right)
$$

Using $\exp(a + b) = \exp(a)\exp(b)$:

$$
g(\mathbf{x}) \approx \exp(\log g(\mathbf{x}^*)) \cdot \exp\!\left(\frac{1}{2}(\mathbf{x} - \mathbf{x}^*)^\top \mathbf{H}(\mathbf{x}^*)(\mathbf{x} - \mathbf{x}^*)\right) \tag{12}
$$

### Step 7: Substitute into the integral

$$
\int_{\mathbb{R}^{n_s}} g(\mathbf{x}) \, d\mathbf{x} \approx \exp(\log g(\mathbf{x}^*)) \int_{\mathbb{R}^{n_s}} \exp\!\left(\frac{1}{2}(\mathbf{x} - \mathbf{x}^*)^\top \mathbf{H}(\mathbf{x}^*)(\mathbf{x} - \mathbf{x}^*)\right) d\mathbf{x} \tag{13}
$$

Note that $\exp(\log g(\mathbf{x}^*))$ is a constant (independent of $\mathbf{x}$) and comes out of the integral.

### Step 8: Convert to standard Gaussian integral form

Define $\mathbf{A} \stackrel{\text{def}}{=} -\mathbf{H}(\mathbf{x}^*)$. From (10), $\mathbf{H}(\mathbf{x}^*) \preceq \mathbf{0}$, so $\mathbf{A} = -\mathbf{H}(\mathbf{x}^*) \succeq \mathbf{0}$. We assume $\mathbf{A}$ is strictly positive definite (non-degenerate case), i.e., $\mathbf{A} \succ \mathbf{0}$.

Now rewrite the exponent in (13):

$$
\frac{1}{2}(\mathbf{x} - \mathbf{x}^*)^\top \mathbf{H}(\mathbf{x}^*)(\mathbf{x} - \mathbf{x}^*) = \frac{1}{2}(\mathbf{x} - \mathbf{x}^*)^\top (-\mathbf{A})(\mathbf{x} - \mathbf{x}^*) = -\frac{1}{2}(\mathbf{x} - \mathbf{x}^*)^\top \mathbf{A}(\mathbf{x} - \mathbf{x}^*)
$$

Substituting into (13):

$$
\int_{\mathbb{R}^{n_s}} g(\mathbf{x}) \, d\mathbf{x} \approx \exp(\log g(\mathbf{x}^*)) \int_{\mathbb{R}^{n_s}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \mathbf{x}^*)^\top \mathbf{A}(\mathbf{x} - \mathbf{x}^*)\right) d\mathbf{x} \tag{14}
$$

### Step 9: Evaluate the Gaussian integral

The integral in (14) has exactly the form of Fact 2 (GI) with $\mathbf{c} = \mathbf{x}^*$, $n = n_s$:

$$
\int_{\mathbb{R}^{n_s}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \mathbf{x}^*)^\top \mathbf{A}(\mathbf{x} - \mathbf{x}^*)\right) d\mathbf{x} = \frac{(2\pi)^{n_s/2}}{|\mathbf{A}|^{1/2}} \tag{15}
$$

### Step 10: Combine

Substituting (15) into (14):

$$
\int_{\mathbb{R}^{n_s}} g(\mathbf{x}) \, d\mathbf{x} \approx \exp(\log g(\mathbf{x}^*)) \cdot \frac{(2\pi)^{n_s/2}}{|\mathbf{A}|^{1/2}} \tag{16}
$$

Recalling $\mathbf{A} = -\mathbf{H}(\mathbf{x}^*)$:

$$
\int_{\mathbb{R}^{n_s}} g(\mathbf{x}) \, d\mathbf{x} \approx \frac{(2\pi)^{n_s/2}}{|-\mathbf{H}(\mathbf{x}^*)|^{1/2}} \cdot \exp(\log g(\mathbf{x}^*)) \tag{17}
$$

### Step 11: Final form of the Laplace-approximated BME integral

Substitute (17) into (7):

$$
I = \frac{|\mathbf{Q}|^{1/2}}{(2\pi)^{n_s/2}} \int_{\mathbb{R}^{n_s}} g(\mathbf{x})\,d\mathbf{x} \approx \frac{|\mathbf{Q}|^{1/2}}{(2\pi)^{n_s/2}} \cdot \frac{(2\pi)^{n_s/2}}{|-\mathbf{H}(\mathbf{x}^*)|^{1/2}} \cdot \exp(\log g(\mathbf{x}^*))
$$

The $(2\pi)^{n_s/2}$ in the numerator and denominator cancel:

$$
\frac{|\mathbf{Q}|^{1/2}}{\cancel{(2\pi)^{n_s/2}}} \cdot \frac{\cancel{(2\pi)^{n_s/2}}}{|-\mathbf{H}(\mathbf{x}^*)|^{1/2}} = \frac{|\mathbf{Q}|^{1/2}}{|-\mathbf{H}(\mathbf{x}^*)|^{1/2}}
$$

Therefore:

$$
\boxed{
I \approx \frac{|\mathbf{Q}|^{1/2}}{|-\mathbf{H}(\mathbf{x}^*)|^{1/2}} \cdot \exp(\log g(\mathbf{x}^*))
} \tag{18}
$$

This is the Laplace approximation to the BME integral. Computing it requires three things:

1. The mode $\mathbf{x}^*$ (Section 8).
2. The Hessian $\mathbf{H}(\mathbf{x}^*)$ at the mode (Section 7).
3. The value $\log g(\mathbf{x}^*)$ at the mode (equation 6).

---

## 6. Gradient of the Log-Target

We derive $\nabla \log g(\mathbf{x})$ component by component, then assemble the vector.

**Starting point:** From equation (6),

$$
\log g(\mathbf{x}) = \underbrace{-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{Q}(\mathbf{x} - \boldsymbol{\mu})}_{\text{Term A}} + \underbrace{\sum_{i=1}^{n_s} \log f_i(x_i)}_{\text{Term B}}
$$

We compute $\frac{\partial}{\partial x_j} \log g(\mathbf{x})$ for an arbitrary component $j \in \{1, \ldots, n_s\}$.

### Gradient of Term A

We need $\frac{\partial}{\partial x_j}\bigl[-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{Q}(\mathbf{x} - \boldsymbol{\mu})\bigr]$.

By Fact 3, $\nabla_{\mathbf{x}}\bigl[(\mathbf{x}-\boldsymbol{\mu})^\top \mathbf{Q}(\mathbf{x}-\boldsymbol{\mu})\bigr] = 2\mathbf{Q}(\mathbf{x}-\boldsymbol{\mu})$, so:

$$
\nabla_{\mathbf{x}} \left[-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \mathbf{Q}(\mathbf{x}-\boldsymbol{\mu})\right] = -\frac{1}{2} \cdot 2\mathbf{Q}(\mathbf{x}-\boldsymbol{\mu}) = -\mathbf{Q}(\mathbf{x}-\boldsymbol{\mu})
$$

The $j$-th component is:

$$
\frac{\partial}{\partial x_j}\text{(Term A)} = -\bigl[\mathbf{Q}(\mathbf{x}-\boldsymbol{\mu})\bigr]_j = -\sum_{k=1}^{n_s} Q_{jk}(x_k - \mu_k) \tag{19}
$$

### Gradient of Term B

We need $\frac{\partial}{\partial x_j}\bigl[\sum_{i=1}^{n_s} \log f_i(x_i)\bigr]$.

Since each $f_i$ depends only on $x_i$:

$$
\frac{\partial}{\partial x_j} \log f_i(x_i) = \begin{cases}
\frac{d}{dx_j} \log f_j(x_j) & \text{if } i = j \\
0 & \text{if } i \neq j
\end{cases}
$$

*Justification:* When $i \neq j$, $f_i(x_i)$ does not involve $x_j$ at all, so $\frac{\partial}{\partial x_j} \log f_i(x_i) = 0$. When $i = j$, we simply have the ordinary derivative $\frac{d}{dx_j} \log f_j(x_j)$.

Summing over $i$:

$$
\frac{\partial}{\partial x_j} \sum_{i=1}^{n_s} \log f_i(x_i) = \sum_{i=1}^{n_s} \frac{\partial}{\partial x_j} \log f_i(x_i) = 0 + \cdots + 0 + \frac{d}{dx_j}\log f_j(x_j) + 0 + \cdots + 0 = \frac{d}{dx_j}\log f_j(x_j) \tag{20}
$$

### Assembling the gradient

Combining (19) and (20) for the $j$-th component:

$$
\frac{\partial}{\partial x_j} \log g(\mathbf{x}) = -\sum_{k=1}^{n_s} Q_{jk}(x_k - \mu_k) + \frac{d}{dx_j} \log f_j(x_j)
$$

In vector form, stacking all $n_s$ components:

$$
\boxed{
\nabla \log g(\mathbf{x}) = -\mathbf{Q}(\mathbf{x} - \boldsymbol{\mu}) + \mathbf{s}'(\mathbf{x})
} \tag{21}
$$

where we define the **soft-gradient vector**:

$$
\mathbf{s}'(\mathbf{x}) = \begin{pmatrix} \frac{d}{dx_1} \log f_1(x_1) \\ \frac{d}{dx_2} \log f_2(x_2) \\ \vdots \\ \frac{d}{dx_{n_s}} \log f_{n_s}(x_{n_s}) \end{pmatrix} \in \mathbb{R}^{n_s}
$$

**Observation:** The gradient has two parts: the first ($-\mathbf{Q}(\mathbf{x}-\boldsymbol{\mu})$) couples all dimensions through $\mathbf{Q}$; the second ($\mathbf{s}'(\mathbf{x})$) is a vector whose $j$-th entry depends only on $x_j$.

---

## 7. Hessian of the Log-Target

We derive $[\nabla^2 \log g(\mathbf{x})]_{jk} = \frac{\partial^2}{\partial x_j \partial x_k} \log g(\mathbf{x})$ by differentiating the gradient (21) a second time.

### Hessian of Term A

By Fact 4:

$$
\frac{\partial^2}{\partial x_j \partial x_k}\left[-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \mathbf{Q}(\mathbf{x}-\boldsymbol{\mu})\right] = -\frac{1}{2} \cdot 2Q_{jk} = -Q_{jk} \tag{22}
$$

In matrix form, the Hessian of Term A is $-\mathbf{Q}$.

### Hessian of Term B

We need $\frac{\partial^2}{\partial x_j \partial x_k}\bigl[\sum_{i=1}^{n_s} \log f_i(x_i)\bigr]$. From equation (20), the first derivative with respect to $x_j$ is $\frac{d}{dx_j}\log f_j(x_j)$. Now differentiate this with respect to $x_k$:

**Case $k \neq j$:**

$$
\frac{\partial}{\partial x_k}\left[\frac{d}{dx_j} \log f_j(x_j)\right] = 0
$$

because $\frac{d}{dx_j} \log f_j(x_j)$ is a function of $x_j$ only and does not depend on $x_k$.

**Case $k = j$:**

$$
\frac{\partial}{\partial x_j}\left[\frac{d}{dx_j} \log f_j(x_j)\right] = \frac{d^2}{dx_j^2} \log f_j(x_j)
$$

Combining:

$$
\frac{\partial^2}{\partial x_j \partial x_k}\left[\sum_{i=1}^{n_s} \log f_i(x_i)\right] = \begin{cases} \frac{d^2}{dx_j^2} \log f_j(x_j) & \text{if } j = k \\ 0 & \text{if } j \neq k \end{cases} \tag{23}
$$

This is a **diagonal matrix**. In matrix notation:

$$
\nabla^2 \text{(Term B)} = \operatorname{diag}\!\left(\frac{d^2}{dx_1^2}\log f_1(x_1), \;\frac{d^2}{dx_2^2}\log f_2(x_2), \;\ldots,\; \frac{d^2}{dx_{n_s}^2}\log f_{n_s}(x_{n_s})\right) \tag{24}
$$

The diagonality comes directly from the separability of the soft PDFs: no cross-terms $\frac{\partial^2}{\partial x_j \partial x_k}$ survive for $j \neq k$.

### Assembling the Hessian

Adding (22) and (24):

$$
\boxed{
\mathbf{H}(\mathbf{x}) = \nabla^2 \log g(\mathbf{x}) = -\mathbf{Q} + \operatorname{diag}(\mathbf{s}''(\mathbf{x}))
} \tag{25}
$$

where we define the **soft-curvature vector**:

$$
\mathbf{s}''(\mathbf{x}) = \begin{pmatrix} \frac{d^2}{dx_1^2}\log f_1(x_1) \\ \vdots \\ \frac{d^2}{dx_{n_s}^2}\log f_{n_s}(x_{n_s}) \end{pmatrix} \in \mathbb{R}^{n_s}
$$

### Structural properties of the Hessian

1. **Decomposition:** $\mathbf{H} = -\mathbf{Q} + \mathbf{D}(\mathbf{x})$ where $\mathbf{D}(\mathbf{x}) = \operatorname{diag}(\mathbf{s}''(\mathbf{x}))$ is diagonal.

2. **Storage cost:** $O(n_s^2)$ for $\mathbf{Q}$ (which is fixed) plus $O(n_s)$ for the diagonal.

3. **Dependence on $\mathbf{x}$:** Only $\mathbf{D}(\mathbf{x})$ depends on $\mathbf{x}$; the $-\mathbf{Q}$ part is a constant. If each $s''_i$ is itself constant (e.g., for Gaussian or truncated-normal PDFs — see Section 10), then $\mathbf{H}$ is independent of $\mathbf{x}$ entirely.

4. **Negative definite $-\mathbf{H}$:** At the mode, $-\mathbf{H}(\mathbf{x}^*) = \mathbf{Q} - \operatorname{diag}(\mathbf{s}''(\mathbf{x}^*))$. Since $\mathbf{Q} \succ 0$ (covariance inverse) and typically $s''_i \leq 0$ (log-concave soft PDFs), we have $-\operatorname{diag}(\mathbf{s}'') \succeq 0$, so $-\mathbf{H} = \mathbf{Q} + |-\operatorname{diag}(\mathbf{s}'')| \succ 0$.

---

## 8. Newton's Method for Mode-Finding

We must find $\mathbf{x}^* = \arg\max_{\mathbf{x}} \log g(\mathbf{x})$, equivalently the root of $\nabla \log g(\mathbf{x}) = \mathbf{0}$.

### Derivation of the Newton update

Newton's method for finding a root of a vector-valued function $\mathbf{F}(\mathbf{x}) = \mathbf{0}$ linearizes $\mathbf{F}$ around the current iterate $\mathbf{x}^{(t)}$:

$$
\mathbf{F}(\mathbf{x}) \approx \mathbf{F}(\mathbf{x}^{(t)}) + \mathbf{J}(\mathbf{x}^{(t)})(\mathbf{x} - \mathbf{x}^{(t)})
$$

where $\mathbf{J}(\mathbf{x}^{(t)}) = \frac{\partial \mathbf{F}}{\partial \mathbf{x}}\big|_{\mathbf{x}^{(t)}}$ is the Jacobian. Setting the right side to $\mathbf{0}$ and solving:

$$
\mathbf{0} = \mathbf{F}(\mathbf{x}^{(t)}) + \mathbf{J}(\mathbf{x}^{(t)})(\mathbf{x}^{(t+1)} - \mathbf{x}^{(t)})
$$

$$
\mathbf{J}(\mathbf{x}^{(t)})(\mathbf{x}^{(t+1)} - \mathbf{x}^{(t)}) = -\mathbf{F}(\mathbf{x}^{(t)})
$$

$$
\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - [\mathbf{J}(\mathbf{x}^{(t)})]^{-1} \mathbf{F}(\mathbf{x}^{(t)}) \tag{26}
$$

In our case, $\mathbf{F}(\mathbf{x}) = \nabla \log g(\mathbf{x})$ and $\mathbf{J}(\mathbf{x}) = \nabla^2 \log g(\mathbf{x}) = \mathbf{H}(\mathbf{x})$. Substituting into (26):

$$
\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - [\mathbf{H}(\mathbf{x}^{(t)})]^{-1} \nabla \log g(\mathbf{x}^{(t)}) \tag{27}
$$

### Equivalent linear-system form

Defining the **Newton step** $\boldsymbol{\delta}^{(t)} = \mathbf{x}^{(t+1)} - \mathbf{x}^{(t)}$, equation (27) becomes:

$$
\mathbf{H}(\mathbf{x}^{(t)}) \, \boldsymbol{\delta}^{(t)} = -\nabla \log g(\mathbf{x}^{(t)})
$$

Multiplying both sides by $-1$ (since $-\mathbf{H} \succ 0$ is easier to work with):

$$
\bigl(-\mathbf{H}(\mathbf{x}^{(t)})\bigr) \, \boldsymbol{\delta}^{(t)} = \nabla \log g(\mathbf{x}^{(t)}) \tag{28}
$$

Then $\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} + \boldsymbol{\delta}^{(t)}$.

Equation (28) is an $n_s \times n_s$ linear system. If $-\mathbf{H}$ is positive definite (which is guaranteed at a strict maximum), it has a unique solution.

### Expanding the Newton step using the Hessian structure

Substituting $\mathbf{H} = -\mathbf{Q} + \operatorname{diag}(\mathbf{s}'')$ from (25) into (28):

$$
\bigl(\mathbf{Q} - \operatorname{diag}(\mathbf{s}''(\mathbf{x}^{(t)}))\bigr) \boldsymbol{\delta}^{(t)} = \nabla \log g(\mathbf{x}^{(t)})
$$

And substituting $\nabla \log g = -\mathbf{Q}(\mathbf{x}^{(t)} - \boldsymbol{\mu}) + \mathbf{s}'(\mathbf{x}^{(t)})$ from (21):

$$
\bigl(\mathbf{Q} - \operatorname{diag}(\mathbf{s}'')\bigr) \boldsymbol{\delta}^{(t)} = -\mathbf{Q}(\mathbf{x}^{(t)} - \boldsymbol{\mu}) + \mathbf{s}'(\mathbf{x}^{(t)}) \tag{29}
$$

This is the explicit form of the linear system solved at each Newton iteration. When $\mathbf{s}''$ is constant (Section 10.3), the left-hand side matrix is the same at every iteration, so it can be factored once.

### Practical modifications

The plain Newton step (27) may fail to improve $\log g$ if the quadratic approximation is poor far from the mode. The implementation uses:

**1. Initialization:**

$$
\mathbf{x}^{(0)} = \operatorname{clip}(\boldsymbol{\mu}, \mathbf{a}, \mathbf{b}) \qquad \text{where } a_i = \inf\operatorname{supp}(f_i),\; b_i = \sup\operatorname{supp}(f_i)
$$

The clip ensures the starting point lies within the support of every soft PDF.

**2. Support clamping at every step:**

$$
\mathbf{x}^{(t+1)} = \operatorname{clip}\bigl(\mathbf{x}^{(t)} + \alpha\boldsymbol{\delta}^{(t)},\; \mathbf{a},\; \mathbf{b}\bigr) \tag{30}
$$

This prevents iterates from reaching regions where some $f_i(x_i) = 0$, which would give $\log f_i = -\infty$.

**3. Backtracking line search:**

Instead of always taking a full step ($\alpha = 1$), the algorithm uses $\alpha \in \{1, \frac{1}{2}, \frac{1}{4}, \ldots, 2^{-20}\}$ and selects the largest $\alpha$ satisfying the **acceptance condition**:

$$
\log g\bigl(\operatorname{clip}(\mathbf{x}^{(t)} + \alpha \boldsymbol{\delta}^{(t)}, \mathbf{a}, \mathbf{b})\bigr) > \log g(\mathbf{x}^{(t)}) - 10^{-4} \tag{31}
$$

This is a relaxed acceptance threshold: it accepts any step that does not decrease $\log g$ by more than $10^{-4}$. (Note: this is *not* the standard Armijo sufficient-decrease condition, which would require a decrease proportional to $\alpha \nabla \log g^\top \boldsymbol{\delta}$. The simpler threshold used here is adequate because Newton steps near the mode are nearly exact, and the support clamping (30) already constrains the search space.)

**4. Hessian regularization:**

If $-\mathbf{H}(\mathbf{x}^{(t)})$ is not positive definite (which can happen away from the mode or for non-log-concave PDFs), the method computes the smallest eigenvalue $\lambda_{\min}$ of $-\mathbf{H}$ and adds a diagonal shift:

$$
-\mathbf{H}_{\text{reg}} = -\mathbf{H} + \bigl(\max(0, -\lambda_{\min}) + 10^{-4}\bigr)\mathbf{I} \tag{32}
$$

This ensures $-\mathbf{H}_{\text{reg}} \succ 0$, making the linear system (28) solvable and the step direction well-defined.

**5. Convergence:**

The iteration stops when $\|\alpha \boldsymbol{\delta}^{(t)}\|_2 < 10^{-6}$ or after 30 iterations.

---

## 9. Assembling the Final Laplace Formula

We now combine the results of Sections 5, 6, and 7 into the formula actually implemented in code.

### Step-by-step derivation of the log-space formula

Starting from equation (18):

$$
I \approx \frac{|\mathbf{Q}|^{1/2}}{|-\mathbf{H}(\mathbf{x}^*)|^{1/2}} \cdot \exp(\log g(\mathbf{x}^*))
$$

Take the logarithm of both sides:

$$
\log I \approx \log\!\left(\frac{|\mathbf{Q}|^{1/2}}{|-\mathbf{H}(\mathbf{x}^*)|^{1/2}}\right) + \log g(\mathbf{x}^*) \tag{33}
$$

Expand the log of the ratio using $\log(a/b) = \log a - \log b$:

$$
\log\!\left(\frac{|\mathbf{Q}|^{1/2}}{|-\mathbf{H}|^{1/2}}\right) = \log |\mathbf{Q}|^{1/2} - \log |-\mathbf{H}|^{1/2}
$$

Using $\log(a^{1/2}) = \frac{1}{2}\log a$:

$$
= \frac{1}{2}\log|\mathbf{Q}| - \frac{1}{2}\log|-\mathbf{H}(\mathbf{x}^*)|
$$

Substituting back into (33):

$$
\boxed{
\log I \approx \log g(\mathbf{x}^*) + \frac{1}{2}\bigl(\log|\mathbf{Q}| - \log|-\mathbf{H}(\mathbf{x}^*)|\bigr)
} \tag{34}
$$

Then $I = \exp(\log I)$, clamped to $[\,10^{-300},\; e^{500}\,]$ for numerical safety.

### Why log-space?

Computing via (34) rather than (18) directly avoids overflow/underflow:

- $\log g(\mathbf{x}^*)$ can be a large negative number (e.g., $-200$), which would underflow in direct exponential.
- The log-determinants can be large, but their difference is moderate.
- `numpy.linalg.slogdet(M)` returns $(s, \log|s \cdot \det(M)|)$ without ever computing the determinant itself, which could easily overflow for large matrices.

### Physical interpretation of each term

| Term | Meaning |
|------|---------|
| $\log g(\mathbf{x}^*)$ | Height of the log-integrand at its peak. Larger = soft data is more consistent with the prior. |
| $\frac{1}{2}\log\lvert\mathbf{Q}\rvert$ | Prior precision contribution. More precise prior (larger $\lvert\mathbf{Q}\rvert$) increases $I$ because the prior concentrates mass where soft PDFs are evaluated. |
| $-\frac{1}{2}\log\lvert-\mathbf{H}\rvert$ | "Width" of the peak. A broader peak (smaller $\lvert-\mathbf{H}\rvert$, i.e., less curvature) gives a larger integral because more volume concentrates near the mode. |

---

## 10. Analytic Derivatives for Specific PDFs

For the Newton solver (Section 8), we need $\frac{d}{dx_i}\log f_i(x_i)$ and $\frac{d^2}{dx_i^2}\log f_i(x_i)$. For certain PDF families, these have closed-form expressions, allowing us to avoid finite differences entirely.

### 10.1 Gaussian Soft PDF

For $f_i(x_i) = \mathcal{N}(x_i; \mu_i^s, \sigma_i^2)$ where $\mu_i^s$ is the soft-data mean and $\sigma_i^2$ is the soft-data variance:

$$
f_i(x_i) = \frac{1}{\sigma_i \sqrt{2\pi}} \exp\!\left(-\frac{(x_i - \mu_i^s)^2}{2\sigma_i^2}\right) \tag{35}
$$

#### Log-PDF derivation

$$
\log f_i(x_i) = \log\!\left(\frac{1}{\sigma_i\sqrt{2\pi}}\right) + \log\!\left(\exp\!\left(-\frac{(x_i - \mu_i^s)^2}{2\sigma_i^2}\right)\right)
$$

$$
= \log(1) - \log(\sigma_i\sqrt{2\pi}) - \frac{(x_i - \mu_i^s)^2}{2\sigma_i^2}
$$

Using $\log(\sigma_i\sqrt{2\pi}) = \log\sigma_i + \frac{1}{2}\log(2\pi)$:

$$
\log f_i(x_i) = -\frac{(x_i - \mu_i^s)^2}{2\sigma_i^2} - \log\sigma_i - \frac{1}{2}\log(2\pi) \tag{36}
$$

We can write this more compactly by defining $z_i = \frac{x_i - \mu_i^s}{\sigma_i}$:

$$
\log f_i(x_i) = -\frac{1}{2}z_i^2 - \log(\sigma_i\sqrt{2\pi}) \tag{37}
$$

#### First derivative

Differentiate (36) with respect to $x_i$. The terms $-\log\sigma_i - \frac{1}{2}\log(2\pi)$ are constants (independent of $x_i$), so:

$$
\frac{d}{dx_i}\log f_i(x_i) = \frac{d}{dx_i}\left[-\frac{(x_i - \mu_i^s)^2}{2\sigma_i^2}\right]
$$

Let $u = x_i - \mu_i^s$. Then $\frac{du}{dx_i} = 1$ and:

$$
\frac{d}{dx_i}\left[-\frac{u^2}{2\sigma_i^2}\right] = -\frac{1}{2\sigma_i^2} \cdot \frac{d}{du}(u^2) \cdot \frac{du}{dx_i} = -\frac{1}{2\sigma_i^2} \cdot 2u \cdot 1 = -\frac{u}{\sigma_i^2}
$$

Substituting back $u = x_i - \mu_i^s$:

$$
\boxed{\frac{d}{dx_i}\log f_i(x_i) = -\frac{x_i - \mu_i^s}{\sigma_i^2}} \tag{38}
$$

#### Second derivative

Differentiate (38) with respect to $x_i$:

$$
\frac{d^2}{dx_i^2}\log f_i(x_i) = \frac{d}{dx_i}\left[-\frac{x_i - \mu_i^s}{\sigma_i^2}\right] = -\frac{1}{\sigma_i^2} \cdot \frac{d}{dx_i}(x_i - \mu_i^s) = -\frac{1}{\sigma_i^2} \cdot 1
$$

$$
\boxed{\frac{d^2}{dx_i^2}\log f_i(x_i) = -\frac{1}{\sigma_i^2}} \tag{39}
$$

This is a **constant** — it does not depend on $x_i$. This fact will be crucial in Section 10.3.

### 10.2 Truncated Normal Soft PDF

For $f_i(x_i) = \text{TruncNorm}(x_i; \mu_i^s, \sigma_i^2, a_i, b_i)$ supported on $[a_i, b_i]$:

$$
f_i(x_i) = \begin{cases} \frac{\phi\!\left(\frac{x_i - \mu_i^s}{\sigma_i}\right)}{\sigma_i \, Z_i} & \text{if } a_i \leq x_i \leq b_i \\ 0 & \text{otherwise} \end{cases} \tag{40}
$$

> **Domain note.** The derivative identities (44)–(45) derived below hold on the **open interior** $a_i < x_i < b_i$. At the boundary points $x_i = a_i$ or $x_i = b_i$, $f_i$ drops discontinuously to zero, so $\log f_i$ is not differentiable. In practice the Newton iterates are clamped strictly inside the support by (30), so this is not a concern.

where $\phi(t) = \frac{1}{\sqrt{2\pi}}e^{-t^2/2}$ is the standard normal PDF and

$$
Z_i = \Phi(\beta_i) - \Phi(\alpha_i) \tag{41}
$$

is the normalization constant, with $\Phi$ the standard normal CDF and

$$
\alpha_i = \frac{a_i - \mu_i^s}{\sigma_i}, \qquad \beta_i = \frac{b_i - \mu_i^s}{\sigma_i} \tag{42}
$$

#### Why $Z_i$ does not depend on $x_i$

This is the key observation. The truncation bounds $a_i$ and $b_i$ are fixed constants (properties of the soft datum). The parameters $\mu_i^s$ and $\sigma_i$ are also fixed. Therefore $\alpha_i$, $\beta_i$, and $Z_i$ are all **constants with respect to $x_i$**.

$Z_i$ is simply the probability that a $\mathcal{N}(\mu_i^s, \sigma_i^2)$ random variable falls in $[a_i, b_i]$:

$$
Z_i = \Pr(a_i \leq X \leq b_i) \quad \text{where } X \sim \mathcal{N}(\mu_i^s, \sigma_i^2)
$$

#### Log-PDF derivation

For $x_i \in [a_i, b_i]$, starting from (40):

$$
f_i(x_i) = \frac{1}{\sigma_i Z_i} \cdot \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{1}{2}\left(\frac{x_i - \mu_i^s}{\sigma_i}\right)^2\right)
$$

Taking the log:

$$
\log f_i(x_i) = \log\!\left(\frac{1}{\sigma_i Z_i}\right) + \log\!\left(\frac{1}{\sqrt{2\pi}}\right) + \left(-\frac{1}{2}\left(\frac{x_i - \mu_i^s}{\sigma_i}\right)^2\right)
$$

$$
= -\log\sigma_i - \log Z_i - \frac{1}{2}\log(2\pi) - \frac{(x_i - \mu_i^s)^2}{2\sigma_i^2}
$$

Grouping the constants (everything that does not depend on $x_i$):

$$
\log f_i(x_i) = -\frac{(x_i - \mu_i^s)^2}{2\sigma_i^2} - \underbrace{\bigl(\log\sigma_i + \tfrac{1}{2}\log(2\pi) + \log Z_i\bigr)}_{\text{constant } C_i} \tag{43}
$$

Compare with the Gaussian case (36):

$$
\log f_i^{\text{Gauss}}(x_i) = -\frac{(x_i - \mu_i^s)^2}{2\sigma_i^2} - \underbrace{\bigl(\log\sigma_i + \tfrac{1}{2}\log(2\pi)\bigr)}_{\text{constant}}
$$

The **only difference** is the additional $-\log Z_i$ in the constant. The $x_i$-dependent part is identical.

#### First derivative

Since $C_i$ does not depend on $x_i$, differentiating (43):

$$
\frac{d}{dx_i}\log f_i(x_i) = \frac{d}{dx_i}\left[-\frac{(x_i - \mu_i^s)^2}{2\sigma_i^2}\right] - \frac{d}{dx_i}(C_i)
$$

$$
= -\frac{x_i - \mu_i^s}{\sigma_i^2} - 0 \tag{by identical calculation to (38)}
$$

$$
\boxed{\frac{d}{dx_i}\log f_i(x_i) = -\frac{x_i - \mu_i^s}{\sigma_i^2}} \tag{44}
$$

Identical to the Gaussian case (38).

#### Second derivative

Differentiating (44):

$$
\boxed{\frac{d^2}{dx_i^2}\log f_i(x_i) = -\frac{1}{\sigma_i^2}} \tag{45}
$$

Again a **constant**, identical to the Gaussian case (39).

#### Intuition

The truncation affects only the normalization $Z_i$, which is a multiplicative constant of $f_i$. Since $\log(c \cdot h(x)) = \log c + \log h(x)$, the constant $\log Z_i$ drops out under differentiation. The curvature of $\log f_i$ comes entirely from the Gaussian kernel $\phi$, not from the truncation.

### 10.3 Constant-Hessian Consequence

**Theorem.** When every soft PDF $f_i$ is either Gaussian or truncated-normal with parameters $(\mu_i^s, \sigma_i^2)$, the Hessian $\mathbf{H}$ is independent of $\mathbf{x}$.

*Proof.* From (25):

$$
\mathbf{H}(\mathbf{x}) = -\mathbf{Q} + \operatorname{diag}\!\left(\frac{d^2}{dx_1^2}\log f_1(x_1), \ldots, \frac{d^2}{dx_{n_s}^2}\log f_{n_s}(x_{n_s})\right)
$$

By (39) and (45), each diagonal entry is $-1/\sigma_i^2$ regardless of $x_i$. Therefore:

$$
\mathbf{H} = -\mathbf{Q} + \operatorname{diag}\!\left(-\frac{1}{\sigma_1^2}, \ldots, -\frac{1}{\sigma_{n_s}^2}\right) = -\mathbf{Q} - \operatorname{diag}(\boldsymbol{\sigma}^{-2}) \tag{46}
$$

where $\boldsymbol{\sigma}^{-2} = (1/\sigma_1^2, \ldots, 1/\sigma_{n_s}^2)^\top$. No dependence on $\mathbf{x}$ remains. $\square$

The negative Hessian is therefore:

$$
-\mathbf{H} = \mathbf{Q} + \operatorname{diag}(\boldsymbol{\sigma}^{-2}) \tag{47}
$$

**Positive definiteness of $-\mathbf{H}$:** Since $\mathbf{Q} = \boldsymbol{\Sigma}^{-1} \succ 0$ (inverse of a positive-definite matrix) and $\operatorname{diag}(\boldsymbol{\sigma}^{-2}) \succ 0$ (diagonal with strictly positive entries), their sum is strictly positive definite:

For any $\mathbf{v} \neq \mathbf{0}$:

$$
\mathbf{v}^\top (-\mathbf{H})\mathbf{v} = \mathbf{v}^\top \mathbf{Q}\mathbf{v} + \mathbf{v}^\top \operatorname{diag}(\boldsymbol{\sigma}^{-2})\mathbf{v} = \underbrace{\mathbf{v}^\top \mathbf{Q}\mathbf{v}}_{> 0} + \underbrace{\sum_i v_i^2/\sigma_i^2}_{> 0} > 0
$$

So $-\mathbf{H} \succ 0$ is guaranteed, and no regularization (32) is ever needed.

**Computational consequence:** Since $\mathbf{H}$ does not depend on $\mathbf{x}$:

1. The matrix $-\mathbf{H}$ and its factorization are computed **once** (cost: $O(n_s^3)$).
2. The Newton step $\boldsymbol{\delta} = (-\mathbf{H})^{-1}\nabla \log g(\mathbf{x}^{(t)})$ at each iteration uses a pre-computed factorization (cost: $O(n_s^2)$ per iteration).
3. The log-determinant $\log|-\mathbf{H}|$ for formula (34) is also computed once.
4. When evaluating the integral at $M$ different conditional means (Section 12), the same $-\mathbf{H}$, $(-\mathbf{H})^{-1}$, and $\log|-\mathbf{H}|$ are reused for all $M$ evaluations.

---

## 11. Finite-Difference Fallback for General PDFs

When a soft PDF $f_i$ does not have closed-form derivatives (e.g., uniform, lognormal, triangular, histogram, mixture, callable), the derivatives in (21) and (25) must be approximated numerically.

### Central difference for the first derivative

The standard central-difference formula for the first derivative is:

$$
\frac{d}{dx_i}\log f_i(x_i) \approx \frac{\log f_i(x_i + \epsilon_i) - \log f_i(x_i - \epsilon_i)}{2\epsilon_i} \tag{48}
$$

*Derivation:* Taylor-expand $\log f_i(x_i + \epsilon_i)$ and $\log f_i(x_i - \epsilon_i)$ about $x_i$:

$$
\log f_i(x_i + \epsilon_i) = \log f_i(x_i) + \epsilon_i \frac{d}{dx_i}\log f_i + \frac{\epsilon_i^2}{2}\frac{d^2}{dx_i^2}\log f_i + \frac{\epsilon_i^3}{6}\frac{d^3}{dx_i^3}\log f_i + O(\epsilon_i^4)
$$

$$
\log f_i(x_i - \epsilon_i) = \log f_i(x_i) - \epsilon_i \frac{d}{dx_i}\log f_i + \frac{\epsilon_i^2}{2}\frac{d^2}{dx_i^2}\log f_i - \frac{\epsilon_i^3}{6}\frac{d^3}{dx_i^3}\log f_i + O(\epsilon_i^4)
$$

Subtracting:

$$
\log f_i(x_i + \epsilon_i) - \log f_i(x_i - \epsilon_i) = 2\epsilon_i\frac{d}{dx_i}\log f_i + \frac{2\epsilon_i^3}{6}\frac{d^3}{dx_i^3}\log f_i + O(\epsilon_i^5)
$$

Dividing by $2\epsilon_i$:

$$
\frac{\log f_i(x_i + \epsilon_i) - \log f_i(x_i - \epsilon_i)}{2\epsilon_i} = \frac{d}{dx_i}\log f_i + \frac{\epsilon_i^2}{6}\frac{d^3}{dx_i^3}\log f_i + O(\epsilon_i^4)
$$

The error is $O(\epsilon_i^2)$ — second-order accurate.

### Central difference for the second derivative

$$
\frac{d^2}{dx_i^2}\log f_i(x_i) \approx \frac{\log f_i(x_i + \epsilon_i) - 2\log f_i(x_i) + \log f_i(x_i - \epsilon_i)}{\epsilon_i^2} \tag{49}
$$

*Derivation:* Adding the two Taylor expansions above:

$$
\log f_i(x_i + \epsilon_i) + \log f_i(x_i - \epsilon_i) = 2\log f_i(x_i) + \epsilon_i^2 \frac{d^2}{dx_i^2}\log f_i + \frac{2\epsilon_i^4}{24}\frac{d^4}{dx_i^4}\log f_i + O(\epsilon_i^6)
$$

Rearranging:

$$
\log f_i(x_i + \epsilon_i) - 2\log f_i(x_i) + \log f_i(x_i - \epsilon_i) = \epsilon_i^2 \frac{d^2}{dx_i^2}\log f_i + O(\epsilon_i^4)
$$

Dividing by $\epsilon_i^2$:

$$
\frac{\log f_i(x_i + \epsilon_i) - 2\log f_i(x_i) + \log f_i(x_i - \epsilon_i)}{\epsilon_i^2} = \frac{d^2}{dx_i^2}\log f_i + O(\epsilon_i^2)
$$

Again second-order accurate.

### Step-size selection

Since `SoftPDF` objects use piecewise-linear interpolation on a grid with $K_i$ breakpoints spanning $[z_{\min}^{(i)}, z_{\max}^{(i)}]$, the grid spacing is:

$$
\Delta z_i = \frac{z_{\max}^{(i)} - z_{\min}^{(i)}}{K_i - 1}
$$

A piecewise-linear function has zero second derivative within each segment. To capture the **global** curvature (how the slope changes across segments), $\epsilon_i$ must span several grid cells. The implementation uses:

$$
\epsilon_i = \operatorname{clip}(3\Delta z_i, \; 0.05, \; 2.0) \tag{50}
$$

The lower bound $0.05$ prevents numerical cancellation for very fine grids; the upper bound $2.0$ prevents the finite-difference stencil from extending far beyond the region of interest.

---

## 12. Vectorized Batch Laplace (Constant-Hessian Case)

When all soft PDFs satisfy the constant-Hessian condition (Section 10.3), the batch evaluation over $M$ conditional means can be fully vectorized. This section derives the batch formulas.

### Setup

We need to evaluate $I_j$ for $j = 1, \ldots, M$, where each $I_j$ uses a different conditional mean $\boldsymbol{\mu}_j$ but the same covariance $\boldsymbol{\Sigma}$ and the same soft PDFs. From (34):

$$
\log I_j \approx \log g_j(\mathbf{x}_j^*) + \frac{1}{2}\bigl(\log|\mathbf{Q}| - \log|-\mathbf{H}|\bigr) \tag{51}
$$

The second term is **the same for all $j$** (since $\mathbf{Q}$ and $\mathbf{H}$ do not depend on $j$). Define:

$$
c_{\text{det}} \stackrel{\text{def}}{=} \frac{1}{2}\bigl(\log|\mathbf{Q}| - \log|-\mathbf{H}|\bigr) \tag{52}
$$

Computed once. Now we only need $\log g_j(\mathbf{x}_j^*)$ for each $j$, which requires finding $\mathbf{x}_j^*$ for each conditional mean $\boldsymbol{\mu}_j$.

### Pre-computation

From (47), the constant negative Hessian is $\mathbf{A} = -\mathbf{H} = \mathbf{Q} + \operatorname{diag}(\boldsymbol{\sigma}^{-2})$.

Pre-compute:

1. $\mathbf{A}^{-1}$ — the Newton step matrix (cost $O(n_s^3)$, done once).
2. $c_{\text{det}} = \frac{1}{2}(\log|\mathbf{Q}| - \log|\mathbf{A}|)$ via `slogdet`.

### Vectorized gradient

Stack all $M$ current iterates into an $(M \times n_s)$ matrix $\mathbf{X}$, and all $M$ conditional means into $\boldsymbol{\mu}_{\text{grid}} \in \mathbb{R}^{M \times n_s}$.

The gradient for the $j$-th evaluation at iterate $\mathbf{x}$ (from equation 21) is:

$$
\nabla \log g_j(\mathbf{x}) = -\mathbf{Q}(\mathbf{x} - \boldsymbol{\mu}_j) + \mathbf{s}'(\mathbf{x})
$$

For the Gaussian/truncated-normal case, from (38)/(44):

$$
s'_i(x_i) = -\frac{x_i - \mu_i^s}{\sigma_i^2}
$$

So the full gradient for row $j$, stacking the components:

$$
[\nabla \log g_j(\mathbf{x}_j)]_i = -\sum_k Q_{ik}(x_{j,k} - \mu_{j,k}) - \frac{x_{j,i} - \mu_i^s}{\sigma_i^2}
$$

In matrix form for all $M$ rows simultaneously:

$$
\mathbf{G} = -(\mathbf{X} - \boldsymbol{\mu}_{\text{grid}})\mathbf{Q}^{\!\top} - (\mathbf{X} - \mathbf{1}_M \otimes \boldsymbol{\mu}^{s\top}) \odot (\mathbf{1}_M \otimes \boldsymbol{\sigma}^{-2\top})
$$

Since $\mathbf{Q}$ is symmetric ($\mathbf{Q}^\top = \mathbf{Q}$), this simplifies to:

$$
\mathbf{G} = -(\mathbf{X} - \boldsymbol{\mu}_{\text{grid}})\mathbf{Q} - (\mathbf{X} - \boldsymbol{\mu}^s) \odot \boldsymbol{\sigma}^{-2} \qquad \in \mathbb{R}^{M \times n_s} \tag{53}
$$

where $\boldsymbol{\mu}^s = (\mu_1^s, \ldots, \mu_{n_s}^s)$ and $\boldsymbol{\sigma}^{-2} = (1/\sigma_1^2, \ldots, 1/\sigma_{n_s}^2)$ are broadcast across the $M$ rows, and $\odot$ denotes elementwise multiplication.

### Vectorized Newton step

From (28), the Newton step for each row is $\boldsymbol{\delta}_j = \mathbf{A}^{-1}\mathbf{g}_j$, where $\mathbf{g}_j$ is the $j$-th row of $\mathbf{G}$.

For all $M$ rows simultaneously:

$$
\boldsymbol{\Delta} = \mathbf{G} \, \mathbf{A}^{-1} \qquad \in \mathbb{R}^{M \times n_s} \tag{54}
$$

*Justification:* Each row of $\mathbf{G}$ is a $1 \times n_s$ vector; right-multiplying by $\mathbf{A}^{-1}$ (an $n_s \times n_s$ matrix) yields a $1 \times n_s$ step. This is a single BLAS `gemm` (matrix-matrix multiply) call, far more efficient than $M$ separate linear solves.

> *Note on transpose convention:* We treat rows as $1 \times n_s$ vectors (right-multiplied), which is mathematically equivalent to the column-vector convention $\boldsymbol{\delta}_j = \mathbf{A}^{-1}\mathbf{g}_j^\top$ since $\mathbf{A}$ is symmetric ($\mathbf{A}^{-1} = (\mathbf{A}^{-1})^\top$).

### Update with clamping

$$
\mathbf{X} \leftarrow \operatorname{clip}(\mathbf{X} + \boldsymbol{\Delta}, \; \mathbf{a}, \; \mathbf{b}) \tag{55}
$$

where $\mathbf{a} = (a_1, \ldots, a_{n_s})$ and $\mathbf{b} = (b_1, \ldots, b_{n_s})$ are the support bounds, broadcast across all $M$ rows.

**Convergence criterion:** $\max_{j,i} |X_{j,i}^{(t+1)} - X_{j,i}^{(t)}| < 10^{-6}$, with at most 20 iterations.

### Vectorized log-target evaluation

After convergence, we have the mode matrix $\mathbf{X}^* \in \mathbb{R}^{M \times n_s}$ (each row is $\mathbf{x}_j^*$). We need $\log g_j(\mathbf{x}_j^*)$ from (6):

$$
\log g_j(\mathbf{x}_j^*) = -\frac{1}{2}(\mathbf{x}_j^* - \boldsymbol{\mu}_j)^\top \mathbf{Q}(\mathbf{x}_j^* - \boldsymbol{\mu}_j) + \sum_{i=1}^{n_s} \log f_i(x_{j,i}^*)
$$

**Gaussian quadratic part** for all $M$ rows:

Let $\mathbf{D} = \mathbf{X}^* - \boldsymbol{\mu}_{\text{grid}} \in \mathbb{R}^{M \times n_s}$.

$$
\text{gauss}_j = -\frac{1}{2}(\mathbf{d}_j)^\top \mathbf{Q} \mathbf{d}_j = -\frac{1}{2}\sum_{k,l} D_{jk}\,Q_{kl}\,D_{jl}
$$

In matrix form: compute $\mathbf{D}\mathbf{Q} \in \mathbb{R}^{M \times n_s}$ (matrix multiply), then take the row-wise dot product with $\mathbf{D}$:

$$
\text{gauss}_j = -\frac{1}{2} \sum_{i=1}^{n_s} [\mathbf{D}\mathbf{Q}]_{j,i} \cdot D_{j,i} \tag{56}
$$

In NumPy: `gauss = -0.5 * np.sum(D @ Q * D, axis=1)`.

**Soft-data part** for all $M$ rows (using (43)):

$$
\text{soft}_j = \sum_{i=1}^{n_s}\left[-\frac{1}{2}\left(\frac{X^*_{j,i} - \mu_i^s}{\sigma_i}\right)^2 - \log(\sigma_i\sqrt{2\pi}) - \log Z_i\right] \tag{57}
$$

Let $\mathbf{Z}_{\text{sc}} = (\mathbf{X}^* - \boldsymbol{\mu}^s) \odot \boldsymbol{\sigma}^{-1} \in \mathbb{R}^{M \times n_s}$.

$$
\text{soft}_j = \sum_{i=1}^{n_s}\left[-\frac{1}{2}Z_{\text{sc},j,i}^2 - C_i\right]
$$

In NumPy: `soft = np.sum(-0.5 * Z_sc**2 - log_norm, axis=1)` where `log_norm` contains the constants $C_i$.

**Assembled log-target:**

$$
\log g_j = \text{gauss}_j + \text{soft}_j \tag{58}
$$

### Final batch result

From (51) and (52):

$$
\log I_j = \log g_j + c_{\text{det}} \tag{59}
$$

$$
I_j = \max\!\bigl(\exp(\log I_j),\; 10^{-300}\bigr) \tag{60}
$$

All $M$ values are computed simultaneously with no Python-level loop.

**Cost summary:**

| Operation | Cost | How many times |
|-----------|------|----------------|
| $\mathbf{A}^{-1}$, $\log|\mathbf{A}|$, $\log|\mathbf{Q}|$ | $O(n_s^3)$ | Once |
| Newton gradient (53) | $O(M \cdot n_s^2)$ | Per iteration |
| Newton step (54) | $O(M \cdot n_s^2)$ | Per iteration |
| Log-target (56)-(58) | $O(M \cdot n_s^2)$ | Once (after convergence) |

Total: $O(n_s^3 + T \cdot M \cdot n_s^2)$ where $T$ is the number of Newton iterations (typically 5–15).

---

## 13. Connection to BME Prediction: The Conditioning Chain

This section derives how the Laplace integral connects to the BME prediction at an unobserved location.

### Joint distribution setup

Let $z_k$ be the unknown value at the prediction location $\mathbf{s}_k$, $\mathbf{z}_h = (z_1, \ldots, z_{n_h})^\top$ be the hard data, and $\mathbf{x} = (x_1, \ldots, x_{n_s})^\top$ be the unknown soft-data values. Under second-order stationarity, the joint distribution is:

$$
\begin{pmatrix} z_k \\ \mathbf{z}_h \\ \mathbf{x} \end{pmatrix} \sim \mathcal{N}\!\left(
\begin{pmatrix} m_k \\ \mathbf{m}_h \\ \mathbf{m}_s \end{pmatrix},
\begin{pmatrix} C_{kk} & \mathbf{C}_{kh} & \mathbf{C}_{ks} \\ \mathbf{C}_{hk} & \mathbf{C}_{hh} & \mathbf{C}_{hs} \\ \mathbf{C}_{sk} & \mathbf{C}_{sh} & \mathbf{C}_{ss} \end{pmatrix}
\right) \tag{61}
$$

where all entries are determined by the covariance model evaluated at the relevant separation distances.

### Step 1: Condition on hard data

Standard Gaussian conditioning rules give $z_k, \mathbf{x} \mid \mathbf{z}_h \sim \mathcal{N}$ with:

**For $z_k \mid \mathbf{z}_h$:**

$$
\mu_k = m_k + \mathbf{C}_{kh}\mathbf{C}_{hh}^{-1}(\mathbf{z}_h - \mathbf{m}_h) \tag{62}
$$

$$
\sigma_k^2 = C_{kk} - \mathbf{C}_{kh}\mathbf{C}_{hh}^{-1}\mathbf{C}_{hk} \tag{63}
$$

These are the kriging mean and variance.

**For $\mathbf{x} \mid \mathbf{z}_h$ (no dependence on $z_k$ yet):**

$$
\boldsymbol{\mu}_{s|h} = \mathbf{m}_s + \mathbf{C}_{sh}\mathbf{C}_{hh}^{-1}(\mathbf{z}_h - \mathbf{m}_h) \tag{64}
$$

$$
\mathbf{K}_{s|h} = \mathbf{C}_{ss} - \mathbf{C}_{sh}\mathbf{C}_{hh}^{-1}\mathbf{C}_{hs} \tag{65}
$$

### Step 2: BME posterior via Bayes' theorem

The BME posterior of $z_k$ given both hard data and soft data is:

$$
p(z_k \mid \mathbf{z}_h, \text{soft}) \propto p(z_k \mid \mathbf{z}_h) \cdot p(\text{soft} \mid z_k, \mathbf{z}_h) \tag{66}
$$

The likelihood of the soft data is the integral over the unknown true values $\mathbf{x}$:

$$
p(\text{soft} \mid z_k, \mathbf{z}_h) = \int \left(\prod_{i=1}^{n_s} f_i(x_i)\right) p(\mathbf{x} \mid z_k, \mathbf{z}_h) \, d\mathbf{x} \tag{67}
$$

This is the BME integral (1), but now conditioned on both $z_k$ and $\mathbf{z}_h$.

### Step 3: Conditional distribution $\mathbf{x} \mid z_k, \mathbf{z}_h$

Stack $(z_k, \mathbf{z}_h)$ into a vector $\mathbf{w} \in \mathbb{R}^{1+n_h}$. The joint covariance block between $\mathbf{x}$ and $\mathbf{w}$ is:

$$
\mathbf{C}_{s,kh} = \begin{pmatrix} \mathbf{C}_{sk} & \mathbf{C}_{sh} \end{pmatrix} \in \mathbb{R}^{n_s \times (1+n_h)}
$$

and the covariance of $\mathbf{w}$ is:

$$
\mathbf{C}_{kh} = \begin{pmatrix} C_{kk} & \mathbf{C}_{kh} \\ \mathbf{C}_{hk} & \mathbf{C}_{hh} \end{pmatrix} \in \mathbb{R}^{(1+n_h) \times (1+n_h)}
$$

By Gaussian conditioning:

$$
\boldsymbol{\mu}_{s|kh}(z_k) = \mathbf{m}_s + \mathbf{C}_{s,kh}\mathbf{C}_{kh}^{-1}\begin{pmatrix} z_k - m_k \\ \mathbf{z}_h - \mathbf{m}_h \end{pmatrix} \tag{68}
$$

$$
\mathbf{K}_{s|kh} = \mathbf{C}_{ss} - \mathbf{C}_{s,kh}\mathbf{C}_{kh}^{-1}\mathbf{C}_{s,kh}^\top \tag{69}
$$

Note that $\mathbf{K}_{s|kh}$ does **not** depend on $z_k$ (standard property of Gaussian conditionals — the conditional covariance is independent of the conditioning value).

### Step 4: Linear dependence of the conditional mean on $z_k$

Partition $\mathbf{C}_{s,kh}\mathbf{C}_{kh}^{-1}$ into its first column and remaining columns:

$$
\mathbf{B} = \mathbf{C}_{s,kh}\mathbf{C}_{kh}^{-1} = \begin{pmatrix} \mathbf{b}_k & \mathbf{B}_h \end{pmatrix}
$$

where $\mathbf{b}_k \in \mathbb{R}^{n_s}$ (first column) and $\mathbf{B}_h \in \mathbb{R}^{n_s \times n_h}$ (remaining columns).

Substituting into (68):

$$
\boldsymbol{\mu}_{s|kh}(z_k) = \mathbf{m}_s + \mathbf{b}_k(z_k - m_k) + \mathbf{B}_h(\mathbf{z}_h - \mathbf{m}_h)
$$

Rearranging:

$$
\boldsymbol{\mu}_{s|kh}(z_k) = \underbrace{\mathbf{b}_k}_{\text{slope}} \cdot z_k + \underbrace{\mathbf{m}_s - \mathbf{b}_k m_k + \mathbf{B}_h(\mathbf{z}_h - \mathbf{m}_h)}_{\mathbf{b}_{\text{const}}} \tag{70}
$$

This shows that the conditional mean is an **affine function** of $z_k$:

$$
\boldsymbol{\mu}_{s|kh}(z_k) = \mathbf{b}_k z_k + \mathbf{b}_{\text{const}} \tag{71}
$$

### Step 5: The BME posterior on a grid

To evaluate the posterior (66), discretize $z_k$ on a grid $\{z_k^{(j)}\}_{j=1}^M$ (typically $M = 100$ points spanning $\mu_k \pm 5\sigma_k$):

$$
p(z_k^{(j)} \mid \mathbf{z}_h, \text{soft}) \propto \underbrace{\mathcal{N}(z_k^{(j)}; \mu_k, \sigma_k^2)}_{\text{prior (kriging)}} \cdot \underbrace{I(z_k^{(j)})}_{\text{soft-data integral}} \tag{72}
$$

where:

$$
I(z_k^{(j)}) = \mathbb{E}_{\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_{s|kh}(z_k^{(j)}),\; \mathbf{K}_{s|kh})} \left[\prod_{i=1}^{n_s} f_i(x_i)\right] \tag{73}
$$

Each $I(z_k^{(j)})$ is the integral from Section 3, evaluated with conditional mean $\boldsymbol{\mu}_j = \boldsymbol{\mu}_{s|kh}(z_k^{(j)}) = \mathbf{b}_k z_k^{(j)} + \mathbf{b}_{\text{const}}$ and covariance $\mathbf{K}_{s|kh}$.

This is exactly the **batch Laplace** problem from Section 12, with:

- $M$ conditional means: $\boldsymbol{\mu}_{\text{grid}} \in \mathbb{R}^{M \times n_s}$, row $j$ is $\mathbf{b}_k z_k^{(j)} + \mathbf{b}_{\text{const}}$.
- Shared covariance: $\boldsymbol{\Sigma} = \mathbf{K}_{s|kh}$, so $\mathbf{Q} = \mathbf{K}_{s|kh}^{-1}$.

### Step 6: Normalization and statistics

The unnormalized posterior samples are:

$$
\tilde{p}_j = \mathcal{N}(z_k^{(j)}; \mu_k, \sigma_k^2) \cdot I_j
$$

Normalize by trapezoidal integration:

$$
p_j = \frac{\tilde{p}_j}{\sum_{l=1}^{M} \tilde{p}_l \cdot \Delta z_l}
$$

where $\Delta z_l$ is the grid spacing (or the trapezoidal weight). Extract statistics:

- **Posterior mean:** $\hat{z}_k = \sum_j z_k^{(j)} \, p_j \, \Delta z_j$
- **Posterior variance:** $\hat{\sigma}^2 = \sum_j (z_k^{(j)} - \hat{z}_k)^2 \, p_j \, \Delta z_j$
- **Posterior mode:** $z_k^{(\arg\max_j p_j)}$
- **Credible interval:** from the CDF $F(z_k^{(j)}) = \sum_{l: z_k^{(l)} \leq z_k^{(j)}} p_l \, \Delta z_l$

---

## 14. Validation: Exact Analytic Test Cases

### Test 1: Single Gaussian soft PDF — full derivation

**Setup:** $x \sim \mathcal{N}(\mu, \sigma^2)$ (prior), $f(x) = \mathcal{N}(x; \mu, \sigma_f^2)$ (soft PDF with the same mean for simplicity).

**Goal:** Compute $I = \mathbb{E}_{x \sim \mathcal{N}(\mu, \sigma^2)}[f(x)]$ exactly.

**Step 1:** Write out the integral.

$$
I = \int_{-\infty}^{\infty} f(x) \,\phi(x;\mu,\sigma^2) \,dx = \int_{-\infty}^{\infty} \frac{1}{\sigma_f\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma_f^2}} \cdot \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} dx
$$

**Step 2:** Combine the exponentials. Using $e^a \cdot e^b = e^{a+b}$:

$$
I = \frac{1}{2\pi\sigma\sigma_f} \int_{-\infty}^{\infty} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma_f^2} - \frac{(x-\mu)^2}{2\sigma^2}\right) dx
$$

**Step 3:** Combine the exponents. Let $u = x - \mu$:

$$
-\frac{u^2}{2\sigma_f^2} - \frac{u^2}{2\sigma^2} = -\frac{u^2}{2}\left(\frac{1}{\sigma_f^2} + \frac{1}{\sigma^2}\right) = -\frac{u^2}{2} \cdot \frac{\sigma^2 + \sigma_f^2}{\sigma^2\sigma_f^2}
$$

Define the **effective precision** $\lambda = \frac{1}{\sigma_f^2} + \frac{1}{\sigma^2} = \frac{\sigma^2 + \sigma_f^2}{\sigma^2 \sigma_f^2}$ and the **effective variance** $\tau^2 = \frac{1}{\lambda} = \frac{\sigma^2\sigma_f^2}{\sigma^2 + \sigma_f^2}$.

$$
I = \frac{1}{2\pi\sigma\sigma_f} \int_{-\infty}^{\infty} \exp\!\left(-\frac{u^2}{2\tau^2}\right) du
$$

**Step 4:** Evaluate the Gaussian integral. Using Fact 2 in one dimension ($n=1$, $\mathbf{A} = 1/\tau^2$):

$$
\int_{-\infty}^{\infty} \exp\!\left(-\frac{u^2}{2\tau^2}\right) du = \sqrt{2\pi} \cdot \tau = \sqrt{2\pi} \cdot \frac{\sigma\sigma_f}{\sqrt{\sigma^2 + \sigma_f^2}}
$$

> *Derivation of the 1-D Gaussian integral:* $\int_{-\infty}^{\infty} e^{-u^2/(2\tau^2)} du = \tau\sqrt{2\pi}$, which follows from the substitution $v = u/\tau$ giving $\int_{-\infty}^{\infty} e^{-v^2/2} \tau\,dv = \tau \cdot \sqrt{2\pi}$.

**Step 5:** Substitute back.

$$
I = \frac{1}{2\pi\sigma\sigma_f} \cdot \sqrt{2\pi} \cdot \frac{\sigma\sigma_f}{\sqrt{\sigma^2 + \sigma_f^2}} = \frac{\sqrt{2\pi} \cdot \sigma\sigma_f}{2\pi \cdot \sigma\sigma_f \cdot \sqrt{\sigma^2 + \sigma_f^2}}
$$

Cancel $\sigma\sigma_f$:

$$
I = \frac{\sqrt{2\pi}}{2\pi \cdot \sqrt{\sigma^2 + \sigma_f^2}} = \frac{1}{\sqrt{2\pi} \cdot \sqrt{\sigma^2 + \sigma_f^2}}
$$

$$
\boxed{I = \frac{1}{\sqrt{2\pi(\sigma^2 + \sigma_f^2)}}} \tag{74}
$$

### Test 2: Two independent Gaussian soft PDFs — full derivation

**Setup:** $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ with $\boldsymbol{\Sigma} = \operatorname{diag}(\sigma_1^2, \sigma_2^2)$ (diagonal, hence components are independent). Soft PDFs: $f_i(x_i) = \mathcal{N}(x_i; \mu_i, v_i)$.

**Step 1:** Write the integral.

$$
I = \int_{\mathbb{R}^2} f_1(x_1)f_2(x_2) \, \phi(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) \, dx_1\,dx_2
$$

**Step 2:** Factor the joint Gaussian. Since $\boldsymbol{\Sigma}$ is diagonal:

$$
\phi(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \phi(x_1; \mu_1, \sigma_1^2) \cdot \phi(x_2; \mu_2, \sigma_2^2)
$$

*Proof:* For diagonal $\boldsymbol{\Sigma}$, the quadratic form decomposes as $(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) = \frac{(x_1-\mu_1)^2}{\sigma_1^2} + \frac{(x_2-\mu_2)^2}{\sigma_2^2}$, and $|\boldsymbol{\Sigma}| = \sigma_1^2\sigma_2^2$, so the joint density factors into the product of marginals.

**Step 3:** Separate the double integral.

$$
I = \int_{-\infty}^{\infty} f_1(x_1)\phi(x_1;\mu_1,\sigma_1^2)\,dx_1 \cdot \int_{-\infty}^{\infty} f_2(x_2)\phi(x_2;\mu_2,\sigma_2^2)\,dx_2
$$

This separation is valid because the integrand factors into a function of $x_1$ only times a function of $x_2$ only.

**Step 4:** Apply result (74) to each factor.

$$
I = \frac{1}{\sqrt{2\pi(\sigma_1^2 + v_1)}} \cdot \frac{1}{\sqrt{2\pi(\sigma_2^2 + v_2)}} \tag{75}
$$

### Test 3: Laplace is exact for Gaussian soft PDFs — proof

**Claim:** When every $f_i$ is Gaussian, the Laplace approximation is exact (not an approximation).

**Proof.** From (6):

$$
\log g(\mathbf{x}) = -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\mathbf{Q}(\mathbf{x}-\boldsymbol{\mu}) + \sum_{i=1}^{n_s} \log f_i(x_i)
$$

When each $f_i$ is Gaussian with parameters $(\mu_i^s, \sigma_i^2)$, from (36):

$$
\log f_i(x_i) = -\frac{(x_i - \mu_i^s)^2}{2\sigma_i^2} - \log(\sigma_i\sqrt{2\pi})
$$

This is a **quadratic polynomial** in $x_i$ (the $x_i$-dependent part is $-\frac{x_i^2 - 2x_i\mu_i^s + (\mu_i^s)^2}{2\sigma_i^2}$) plus a constant.

Therefore $\sum_i \log f_i(x_i) = -\frac{1}{2}\mathbf{x}^\top \operatorname{diag}(\boldsymbol{\sigma}^{-2})\mathbf{x} + \mathbf{x}^\top \operatorname{diag}(\boldsymbol{\sigma}^{-2})\boldsymbol{\mu}^s + \text{const}$, which is a quadratic polynomial in $\mathbf{x}$.

Adding the first term $-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\mathbf{Q}(\mathbf{x}-\boldsymbol{\mu})$, which is also quadratic:

$$
\log g(\mathbf{x}) = \text{(quadratic in } \mathbf{x}\text{)} + \text{(quadratic in } \mathbf{x}\text{)} = \text{quadratic in } \mathbf{x}
$$

A quadratic function has the Taylor expansion:

$$
h(\mathbf{x}) = h(\mathbf{x}^*) + \nabla h(\mathbf{x}^*)^\top(\mathbf{x}-\mathbf{x}^*) + \frac{1}{2}(\mathbf{x}-\mathbf{x}^*)^\top \nabla^2 h(\mathbf{x}^*)(\mathbf{x}-\mathbf{x}^*) + 0
$$

with **no remainder** (all derivatives of order $\geq 3$ vanish identically for a quadratic function). Therefore the second-order Taylor expansion used in the Laplace approximation (equation 11) is **exact**, not an approximation.

Consequently, the Laplace approximation reproduces the exact value (74)/(75) for Gaussian soft PDFs. This serves as a strong unit test. $\square$

---

## 15. Error Analysis and Approximation Quality

### Source of Laplace approximation error

The Laplace approximation drops the $O(\|\mathbf{x} - \mathbf{x}^*\|^3)$ remainder in the Taylor expansion (8). The leading error term involves the **third derivatives** of $\log g$:

$$
\text{Error} \sim \sum_{j,k,l} \frac{\partial^3 \log g}{\partial x_j \partial x_k \partial x_l}\bigg|_{\mathbf{x}^*} \times (\text{moments of the Gaussian approximation})
$$

The third derivative of the Gaussian part $-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\mathbf{Q}(\mathbf{x}-\boldsymbol{\mu})$ is **exactly zero** (it's quadratic). Therefore the leading error comes entirely from the **third derivative of $\log f_i$**:

$$
\frac{d^3}{dx_i^3}\log f_i(x_i)\bigg|_{x_i = x_i^*}
$$

### When Laplace is accurate

- **Gaussian soft PDFs:** $\frac{d^3}{dx_i^3}\log f_i = 0$ (all third and higher derivatives vanish). Laplace is **exact**.
- **Truncated-normal soft PDFs:** Within the unclamped region, $\log f_i$ is quadratic with zero third derivative. Near the truncation boundaries, the clamped support introduces effective higher-order terms, but these are small when the truncation bounds are several $\sigma$ from the mode.
- **Near-Gaussian soft PDFs:** Any soft PDF that is approximately log-quadratic near its mode will have small third derivatives, and Laplace will be accurate.

### When Laplace is inaccurate

- **Uniform / interval soft PDFs:** $f_i$ is a box function. $\log f_i$ is constant on the interior and $-\infty$ outside, creating a non-smooth boundary. The Hessian is zero on the interior, making the Gaussian approximation degenerate.
- **Bimodal / mixture soft PDFs:** $\log g$ may have multiple local maxima. The Laplace approximation captures only the neighborhood of one mode.
- **Highly skewed soft PDFs** (e.g., lognormal): Significant third-derivative terms lead to poor Gaussian approximation.

In these cases, the implementation falls back to **Expectation Propagation** (Section 16.1) or **Laplace Importance Sampling** (Section 16.2).

---

## 16. Related Methods: EP and Laplace Importance Sampling

### 16.1 Expectation Propagation (EP)

EP iteratively approximates each non-Gaussian factor $f_i(x_i)$ with a Gaussian **site**:

$$
\tilde{t}_i(x_i) = s_i \exp\!\left(\tau_i x_i - \frac{1}{2}\lambda_i x_i^2\right) \tag{76}
$$

where $s_i > 0$ is a scale, $\tau_i$ is a natural location parameter, and $\lambda_i > 0$ is a precision parameter.

#### The EP iteration

**Step 1: Compute the posterior approximation $q(\mathbf{x})$.**

With all sites included, the approximate posterior is a Gaussian:

$$
q(\mathbf{x}) \propto \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) \cdot \prod_{i=1}^{n_s} \tilde{t}_i(x_i)
$$

Since the prior is $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ with natural parameters $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$, $\boldsymbol{\eta} = \boldsymbol{\Lambda}\boldsymbol{\mu}$, and each site contributes $\lambda_i$ to precision and $\tau_i$ to the natural mean:

$$
\boldsymbol{\Lambda}_q = \boldsymbol{\Lambda} + \operatorname{diag}(\boldsymbol{\lambda}), \qquad \boldsymbol{\eta}_q = \boldsymbol{\eta} + \boldsymbol{\tau}
$$

$$
\boldsymbol{\Sigma}_q = \boldsymbol{\Lambda}_q^{-1}, \qquad \boldsymbol{\mu}_q = \boldsymbol{\Sigma}_q \boldsymbol{\eta}_q
$$

**Step 2: Form the cavity distribution for site $i$.**

Remove site $i$ from $q$ to get the "cavity" distribution $q_{-i}(x_i)$. Working in natural-parameter form (as in the implementation):

$$
\text{Cavity precision: } \lambda_{-i} = \frac{1}{[\boldsymbol{\Sigma}_q]_{ii}} - \lambda_i
$$

$$
\text{Cavity natural mean: } \nu_{-i} = \frac{[\boldsymbol{\mu}_q]_i}{[\boldsymbol{\Sigma}_q]_{ii}} - \nu_i
$$

$$
\text{Cavity mean: } \mu_{-i} = \frac{\nu_{-i}}{\lambda_{-i}}
$$

*Derivation:* The marginal of $q$ at site $i$ has precision $1/[\boldsymbol{\Sigma}_q]_{ii}$ and natural mean $[\boldsymbol{\mu}_q]_i / [\boldsymbol{\Sigma}_q]_{ii}$. Since the site-$i$ approximation contributes $(\lambda_i, \nu_i)$ to these natural parameters, removing it gives the cavity natural parameters above.

In summary: $q_{-i}(x_i) = \mathcal{N}(x_i; \mu_{-i}, \sigma_{-i}^2)$ where $\sigma_{-i}^2 = 1/\lambda_{-i}$.

**Step 3: Moment matching.**

Compute the moments of the **tilted distribution** $f_i(x_i) \cdot q_{-i}(x_i)$:

$$
Z_i = \int f_i(x_i) \, \mathcal{N}(x_i; \mu_{-i}, \sigma_{-i}^2) \, dx_i
$$

$$
\hat{m}_i = \frac{1}{Z_i}\int x_i \, f_i(x_i) \, \mathcal{N}(x_i; \mu_{-i}, \sigma_{-i}^2) \, dx_i
$$

$$
\hat{v}_i = \frac{1}{Z_i}\int (x_i - \hat{m}_i)^2 \, f_i(x_i) \, \mathcal{N}(x_i; \mu_{-i}, \sigma_{-i}^2) \, dx_i
$$

These 1-D integrals are evaluated by trapezoidal quadrature on the soft-PDF grid.

**Step 4: Update site parameters.**

$$
\lambda_i^{\text{new}} = \frac{1}{\hat{v}_i} - \lambda_{-i}, \qquad \tau_i^{\text{new}} = \frac{\hat{m}_i}{\hat{v}_i} - \frac{\mu_{-i}}{\sigma_{-i}^2}
$$

With damping: $\lambda_i \leftarrow \gamma \lambda_i^{\text{new}} + (1-\gamma)\lambda_i$ (damping factor $\gamma \in (0, 1]$).

**Step 5:** Repeat steps 1–4 for all $i$ until convergence.

#### EP marginal likelihood

$$
\log Z_{\text{EP}} = \sum_{i=1}^{n_s} \log \hat{s}_i + \frac{1}{2}\bigl(\log|\boldsymbol{\Lambda}| - \log|\boldsymbol{\Lambda}_q|\bigr) + \frac{1}{2}\bigl(\boldsymbol{\eta}_q^\top\boldsymbol{\Sigma}_q\boldsymbol{\eta}_q - \boldsymbol{\eta}^\top\boldsymbol{\Sigma}\boldsymbol{\eta}\bigr) \tag{77}
$$

where $\hat{s}_i$ are the site scales from the final sweep.

### 16.2 Laplace Importance Sampling (LIS)

LIS provides an **unbiased** correction to the Laplace approximation by using the Laplace mode and Hessian to construct an importance-sampling proposal.

**Step 1: Compute the Laplace mode and Hessian** (Sections 8 and 7):

$$
\mathbf{x}^* = \arg\max_{\mathbf{x}} \log g(\mathbf{x}), \qquad \mathbf{H} = \nabla^2 \log g(\mathbf{x}^*)
$$

**Step 2: Define the proposal distribution:**

$$
q(\mathbf{x}) = \mathcal{N}(\mathbf{x};\; \mathbf{x}^*,\; (-\mathbf{H})^{-1}) \tag{78}
$$

This is the multivariate Gaussian centered at the Laplace mode with covariance equal to the inverse of the negative Hessian — exactly the Gaussian that the Laplace approximation implicitly fits.

**Step 3: Draw $N$ samples** $\mathbf{x}_1, \ldots, \mathbf{x}_N \sim q$.

**Step 4: Compute importance weights** in log-space.

The BME integral from (7) is:

$$
I = \frac{|\mathbf{Q}|^{1/2}}{(2\pi)^{n_s/2}} \int g(\mathbf{x})\,d\mathbf{x}
$$

By importance sampling with proposal $q$:

$$
\int g(\mathbf{x})\,d\mathbf{x} = \int \frac{g(\mathbf{x})}{q(\mathbf{x})} q(\mathbf{x})\,d\mathbf{x} = \mathbb{E}_{q}\!\left[\frac{g(\mathbf{x})}{q(\mathbf{x})}\right] \approx \frac{1}{N}\sum_{j=1}^N \frac{g(\mathbf{x}_j)}{q(\mathbf{x}_j)}
$$

So $I \approx \frac{|\mathbf{Q}|^{1/2}}{(2\pi)^{n_s/2}} \cdot \frac{1}{N}\sum_{j=1}^N \frac{g(\mathbf{x}_j)}{q(\mathbf{x}_j)}$.

We now derive the log importance weight carefully, tracking all normalizing constants.

Recall that $g(\mathbf{x})$ from (6) is **unnormalized** — it does not include the $(2\pi)^{-n_s/2}|\mathbf{Q}|^{1/2}$ prior normalizer. Meanwhile $q(\mathbf{x})$ is a fully normalized Gaussian density. Define the **unnormalized** log-weight:

$$
\log \tilde{w}_j = \log g(\mathbf{x}_j) - \log q(\mathbf{x}_j) \tag{79}
$$

Expanding $\log g$ from (6):

$$
\log g(\mathbf{x}_j) = -\frac{1}{2}(\mathbf{x}_j - \boldsymbol{\mu})^\top\mathbf{Q}(\mathbf{x}_j - \boldsymbol{\mu}) + \sum_i \log f_i(x_{j,i})
$$

Expanding $\log q$ from (78) (the full normalized Gaussian):

$$
\log q(\mathbf{x}_j) = -\frac{1}{2}(\mathbf{x}_j - \mathbf{x}^*)^\top(-\mathbf{H})(\mathbf{x}_j - \mathbf{x}^*) + \frac{1}{2}\log|{-\mathbf{H}}| - \frac{n_s}{2}\log(2\pi)
$$

Substituting into (79), the unnormalized log-weight is:

$$
\log \tilde{w}_j = \underbrace{\log g(\mathbf{x}_j)}_{\text{log-target}} - \underbrace{\left[-\frac{1}{2}(\mathbf{x}_j-\mathbf{x}^*)^\top(-\mathbf{H})(\mathbf{x}_j-\mathbf{x}^*)\right]}_{\text{quadratic kernel of } q} - \frac{1}{2}\log|-\mathbf{H}| + \frac{n_s}{2}\log(2\pi)
$$

To obtain $I$, we must also incorporate the prior normalizer $\frac{|\mathbf{Q}|^{1/2}}{(2\pi)^{n_s/2}}$ that multiplies $\int g\,d\mathbf{x}$. This gives a correction term:

$$
\log\!\left(\frac{|\mathbf{Q}|^{1/2}}{(2\pi)^{n_s/2}}\right) = \frac{1}{2}\log|\mathbf{Q}| - \frac{n_s}{2}\log(2\pi)
$$

The $(2\pi)$ terms from this correction and from $\log q$ cancel. Combining everything, the **fully normalized** log importance weight used in the implementation is:

$$
\log w_j = \log g(\mathbf{x}_j) - \left[-\frac{1}{2}(\mathbf{x}_j-\mathbf{x}^*)^\top(-\mathbf{H})(\mathbf{x}_j-\mathbf{x}^*)\right] + \frac{1}{2}\bigl(\log|\boldsymbol{\Sigma}_{\text{prop}}| - \log|\boldsymbol{\Sigma}|\bigr) \tag{80}
$$

where $\boldsymbol{\Sigma} = \mathbf{Q}^{-1}$ (prior covariance) and $\boldsymbol{\Sigma}_{\text{prop}} = (-\mathbf{H})^{-1}$ (proposal covariance). This matches the code, which first computes `log_w[j] = lt - lq` (the quadratic-kernel ratio) and then adds the determinant correction `log_w += 0.5 * (logdet_prop - logdet_cov)`.

**Step 5: Estimate $I$** using the log-sum-exp trick for numerical stability:

$$
I \approx \exp\!\left(\max_j \log w_j\right) \cdot \frac{1}{N}\sum_{j=1}^N \exp(\log w_j - \max_j \log w_j) \tag{81}
$$

**Properties:**
- LIS is **unbiased** (unlike pure Laplace, which is biased by the Taylor truncation).
- The variance is much lower than raw Monte Carlo because the proposal is centered on the mode and shaped by the curvature.
- Cost: $O(n_s^3)$ for mode + Hessian, then $O(N \times n_s^2)$ for sampling.

---

## 17. References

1. **Tierney, L. & Kadane, J.B.** (1986). Accurate Approximations for Posterior Moments and Marginal Densities. *Journal of the American Statistical Association*, 81(393), 82–86. https://doi.org/10.1080/01621459.1986.10478240

2. **Rue, H., Martino, S. & Chopin, N.** (2009). Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations. *Journal of the Royal Statistical Society: Series B*, 71(2), 319–392. https://doi.org/10.1111/j.1467-9868.2008.00700.x

3. **Minka, T.** (2001). Expectation Propagation for approximate Bayesian inference. *Proceedings of the 17th Conference on Uncertainty in Artificial Intelligence (UAI 2001)*, 362–369.

4. **Rasmussen, C.E. & Williams, C.K.I.** (2006). *Gaussian Processes for Machine Learning*. MIT Press, Chapter 3. https://gaussianprocess.org/gpml/

5. **Christakos, G.** (2000). *Modern Spatiotemporal Geostatistics*. Oxford University Press. ISBN 0-19-513895-3. (Foundation for the BME framework.)

6. **Serre, M.L. & Christakos, G.** (1999). Modern geostatistics: computational BME analysis in the light of uncertain physical knowledge — the Equus Beds study. *Stochastic Environmental Research and Risk Assessment*, 13(1–2), 1–26. https://doi.org/10.1007/s004770050029

7. **Nocedal, J. & Wright, S.J.** (2006). *Numerical Optimization*. 2nd ed., Springer. https://doi.org/10.1007/978-0-387-40065-5 Chapter 3 (Line Search Methods), Chapter 7 (Large-Scale Unconstrained Optimization). (Reference for Newton's method, backtracking line search, Hessian modification.)

8. **Bishop, C.M.** (2006). *Pattern Recognition and Machine Learning*. Springer. ISBN 978-0-387-31073-2. Section 4.4 (Laplace Approximation). (Accessible introduction to the Laplace method in a machine learning context.)
