# Algorithm: Scalable Maximization of Spectral Radius with Phase Variables

## Problem
Maximize
$$
\max_{x_1,\dots,x_m}\;
\rho\!\left(\sum_{k=1}^m D_k e^{-j x_k}\right),
$$
where $D_k \in \mathbb{C}^{n \times n}$ are given matrices and $x_k \in \mathbb{R}$ are decision variables.

---

## Key Idea
Eliminate the phase variables $x_k$ analytically and optimize over a single unit-norm vector $v$.

Using the variational characterization of the spectral radius:
$$
\max_{x} \rho\!\left(\sum_{k=1}^m D_k e^{-j x_k}\right)
=
\max_{\|v\|=1} \sum_{k=1}^m |v^* D_k v|.
$$

For a fixed $v$, the optimal phases are:
$$
x_k^\star = \arg\!\left(v^* D_k v\right).
$$

---

## Optimization Problem
Solve:
$$
\max_{\|v\|=1} \; g(v),
\quad
g(v) := \sum_{k=1}^m |v^* D_k v|.
$$

This is a smooth (almost everywhere), nonconvex optimization on the unit sphere.

---

## Algorithm Outline

### Step 1: Generate Initial Points
Construct a small set of initial vectors $\{v_0^{(i)}\}$:
- Dominant eigenvectors of selected matrices $D_k$
- Dominant eigenvector of $\sum_k |D_k|$
- One or more random unit vectors

---

### Step 2: Riemannian Gradient Ascent
For each initialization $v_0$:

1. **Objective**
   $$
   g(v) = \sum_{k=1}^m |v^* D_k v|
   $$

2. **Euclidean Gradient**
   $$
   \nabla g(v)
   =
   2 \sum_{k=1}^m
   \mathrm{Re}\!\left(
   \frac{v^* D_k v}{|v^* D_k v|}
   \right)
   D_k v
   \quad
   (\text{when } v^* D_k v \neq 0)
   $$

3. **Projection to Tangent Space**
   $$
   \nabla_{\mathcal{S}} g
   =
   \nabla g - (v^* \nabla g)\, v
   $$

4. **Update**
   $$
   v \leftarrow \frac{v + \alpha \nabla_{\mathcal{S}} g}{\|v + \alpha \nabla_{\mathcal{S}} g\|}
   $$
   with step size $\alpha$ determined by line search or fixed schedule.

5. **Stopping Criterion**
   - Gradient norm below tolerance, or
   - Maximum iterations reached

---

### Step 3: Select Best Solution
Choose the $v^\star$ with the largest objective value $g(v)$ among all runs.

---

### Step 4: Recover Optimal Phases
Compute the optimal phases:
$$
x_k^\star = \arg\!\left(v^{\star*} D_k v^\star\right),
\quad k = 1,\dots,m.
$$

---

### Step 5 (Optional): Phase-Space Refinement
Use $\{x_k^\star\}$ as initialization for a local gradient-based optimizer on the original phase variables.

---

## Complexity
- Per iteration: $O(m n^2)$
- Independent of phase dimension $m$ in the search space
- Scales well for large $m$

---

## Remarks
- Avoids exponential sampling over phase variables
- Exploits analytical phase alignment
- Multiple initializations mitigate local maxima
- Can be combined with SDP/LMI solutions for strong warm starts

---
