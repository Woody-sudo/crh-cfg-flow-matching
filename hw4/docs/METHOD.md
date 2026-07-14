# Method

## First-principles view

A straight conditional flow already contains a local estimate of where the current state will end. For the frozen model's noise-to-data convention, progress is $s\in[0,1]$ and the affine endpoint forecast is

$$
\widehat{x}_0 = x_s + (1-s)v(x_s,s).
$$

Classifier-free guidance spans an affine family of fields without additional generator calls:

$$
v_w=v_u+w(v_c-v_u).
$$

Combining the two relations turns each available guidance scale into an explicit candidate endpoint. This makes guidance selection a small constrained control problem rather than a fixed global hyperparameter.

## Receding-horizon decision

At predefined controller steps:

1. Evaluate the frozen conditional and unconditional velocity fields once.
2. Vectorize the candidate field bank over scales $w\in\mathcal W$.
3. Forecast every candidate endpoint analytically.
4. Score candidate endpoints with a frozen attribute evaluator.
5. Form a feasible set from the target-margin and protected-drift constraints.
6. Select the feasible scale with minimum intervention and temporal movement.
7. Hold that scale across the current Heun pair; update again at the next controller step.

The evaluator is outside the generator NFE budget. Candidate construction reuses the two fields already required by ordinary CFG, so CRH-CFG changes control logic but not generator forward count.

## Reliability gate and fallback

Early endpoint forecasts can be unreliable. The released full configuration therefore keeps a default scale during an initial gate interval. If the constrained feasible set is empty, the controller minimizes a deterministic penalty combining target shortfall, protected drift, intervention size, and change from the previous scale. This ensures every sample receives a valid decision while exposing fallback frequency as an observable diagnostic.

## Time convention

The course prompt uses a data-to-noise time $t$, whereas the frozen generator uses noise-to-data progress $s=1-t$. The equivalent endpoint equations are

$$
\widehat{x}_0=x_t-t u_t
\quad\Longleftrightarrow\quad
\widehat{x}_0=x_s+(1-s)v_s,
$$

with $u_t=-v_s$. Unit tests cover both forms and the endpoint semantics $w=0$ (unconditional) and $w=1$ (conditional).

