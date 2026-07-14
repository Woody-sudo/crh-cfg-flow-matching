# HW3 — Conditional Flow Matching and CFG baseline

HW3 establishes the frozen generator that HW4 later controls. The project adds discrete CelebA attribute conditioning to a straight-path Flow Matching model, trains with field/row dropout so the same network supports unconditional and conditional predictions, and composes them with classifier-free guidance (CFG).

## Model and conditions

The generator transports Gaussian noise to 64×64 CelebA images along

$$
x_s=(1-s)\epsilon+s x_0,\qquad v^*=x_0-\epsilon.
$$

The U-Net receives three discrete condition fields:

- smiling: null / absent / present;
- bangs: null / absent / present;
- hair: null / black / blond / brown / other.

During sampling, the frozen model evaluates an unconditional and a conditional velocity and applies

$$
v_w=v_u+w(v_c-v_u).
$$

The recommended qualitative baseline is Heun with 50 steps and guidance scale $w=2$.

## What the evidence shows

An independent ResNet-18 attribute evaluator was trained separately from the generator. On 256 fixed samples per condition, increasing CFG from $w=0$ to $w=2$ changed predicted target success as follows:

| Target | $w=0$ | $w=2$ |
| --- | ---: | ---: |
| smiling present | 44.5% | 99.6% |
| bangs present | 16.8% | 96.9% |
| black hair | 30.9% | 97.7% |
| blond hair | 9.0% | 97.7% |
| smiling + no bangs + blond | 3.1% joint | 94.5% joint |

The strongest evidence is for smiling and bangs. Black/blond hair are useful but evaluator-limited; brown hair remains weaker. Full rows are in [`results/controllability.csv`](results/controllability.csv), and evaluator quality is disclosed in [`results/evaluator_metrics.csv`](results/evaluator_metrics.csv).

![HW3 conditional sample grid](figures/smiling_no_bangs_blond.png)

## Side study: solver speed

Before the conditional baseline, HW3 tested whether second-order Heun integration could accelerate the HW2 flow. The result was negative: at equal NFE, Euler remained better, while Heun-50 obtained only a small KID reduction at roughly 1.91× latency and 2× NFE. This negative result is retained because it motivated the later focus on control rather than solver order.

[Controllability report](docs/CONTROLLABILITY.md) · [Speed study](docs/SPEED_STUDY.md) · [Training configuration](configs/flow_matching_hw3_gfm.yaml)
