# Constraint-Aware Receding-Horizon CFG

CRH-CFG is a training-free controller for attribute-conditioned flow matching. It reuses the frozen conditional/unconditional velocity pair, analytically forecasts candidate clean endpoints, scores them with an external attribute evaluator, and selects the smallest feasible guidance intervention. The generator architecture and number of generator field evaluations remain unchanged.

> Course project by **Mu Chen** for CMU 10-799: Diffusion & Flow Matching (Spring 2026).

[Paper](paper/main.pdf) · [Poster](poster/crh_cfg_poster.pdf) · [Editable poster](poster/crh_cfg_poster.pptx) · [Compact results](results/pareto.csv)

![Paired qualitative comparison](figures/blond_hair_paired_methods.png)

## Method in one equation

For the frozen unconditional and conditional fields, CRH-CFG constructs

$$
v_w = v_u + w(v_c-v_u), \qquad \widehat{x}_0(w)=x_s+(1-s)v_w.
$$

At a controller update, candidate scales are evaluated at their forecast endpoints. The controller enforces a target-margin constraint and a protected-attribute drift budget, then chooses the feasible scale nearest ordinary conditional guidance ($w=1$) and the previous decision. If no candidate is feasible, a deterministic penalized fallback is used. See [METHOD.md](docs/METHOD.md) for the full control logic and conventions.

## Main empirical readout

The final paired evaluation uses 1,000 fixed seeds for each of three CelebA attribute targets. Relative to the inherited HW3 baseline, full CRH-CFG:

- retains comparable target success (98.7%, 98.4%, and 99.3%);
- adds no generator forward evaluations and approximately 1.1–2.7% measured runtime overhead;
- strongly reduces intervention-induced protected drift relative to unconstrained feedback variants;
- does **not** establish a conditional quality improvement: KID uses the full CelebA marginal as reference and moves differently across targets.

These are descriptive results for the frozen checkpoint and evaluator used in the course study, not a claim of universal improvement. Exact rows are in [`results/main_results.csv`](results/main_results.csv); limitations are documented in [LIMITATIONS.md](docs/LIMITATIONS.md).

## Repository map

```text
src/                 Controller, endpoint forecast, sampling, and metrics
tests/               CPU unit tests for invariants and NFE accounting
configs/             Released controller configurations
results/             Compact tables from frozen paired evaluations
figures/             Selected paired samples, schedules, and cases
paper/               CVPR-style PDF and LaTeX source
poster/              24x36 course poster (PDF and editable PPTX)
docs/                Method, reproduction scope, and limitations
```

## Quick verification

The controller tests do not require CelebA, a checkpoint, or a GPU.

```bash
git clone https://github.com/Woody-sudo/crh-cfg-flow-matching.git
cd crh-cfg-flow-matching
uv sync --dev
uv run pytest -q
```

The release intentionally omits CelebA images/annotations, trained weights, cloud logs, and evaluator checkpoints. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the boundary between immediately runnable controller checks and the full empirical rerun.

## Acknowledgments and provenance

The study was developed from the CMU 10-799 course workflow. The public repository contains the author’s HW4 controller and presentation artifacts; it does not redistribute the course starter repository. Flow Matching, classifier-free guidance, CelebA, and evaluation references are listed in the paper bibliography.

## License

Code in this release is available under the [MIT License](LICENSE). Paper, poster, result tables, and figures are provided for scholarly inspection and citation; third-party datasets, checkpoints, fonts, and templates remain subject to their original terms.

