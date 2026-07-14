# Reproducibility scope

This repository separates lightweight controller verification from the data- and checkpoint-dependent course experiment.

## Runnable from this repository

Install the development environment and run:

```bash
uv sync --dev
uv run pytest -q
```

The CPU tests verify candidate vectorization, endpoint identities, time orientation, infeasible-set fallback, fixed-seed noise, test-split isolation, and unchanged generator forward counts.

## Full empirical rerun requirements

The reported CelebA evaluation additionally requires:

- the frozen HW3 attribute-conditioned flow-matching checkpoint;
- the frozen five-attribute evaluator checkpoint;
- CelebA obtained under its original access and usage terms;
- the inherited train/validation/test split and the released fixed test seeds;
- a CUDA-capable environment for practical runtime.

Those artifacts are not redistributed here. The released YAML files record the controller settings, while the paper documents the sampler, seed pairing, evaluation population, metrics, and claim limits. `results/main_results.csv` is the immutable full row-level aggregate used by the paper; `results/pareto.csv` is a compact view.

## Result interpretation

All methods use paired initial noise. Runtime percentages are measurements on the original execution environment and should not be treated as hardware-independent constants. KID compares generated samples with the inherited full CelebA marginal; it is therefore an unconditional realism/diversity diagnostic, not direct evidence of target-conditional fidelity.

