# HW4 — Constraint-aware receding-horizon CFG

HW4 asks a different question from HW3: once a frozen conditional generator is controllable, can guidance be selected online without retraining or adding generator evaluations?

CRH-CFG reuses the HW3 conditional/unconditional pair, forecasts candidate clean endpoints, scores target margin and protected-attribute drift, and chooses the minimum feasible intervention. The controller keeps the generator and its NFE fixed.

## Paper and poster

| Artifact | View | Source |
| --- | --- | --- |
| Paper | [Open PDF](paper/main.pdf) | [LaTeX](paper/main.tex) · [BibTeX](paper/main.bib) |
| Course poster | [Open PDF](poster/crh_cfg_poster.pdf) | [Editable PowerPoint](poster/crh_cfg_poster.pptx) |

[Method](docs/METHOD.md) · [Results](results/pareto.csv) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Limitations](docs/LIMITATIONS.md)

[![Open the HW4 paper](figures/blond_hair_paired_methods.png)](paper/main.pdf)

The final paired study uses 1,000 fixed seeds per target. Full CRH-CFG retains comparable target success with roughly 1.1–2.7% measured runtime overhead and prevents the large intervention-induced protected drift seen in weaker feedback variants. Conditional quality remains unresolved because KID uses the full CelebA marginal rather than target-conditioned references.
