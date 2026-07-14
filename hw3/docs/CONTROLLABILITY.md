# HW3 controllability report

## Question

Does the conditional Flow Matching checkpoint respond consistently to semantic attributes under classifier-free guidance?

## Protocol

- Generator: conditional U-Net, 50,000 training iterations, EMA checkpoint.
- Sampler: fixed-step Heun, 50 steps (100 generator field evaluations).
- CFG scales: 0, 1, 2, and 3.
- Evaluation: 256 generated samples per condition.
- Evaluator: independently trained ImageNet-initialized ResNet-18 with binary heads for smiling, bangs, black hair, blond hair, and brown hair.
- Split: deterministic 80/10/10 split over 63,715 CelebA examples.

The evaluator is not part of generator training. Its test F1 is 0.891 for smiling, 0.840 for bangs, 0.764 for black hair, 0.760 for blond hair, and 0.639 for brown hair. Accordingly, smiling/bangs support the strongest claims; brown hair is diagnostic.

## Results

Target success rises sharply with guidance for every tested single attribute. At $w=2$, success reaches 99.6% for smiling present, 96.9% for bangs present, 97.7% for black hair, and 97.7% for blond hair. The joint smiling/no-bangs/blond condition rises from 3.1% at $w=0$ to 94.5% at $w=2$.

Scale $w=3$ often saturates the evaluator, but saturation alone does not demonstrate higher perceptual quality. The project therefore freezes $w=2$ as the HW3 operating baseline: it yields near-saturated controllability without selecting the largest tested scale.

## Claim boundary

These numbers measure agreement with one external classifier, not human preference or causal attribute disentanglement. The study does not claim that higher guidance universally improves image quality. Hair-color claims must be read together with the evaluator's lower F1, especially for brown hair.

Machine-readable outputs: [`controllability.csv`](../results/controllability.csv) and [`evaluator_metrics.csv`](../results/evaluator_metrics.csv).

