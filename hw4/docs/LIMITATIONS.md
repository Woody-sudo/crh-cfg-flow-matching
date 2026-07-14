# Limitations

- The study covers one frozen CelebA flow-matching checkpoint, three positive attribute targets, and one frozen external evaluator.
- The evaluator can be miscalibrated or exploitable by controller feedback. It is a control signal, not ground truth.
- The common-reference preservation analysis does not show that full CRH-CFG improves preservation over ordinary conditional guidance; it shows that the full controller avoids the large intervention-induced drift of weaker feedback variants.
- KID uses the full CelebA marginal rather than target-conditioned real subsets. A lower KID row is not automatically a better conditional generator, and the mixed target-wise movement leaves conditional quality unresolved.
- Controller overhead excludes no work: evaluator forwards are recorded separately, while generator NFE remains matched. Wall-clock overhead is environment dependent.
- Candidate scales are discrete, thresholds are validation-calibrated, and the affine endpoint forecast is most reliable near the data endpoint.
- The public release omits third-party data and trained weights, so it supports immediate logic verification but not a one-command reproduction of the full image study.

