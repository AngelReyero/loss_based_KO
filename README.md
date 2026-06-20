# Semi-knockoffs: a model-agnostic conditional independence testing method with finite-sample guarantees

Conditional independence testing (CIT) is essential for reliable scientific discovery, as it prevents spurious findings and enables controlled feature selection. Recent CIT methods leverage machine learning (ML) models as surrogates for the underlying distribution. However, model-agnostic approaches typically require a train–test split, which can reduce statistical power.

We introduce **Semi-knockoffs**, a CIT method that can accommodate any pre-trained model, avoids the train–test split, and provides valid *p*-values as well as false discovery rate (FDR) control in high-dimensional settings. Unlike methods that rely on the model–$X$ assumption (i.e., a known input distribution), Semi-knockoffs only requires estimates of conditional expectations for continuous variables, making the procedure less restrictive and more practical for integration with modern machine learning workflows.

To ensure validity when these expectations are estimated, we establish two new theoretical results:

1. **Stability** for regularized models trained with a null feature.
2. **Double robustness**, which retains statistical power even when one of the required components is misspecified.

All experimental figures, together with the CSV files required to reproduce them, are available in the `results` folder. The `src` folder contains the code used to reproduce all experiments.

For comparisons with Knockoffs, dCRT, and LOCO, we use the implementations provided by **HiDimStat**, while **pyHRT** is used for HRT.

## Paper

```bibtex
@inproceedings{Semi-knockoffs,
  title     = {Semi-knockoffs: a model-agnostic conditional independence testing method with finite-sample guarantees},
  author    = {Angel Reyero-Lobo and Bertrand Thirion and Pierre Neuvial},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

**Arxiv:** [https://arxiv.org/abs/2601.23124](https://arxiv.org/abs/2601.23124)

