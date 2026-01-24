# Semi-knockoffs: a model-agnostic conditional independence testing method with finite-sample guarantees

Conditional independence testing (CIT) is essential for reliable scientific discovery, as it prevents spurious findings and enables controlled feature selection. Recent CIT methods leverage machine learning (ML) models as surrogates for the underlying distribution. However, model-agnostic approaches typically require a train–test split, which can reduce statistical power.  

We introduce **Semi-knockoffs**, a CIT method that can accommodate any pre-trained model, avoids the train–test split, and provides valid p-values as well as false discovery rate (FDR) control in high-dimensional settings. Unlike methods that rely on the model–$X$ assumption (i.e., known input distribution), Semi-knockoffs only require conditional expectations for continuous variables, making the procedure less restrictive and more practical for integration with machine learning workflows.  

To ensure validity when these expectations are estimated, we provide two new theoretical results:  
1. **Stability** for regularized models trained with a null feature.  
2. **Double-robustness**, which allows power to be retained even when one component is misspecified.  

All experimental figures, along with the CSV files needed to reproduce them, are available in the `results` folder, and the `src` folder contains the code to replicate the experiments.  
For comparisons with Knockoffs, dCRT, and LOCO, we use the implementations from **Hidimstats**, and for HRT, we rely on **pyHRT**.
