# Forest Gam Explainer
Explain Tree Ensemble through Generalized Additive Models (GAMs)

## Installation
1. Install all the dependencies with `pip install -r requirements.txt`
2. Install the package in edit mode with `pip install -e .`

## Example of usage
```python
explainer = GamExplainer(verbose=True,feat_importance_method="gain", n_sample_gam=100000, n_sample_test=5000)
gam = explainer.explain(forest_to_explain)
plot_feature_importance(explainer)
plot_splines(explainer)
plot_thresholds_hist(explainer, feat_name)
```
Inside the folder notebooks-examples you can find example of usage for this package
