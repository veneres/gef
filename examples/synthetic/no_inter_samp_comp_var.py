from collections import defaultdict
from tqdm import tqdm
from gamexplainer import GamExplainer
import lightgbm as lgbm
import pickle
import json


def main():
    with open("config.json") as f:
        config_dict = json.load(f)
        model_path = config_dict["no_inter_model_path"]
        out_path = config_dict["sampling_comparison_var_path"]

    forest = lgbm.Booster(model_file=model_path)

    sampling_methods = ["all", "quantile", "equal", "kmeans", "equi_size"]
    range_m = range(500, 20001, 750)
    explanation_params = {"verbose": False,
                          "feat_importance_method": "gain",
                          "n_spline_terms": 5,
                          }
    acc_methods = defaultdict(list)
    for m in tqdm(range_m):
        explanation_params["sample_n"] = m
        for sampling_method in sampling_methods:
            explanation_params["sample_method"] = sampling_method
            explainer = GamExplainer(**explanation_params)
            explainer.explain(forest, lam_search_space=[0.01, 0.1, 1])
            acc_methods[sampling_method].append(explainer.loss_res)

    with open(out_path, 'wb') as f:
        pickle.dump(acc_methods, f)


if __name__ == '__main__':
    main()
