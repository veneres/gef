import json
import pickle
from collections import defaultdict

from tqdm import tqdm

from gamexplainer import GamExplainer
import lightgbm as lgbm


def main():
    with open("config.json") as f:
        config_dict = json.load(f)
        model_path = config_dict["model_path"]
        sampling_analysis_out = config_dict["sampling_analysis_out"]

    forest = lgbm.Booster(model_file=model_path)

    sampling_methods = ["all", "quantile", "equal", "kmeans", "equi_size"]
    range_m = range(50, 5001, 250)

    explanation_params = {"verbose": False,
                          "interaction_importance_method": "count_path",
                          "feat_importance_method": "gain",
                          "n_spline_terms": 4,
                          "sample_method": "all",
                          "n_spline_per_term": 50,
                          "inter_max_distance": 64,
                          "n_inter_terms": 0,
                          "n_sample_gam": int(1e5),
                          "portion_sample_test": 0.3,
                          "classification": True
                          }

    acc_methods = defaultdict(list)
    for m in tqdm(range_m):
        explanation_params["sample_n"] = m
        for sampling_method in sampling_methods:
            explanation_params["sample_method"] = sampling_method
            explainer = GamExplainer(**explanation_params)
            _ = explainer.explain(forest, lam_search_space=[0.1, 1])

            acc_methods[sampling_method].append(explainer.loss_res)

    with open(sampling_analysis_out, 'wb') as f:
        pickle.dump(acc_methods, f)


if __name__ == '__main__':
    main()
