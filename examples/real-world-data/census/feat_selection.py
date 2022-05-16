import numpy as np
from tqdm import tqdm
from math import comb
from gamexplainer import GamExplainer
import lightgbm as lgbm
import json


def main():
    with open("config.json") as f:
        config_dict = json.load(f)
        model_path = config_dict["model_path"]
        feat_sel_out = config_dict["feat_sel_out"]

    forest = lgbm.Booster(model_file=model_path)
    range_n_splines = range(1, 11)
    range_n_inter = range(0, 9)
    explanation_params = {"verbose": False,
                          "sample_method": "all",
                          "classification": True,
                          "inter_max_distance": 256}

    acc = np.zeros((len(range_n_splines), len(range_n_inter)))
    for i, n_splines in enumerate(range_n_splines):
        explanation_params["n_spline_terms"] = n_splines
        for j, n_inter in enumerate(range_n_inter):
            if n_inter > comb(n_splines, 2):
                continue
            explanation_params["n_inter_terms"] = n_inter
            explainer = GamExplainer(**explanation_params)
            _ = explainer.explain(forest, lam_search_space=[0.1, 1])
            print(f"Fit {n_splines=}, {n_inter=} completed")
            acc[i, j] = explainer.loss_res

    np.save(feat_sel_out, acc)


if __name__ == '__main__':
    main()
