import pickle

from gamexplainer import GamExplainer
import json
import lightgbm as lgbm


def main():
    with open("config.json") as f:
        config_dict = json.load(f)
        model_path = config_dict["no_inter_model_path"]
        out_path = config_dict["no_inter_explainer"]

    forest = lgbm.Booster(model_file=model_path)

    explanation_params = {"n_spline_terms": 5,
                          "sample_method": "equi_size",
                          "sample_n": 12000,
                          "verbose": True,
                          "n_inter_terms": 0}

    explainer = GamExplainer(**explanation_params)
    explainer.explain(forest, lam_search_space=[0.01, 0.1, 1])

    with open(out_path, "wb") as f:
        pickle.dump(explainer, f)


if __name__ == '__main__':
    main()
