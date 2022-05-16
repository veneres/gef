import json
import pickle

from gamexplainer import GamExplainer
import lightgbm as lgbm


def main():
    with open("config.json") as f:
        config_dict = json.load(f)
        model_path = config_dict["model_path"]
        explainer_out = config_dict["explainer_out"]

    forest = lgbm.Booster(model_file=model_path)
    explanation_params = {
        "verbose": False,
        "interaction_importance_method": "count_path",
        "feat_importance_method": "gain",
        "n_spline_terms": 7,
        "sample_method": "equi_size",
        "sample_n": 4500,
        "n_spline_per_term": 50,
        "inter_max_distance": 64,
        "n_inter_terms": 0,
        "n_sample_gam": int(1e5),
        "portion_sample_test": 0.3,
        "classification": False
    }
    explainer = GamExplainer(**explanation_params)
    explainer.explain(forest)

    with open(explainer_out, "wb") as f:
        pickle.dump(explainer, f)


if __name__ == '__main__':
    main()
