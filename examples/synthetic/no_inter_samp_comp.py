from gamexplainer import GamExplainer
from tqdm import tqdm
import lightgbm as lgbm
import json
import pickle


def main():
    with open("config.json") as f:
        config_dict = json.load(f)
        model_path = config_dict["no_inter_model_path"]
        out_path = config_dict["sampling_comparison_path"]

    forest = lgbm.Booster(model_file=model_path)

    sampling_methods = ["all", "quantile", "equal", "kmeans", "equi_size"]
    results = {}
    for i, sampling_method in tqdm(enumerate(sampling_methods)):
        explainer = GamExplainer(sample_method=sampling_method,
                                 n_spline_terms=5,
                                 n_inter_terms=0,
                                 sample_n=100)
        explainer.explain(forest, lam_search_space=[0.1, 0.5, 1])
        results[sampling_method] = explainer

    with open(out_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
