import itertools

import lightgbm as lgbm
from gamexplainer.datasets import dataset_from_fun
from sklearn.model_selection import train_test_split
from gamexplainer import GamExplainer
import numpy as np
from numpy.random import default_rng
from gamexplainer.utils import average_precision, precision_at_k
from synthetic_fun import fun_interaction
from collections import defaultdict
from tqdm import tqdm
import pickle
import argparse


def create_pairs(elements: list, n_pairs: int):
    pairs = itertools.combinations(elements, 2)
    all_pairs = list(itertools.combinations(pairs, n_pairs))
    return all_pairs


def get_relevance_list(explainer, real_interactions):
    scores = []  # scores for each pair
    for i, row in enumerate(explainer.interaction_matrix):
        for j, score in enumerate(row):
            if score != 0:
                scores.append(((i, j), score))
    scores.sort(key=lambda x: x[1], reverse=True)

    rel_list = []  # relevance for each pair sorted by score
    for (i, j), score in scores:
        rel_list.append((i, j) in real_interactions)

    return rel_list


def get_ap(explainer, real_interactions):
    rel_list = get_relevance_list(explainer, real_interactions)
    return average_precision(rel_list)


def get_p_k(explainer, real_interactions, k):
    rel_list = get_relevance_list(explainer, real_interactions)
    return precision_at_k(rel_list, k)


def main():
    parser = argparse.ArgumentParser(
        description="Test difference in finding interactions between the different proposed  strategies in gamexplainer")

    parser.add_argument('n_pairs', metavar='n', type=int, help='Number of interaction to add (max 5)')

    explanation_params = {"verbose": False,
                          "n_spline_terms": 5,
                          "n_inter_terms": 1,
                          "inter_max_distance": 32}

    args = parser.parse_args()

    n_pairs = args.n_pairs

    res_map = defaultdict(list)
    res_ap_3 = defaultdict(list)
    res_ap_5 = defaultdict(list)

    elements = list(range(5))

    real_interactions = create_pairs(elements, n_pairs)

    # 10 trial
    for inter in tqdm(real_interactions):
        noise_gen = np.random.default_rng(seed=42)

        synth_df = dataset_from_fun(n_sample=100000,
                                    n_features=5,
                                    fun=fun_interaction,
                                    random_state=42,
                                    rnd_gen=noise_gen,
                                    real_interactions=inter)

        X_train, X_test, y_train, y_test = train_test_split(synth_df.drop("y", axis=1),
                                                            synth_df["y"],
                                                            test_size=0.2,
                                                            shuffle=False)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)

        forest = lgbm.LGBMRegressor(n_estimators=1000, num_leaves=32, learning_rate=0.1, n_jobs=10)
        forest.fit(X_train, y_train)
        forest_to_explain = forest

        #for strategy in ["h_stat", "count_path", "gain_path", "pair_gain"]:
        for strategy in ["count_path"]:
            explanation_params["interaction_importance_method"] = strategy
            explainer_h_stat = GamExplainer(**explanation_params)
            explainer_h_stat.explain(forest_to_explain, lam_search_space=[0.1, 0.5])

            # Get stats
            res_map[strategy].append(get_ap(explainer_h_stat, inter))
            if res_map[strategy][-1] == 1:
                print(inter)
                break
            res_ap_3[strategy].append(get_p_k(explainer_h_stat, inter, 3))
            res_ap_5[strategy].append(get_p_k(explainer_h_stat, inter, 5))

    stats_res = {"map": res_map, "ap_3": res_ap_3, "ap_5": res_ap_5, "interactions": real_interactions}

    with open('precomputed_results/inter_strategies_bench.pickle', 'wb') as f:
        pickle.dump(stats_res, f)

    for key, values in res_map.items():
        print(f"{key}: {values}")


if __name__ == '__main__':
    main()
