import numpy as np
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from gamexplainer import GamExplainer
from gamexplainer.datasets import dataset_from_fun
from synthetic_fun import fun_interaction


def main():
    noise_gen = np.random.default_rng(seed=42)
    synth_df = dataset_from_fun(n_sample=100000,
                                n_features=5,
                                fun=fun_interaction,
                                random_state=42,
                                rnd_gen=noise_gen,
                                real_interactions=((0, 1), (0, 4), (1, 4)))

    X_train, X_test, y_train, y_test = train_test_split(synth_df.drop("y", axis=1),
                                                        synth_df["y"],
                                                        test_size=0.2,
                                                        shuffle=False)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)

    forest = lgbm.LGBMRegressor(n_estimators=1000, num_leaves=32, learning_rate=0.01, n_jobs=20)
    forest.fit(X_train, y_train)

    explanation_params = {"n_spline_terms": 5,
                          "sample_method": "equi_size",
                          "sample_n": 12000,
                          "inter_max_distance": 32,
                          "verbose": True,
                          "n_inter_terms": 3}

    explainer = GamExplainer(**explanation_params)
    explainer.explain(forest, lam_search_space=[0.01, 0.05, 0.1, 1])

    print("MSE")
    print("-" * 50)
    print("MSE forest vs original labels")
    print(round(mean_squared_error(y_test, forest.predict(X_test), squared=False), 3))
    print("MSE explainer vs original labels")
    print(round(mean_squared_error(y_test, explainer.gam.predict(X_test), squared=False), 3))
    print("MSE explainer vs forest labels")
    print(round(mean_squared_error(forest.predict(X_test), explainer.gam.predict(X_test), squared=False), 3))

    print("R2")
    print("-" * 50)
    print("R2 forest vs original labels")
    print(round(r2_score(y_test, forest.predict(X_test)), 3))
    print("R2 explainer vs original labels")
    print(round(r2_score(y_test, explainer.gam.predict(X_test)), 3))
    print("R2 explainer vs forest labels")
    print(round(r2_score(forest.predict(X_test), explainer.gam.predict(X_test)), 3))


if __name__ == '__main__':
    main()
