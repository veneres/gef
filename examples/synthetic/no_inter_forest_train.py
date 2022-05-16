import lightgbm as lgbm
from gamexplainer.datasets import dataset_from_fun
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from synthetic_fun import fun_without_interaction
import json


def main():
    with open("config.json") as f:
        config_dict = json.load(f)
        model_path = config_dict["no_inter_model_path"]
        out_path_test_X = config_dict["no_inter_test_X"]
        out_path_test_y = config_dict["no_inter_test_y"]

    noise_gen = np.random.default_rng(seed=42)
    synth_df = dataset_from_fun(n_sample=10000,
                                n_features=5,
                                fun=fun_without_interaction,
                                random_state=42,
                                rnd_gen=noise_gen)
    X_train, X_test, y_train, y_test = train_test_split(synth_df.drop("y", axis=1),
                                                        synth_df["y"],
                                                        test_size=0.2,
                                                        shuffle=False)

    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.25,
                                                      shuffle=False)
    parameters = {
        "n_jobs": [40],
        "deterministic": [True],
        "force_col_wise": [True],
        "n_estimators": np.geomspace(10, 1000, num=3, dtype=int),
        "num_leaves": np.geomspace(32, 256, num=4, dtype=int),
        "learning_rate": np.geomspace(1e-4, 1e-1, num=4),
        "verbose": [-1]
    }
    CV_regressor = GridSearchCV(lgbm.LGBMRegressor(), parameters, scoring="neg_root_mean_squared_error")
    CV_regressor.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_val.values, y_val.values)])

    forest = lgbm.LGBMRegressor(**CV_regressor.best_params_)

    forest.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_val.values, y_val.values)])

    print("Best parameters for the model found")
    print(CV_regressor.best_params_)

    forest.booster_.save_model(model_path)
    print("Saved to:")
    print(model_path)

    print("Saving the test datasets...")
    np.save(out_path_test_X, X_test)
    print(f"X_test saved to: {out_path_test_X}")
    np.save(out_path_test_y, y_test)
    print(f"y_test saved to: {out_path_test_y}")


if __name__ == '__main__':
    main()
