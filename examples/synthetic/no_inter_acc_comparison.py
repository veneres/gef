import pickle

from sklearn.metrics import mean_squared_error, r2_score
import json
import lightgbm as lgbm
import numpy as np


def main():
    with open("config.json") as f:
        config_dict = json.load(f)
        model_path = config_dict["no_inter_model_path"]
        explainer_path = config_dict["no_inter_explainer"]
        X_test_path = config_dict["no_inter_test_X"]
        y_test_path = config_dict["no_inter_test_y"]

    forest = lgbm.Booster(model_file=model_path)
    with open(explainer_path, "rb") as f:
        explainer = pickle.load(f)

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

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
