# simple script to create and load test forests for LightGBM, skLearn and XGBoost
from typing import Tuple

import numpy
import numpy as np
import os
from gamexplainer.datasets import dataset_from_fun, fun_interaction
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import lightgbm as lgbm
from sklearn import ensemble as sklearn_ensemble
from sklearn.tree import plot_tree
import xgboost as xgb

CD = os.path.dirname(os.path.realpath(__file__))
IMG_PATH = os.path.join(CD, "img")
LGBM_PREFIX = "lgbm"
SKLEARN_PREFIX = "sklearn"
XGBOOST_PREFIX = "xgboost"
DATASET_PATH = os.path.join(CD, "dataset.csv")
LGBM_DUMP_PATH = os.path.join(CD, "lgbm.joblib")
SKLEARN_DUMP_PATH = os.path.join(CD, "sklearn.joblib")
XGBOOST_DUMP_PATH = os.path.join(CD, "xgboost.joblib")


def load_dataset() -> Tuple[numpy.array, numpy.array, numpy.array, numpy.array]:
    dataset_size = 10000
    try:
        synth_df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        synth_df = dataset_from_fun(dataset_size, 6, fun_interaction, rnd_gen=np.random.default_rng(42))
        synth_df.to_csv(DATASET_PATH, index=False)
    train_ratio = 0.7
    train_size = int(dataset_size * train_ratio)
    X = synth_df.drop("y", axis=1)
    y = synth_df["y"]
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    return X_train, X_test, y_train, y_test


def load_lgbm_forest() -> lgbm.LGBMRegressor:
    try:
        forest = joblib.load(LGBM_DUMP_PATH)
    except FileNotFoundError or KeyError:
        X_train, X_test, y_train, y_test = load_dataset()
        forest = lgbm.LGBMRegressor(n_estimators=5, num_leaves=16, random_state=42)
        forest.fit(X_train, y_train)
        joblib.dump(forest, LGBM_DUMP_PATH)
    return forest


def load_sklearn_forest() -> sklearn_ensemble.GradientBoostingRegressor:
    try:
        forest = joblib.load(SKLEARN_DUMP_PATH)
    except FileNotFoundError or KeyError:
        X_train, X_test, y_train, y_test = load_dataset()
        forest = sklearn_ensemble.GradientBoostingRegressor(n_estimators=5, random_state=42, max_depth=3)
        forest.fit(X_train, y_train)
        joblib.dump(forest, SKLEARN_DUMP_PATH)
    return forest


def load_xgboost_forest() -> xgb.XGBRegressor:
    try:
        forest = joblib.load(XGBOOST_DUMP_PATH)
    except FileNotFoundError or KeyError:
        X_train, X_test, y_train, y_test = load_dataset()
        forest = xgb.XGBRegressor(n_estimators=5, max_depth=3, random_state=42)
        forest.fit(X_train, y_train)
        joblib.dump(forest, XGBOOST_DUMP_PATH)
    return forest


def print_test_forests() -> None:
    n_trees = 5
    # lgbm trees images
    forest = load_lgbm_forest()
    for i in range(n_trees):
        lgbm.plot_tree(forest, ax=plt.gca(), tree_index=i, show_info=["split_gain", 'data_percentage', "leaf_weight"])
        plt.savefig(os.path.join(IMG_PATH, f"{LGBM_PREFIX}_{i}.pdf"), dpi=600)
    # sklearn trees images
    forest = load_sklearn_forest()
    trees = forest.estimators_.ravel()
    for i, tree in enumerate(trees):
        plt.figure(i)
        plot_tree(tree)
        plt.savefig(os.path.join(IMG_PATH, f"{SKLEARN_PREFIX}_{i}.pdf"), dpi=600)
    # xgboost trees images
    forest = load_xgboost_forest()
    for i in range(n_trees):
        plt.figure(i)
        xgb.plot_tree(forest, num_trees=i)
        plt.savefig(os.path.join(IMG_PATH, f"{XGBOOST_PREFIX}_{i}.pdf"), dpi=600)


if __name__ == '__main__':
    print_test_forests()
