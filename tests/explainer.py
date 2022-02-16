# TODO to be finished
"""
from unittest import TestCase
import lightgbm as lgbm
import numpy as np

from gamexplainer.datasets import dataset_from_fun, fun_interaction
from gamexplainer import GamExplainer


class TestExplainer(TestCase):
    def setUp(self):
        dataset_size = 100
        synth_df = dataset_from_fun(100, 6, fun_interaction, rnd_gen=np.random.default_rng(42))
        train_ratio = 0.7
        train_size = int(dataset_size * train_ratio)
        X = synth_df.drop("y", axis=1).values
        y = synth_df["y"].values
        self.X_train = X[:train_size]
        self.X_test = X[train_size:]
        self.y_train = y[:train_size]
        self.y_test = y[train_size:]
        self.forest = lgbm.LGBMRegressor(num_leaves=8, n_estimators=2, random_state=42)
        self.forest.fit(self.X_train, self.y_train)

    def test_lgbm_explain(self):
        forest_to_explain = forest
        explainer = GamExplainer()
        explainer.explain(forest_to_explain)

    def test_explainer_synth_data():
        X_train, X_test, y_train, y_test = datasets.load_dataset(6, datasets.no_inter_sample_fun)
        train_data = lgbm.Dataset(X_train, label=y_train)
        test_data = lgbm.Dataset(X_test, label=y_test, reference=train_data)
        lgbm_info = {}
        tree_params = {
            'objective': 'regression',
            'max_depth': 6
        }
        tree_num_round = 10000
        early_stopping_rounds = 50
        tree_verbose_eval = True
        # a Booster object is returned
        forest = lgbm.train(tree_params,
                            train_data,
                            num_boost_round=tree_num_round,
                            early_stopping_rounds=early_stopping_rounds,
                            valid_sets=[test_data],
                            evals_result=lgbm_info,
                            verbose_eval=tree_verbose_eval)
        forest_to_explain = forest
        explainer = GamExplainer()
        explainer.explain(forest_to_explain)
"""