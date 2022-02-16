import unittest
from unittest import TestCase
import numpy as np
from test_forests import load_lgbm_forest, load_dataset, load_sklearn_forest, load_xgboost_forest
from gamexplainer.metaforest import MetaForest


class TestMetaforest(TestCase):
    def setUp(self) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = load_dataset()
        self.lgbm_forest = load_lgbm_forest()
        self.sklearn_forest = load_sklearn_forest()
        self.xgboost_forest = load_xgboost_forest()

    def __check_tree_structure(self, tree_structure: dict) -> None:
        if "leaf_value" in tree_structure or len(tree_structure) == 0:
            return
        keys_to_check = ["split_feature", "split_gain", "threshold", "left_child", "right_child"]

        for key in keys_to_check:
            value = tree_structure.get(key, None)
            self.assertIsNotNone(value)

        self.__check_tree_structure(tree_structure["left_child"])
        self.__check_tree_structure(tree_structure["right_child"])

    def __check_forest_structure(self, tree_dump: dict) -> None:
        tree_info = tree_dump.get("tree_info", None)
        self.assertIsNotNone(tree_info)
        for tree in tree_info:
            tree_index = tree.get("tree_index", None)
            self.assertIsNotNone(tree_index)

            tree_structure = tree.get("tree_structure", None)
            self.assertIsNotNone(tree_structure)
            self.__check_tree_structure(tree_structure)

    def __check_feat_imp(self, tree_structure: dict, imp_type: str, acc: list) -> None:
        if "leaf_value" in tree_structure or len(tree_structure) == 0:
            return
        if imp_type == "gain":
            node_value = tree_structure["split_gain"]
        else:
            node_value = 1

        acc[tree_structure["split_feature"]] += node_value
        self.__check_feat_imp(tree_structure["left_child"], imp_type, acc)
        self.__check_feat_imp(tree_structure["right_child"], imp_type, acc)

    def __check_meta_forest(self, meta_forest, original_forest):

        # Check tree structure
        model_dump = meta_forest.dump_model()
        self.__check_forest_structure(model_dump)

        # Check only split feats importance
        n_feature_to_assess = meta_forest.num_features()

        feat_imp_computed = [0 for _ in range(n_feature_to_assess)]
        for tree in model_dump["tree_info"]:
            tree_structure = tree.get("tree_structure", None)
            self.__check_feat_imp(tree_structure, "split", feat_imp_computed)
        feat_imp_computed = feat_imp_computed / np.sum(feat_imp_computed)

        for feat_index, (feat_name, value) in enumerate(meta_forest.feature_importances("split")):
            self.assertAlmostEqual(feat_imp_computed[feat_index], value)

        # Check predict
        meta_forest_pred = meta_forest.predict(self.X_test)
        original_pred = original_forest.predict(self.X_test)
        self.assertIsNone(np.testing.assert_allclose(meta_forest_pred, original_pred))

    def test_lgbm(self) -> None:
        meta_forest = MetaForest(self.lgbm_forest)
        original_forest = self.lgbm_forest
        self.__check_meta_forest(meta_forest, original_forest)

        # Check feats naming
        self.assertTrue(meta_forest.feature_names() == original_forest.feature_name_)

        # Check n features
        n_feature_to_assess = meta_forest.num_features()
        self.assertEqual(n_feature_to_assess, original_forest.n_features_)

    def test_sklearn(self) -> None:
        meta_forest = MetaForest(self.sklearn_forest)
        original_forest = self.sklearn_forest
        self.__check_meta_forest(meta_forest, original_forest)



        # Check n features
        n_feature_to_assess = meta_forest.num_features()
        self.assertEqual(n_feature_to_assess, original_forest.n_features_)

        # We lose the original name of the features in sklearn
        # self.assertTrue(meta_forest.feature_names() == original_forest.feature_name_)

    def test_xgboost(self) -> None:
        meta_forest = MetaForest(self.xgboost_forest)
        original_forest = self.xgboost_forest
        self.__check_meta_forest(meta_forest, original_forest)

        # Check n features
        n_feature_to_assess = meta_forest.num_features()
        self.assertEqual(n_feature_to_assess, len(original_forest.feature_importances_))

        # We lose the original name of the features in sklearn
        # self.assertTrue(meta_forest.feature_names() == original_forest.feature_name_)


if __name__ == '__main__':
    unittest.main()
