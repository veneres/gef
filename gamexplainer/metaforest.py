import pandas as pd
import lightgbm as lgbm
import xgboost as xgb
from collections import Counter
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


class MetaForest:
    """
    Class implementation of the MetaForest, used to represent a forest in a standard way.
    This representation aims to support multiple packages, so as to retrieve all the 
    information needed by ``gamexplainer`` with standard methods/functions.
    For now the supported libraries are: 
    
        * **lightgbm**;
        * **xgboost**;
        * **sklearn.ensemble**.

    :param forest: The forest object to be represented.
    :type forest: object
    """

    def __init__(self, forest=None):
        self.forest = forest

        if isinstance(self.forest, lgbm.LGBMModel):
            self.forest = self.forest._Booster

        if isinstance(self.forest, xgb.XGBModel):
            self.forest = self.forest.get_booster()

    @staticmethod
    def _new_node(s, f, t, g, c):
        """
        Creates a new dictionary with various information about a node, i.e. the feature used
        in splits, its threshold values, the gain of splitting ecc.

        :param s: The node index.
        :type s: int
        :param f: The number of the feature used in splits.
        :type f: int
        :param t: The threshold of the feature.
        :type t: float
        :param g: The gain of splitting the node based on the current feature.
        :type g: float
        :return: A dictionary with specific data on the current node.
        :rtype: dict
        :param c: The number of observations reaching the node.
        :type c: int
        """
        tree_dict = {'split_index': s,
                     'split_feature': f,
                     'threshold': t,
                     'split_gain': g,
                     'internal_count': c,
                     'left_child': {},
                     'right_child': {}}
        return tree_dict

    def _split_aux(self):
        """
        Calculates the feature importance based on 'split' criteria.

        :return: A dictionary with the split value for each feature.
        :rtype: dict
        """
        split_imp = Counter()
        n_estimators = self.forest.n_estimators_
        estimators = self.forest.estimators_.ravel()
        for i in range(0, n_estimators):
            tree = estimators[i]
            for j in range(len(tree.tree_.feature)):
                f = tree.tree_.feature[j]
                if f != -2:
                    split_imp[f] += 1
        for k in range(self.forest.n_features_):
            if k not in split_imp:
                split_imp[k] = 0
        split_imp = {k: v for k, v in sorted(split_imp.items(), key=lambda item: item[0], reverse=False)}
        return split_imp

    def _df_to_dict_xgb(self, tree_df, i, n, offset):
        """
        Creates a new dictionary with various information about the tree structure.
        Compatible with ``xgboost``.

        :param tree_df: A single tree of the given forest.
        :type tree_df: pd.DataFrame
        :param i: The first node of the tree, normally set to 0 when calling the function.
        :type i: int
        :param n: The length of the tree.
        :type n: int
        :param offset: Leaf index in in the dataframe considered.
        :type offset: int
        :return: A dictionary with specific data about the tree structure.
        :rtype: dict
        """
        if i >= n:
            return {}
        elif tree_df.loc[i]['Feature'] == "Leaf":
            return {
                'leaf_index': tree_df.loc[i]['Node'] - offset,
                'leaf_value': tree_df.loc[i]['Gain'],
                'leaf_count': tree_df.loc[i]['Cover']
            }
        else:
            temp = MetaForest._new_node(tree_df.loc[i]['Node'], tree_df.loc[i]['Feature'], tree_df.loc[i]['Split'],
                                        tree_df.loc[i]['Gain'], tree_df.loc[i]['Cover'])
            root = temp
            root['left_child'] = self._df_to_dict_xgb(tree_df, 2 * i + 1, n, offset)
            root['right_child'] = self._df_to_dict_xgb(tree_df, 2 * i + 2, n, offset)
        return root

    def _df_to_dict_gb(self, tree_df, i, n):
        """
        Creates a new dictionary with various information about the tree structure.
        Compatible with ``sklearn.ensemble``.

        :param tree_df: A single tree of the given forest.
        :type tree_df: pd.DataFrame
        :param i: The first node of the tree, normally set to 0 when calling the function.
        :type i: int
        :param n: The length of the tree.
        :type n: int
        :return: A dictionary with specific data about the tree structure.
        :rtype: dict
        """
        if i >= n:
            return {}
        elif tree_df.loc[i]['Child_left'] == tree_df.loc[i]['Child_right']:
            return {
                'leaf_index': tree_df.loc[i]['Child_left'],
                'leaf_value': tree_df.loc[i]['Impurity'],
                'leaf_count': tree_df.loc[i]['Cover']
            }
        else:
            temp = MetaForest._new_node(i, tree_df.loc[i]['Feature'], tree_df.loc[i]['Threshold'],
                                        tree_df.loc[i]['Impurity'], tree_df.loc[i]['Cover'])
            root = temp
            root['left_child'] = self._df_to_dict_gb(tree_df, tree_df.loc[i]['Child_left'], n)
            root['right_child'] = self._df_to_dict_gb(tree_df, tree_df.loc[i]['Child_right'], n)
        return root

    def _df_to_tree_info_xgb(self):
        """
        Creates a new dictionary with various information about the forest, represented in JSON format.
        Compatible with ``xgboost``.

        :return: A dictionary with specific information about the forest.
        :rtype: dict
        """
        booster = self.forest
        trees_df = booster.trees_to_dataframe()
        feat_names = booster.feature_names
        trees_df['Feature'] = trees_df['Feature'].apply(lambda x: feat_names.index(x) if x != 'Leaf' else 'Leaf')
        n_trees = trees_df.loc[len(trees_df) - 1]['Tree'] + 1
        grouped = trees_df.groupby(trees_df.Tree)
        trees_list = []
        for i in range(0, n_trees):
            tree_dict = {'tree_index': i}
            tree = grouped.get_group(i).reset_index()
            n = len(tree)
            offset = int(tree.query('Feature == "Leaf"').head(1)["Node"])
            tree_dict['tree_structure'] = self._df_to_dict_xgb(tree, 0, n, offset)
            trees_list.append(tree_dict)
        tree_info = {'tree_info': trees_list}
        return tree_info

    def _df_to_tree_info_gb(self):
        """
        Creates a new dictionary with various information about the forest, represented as JSON format.
        Compatible with ``sklearn.ensemble``.

        :return: A dictionary with specific information about the forest.
        :rtype: dict
        """
        gbr = self.forest
        n_estimators = gbr.n_estimators_
        estimators = gbr.estimators_.ravel()
        trees_list = []
        for i in range(0, n_estimators):
            tree = estimators[i]
            tree_df = pd.DataFrame({
                'Child_left': tree.tree_.children_left,
                'Child_right': tree.tree_.children_right,
                'Feature': tree.tree_.feature,
                'Threshold': tree.tree_.threshold,
                'Impurity': tree.tree_.impurity,
                'Cover': tree.tree_.n_node_samples
            }, dtype=object)
            num_leaves = len(tree_df.query('Child_left == Child_right'))
            leaf_index = list(range(num_leaves))
            for j in range(len(tree_df)):
                if tree_df.loc[j]['Child_left'] == tree_df.loc[j]['Child_right']:
                    tree_df.iloc[j][['Child_left', 'Child_right']] = leaf_index.pop(0)
            tree_dict = {'tree_index': i}
            n = len(tree_df)
            tree_dict['tree_structure'] = self._df_to_dict_gb(tree_df, 0, n)
            trees_list.append(tree_dict)
        tree_info = {'tree_info': trees_list}
        return tree_info

    def dump_model(self):
        """
        Dumps Booster to JSON format.
        An exception is raised if the given forest is not supported.

        :return: JSON format of Booster.
        :rtype: dict
        """
        if type(self.forest) is lgbm.Booster:
            return self.forest.dump_model()
        elif type(self.forest) is xgb.Booster:
            return self._df_to_tree_info_xgb()
        elif type(self.forest) is GradientBoostingRegressor or type(self.forest) is GradientBoostingClassifier:
            return self._df_to_tree_info_gb()
        else:
            raise Exception(f"{type(self.forest)} not supported")

    def feature_names(self):
        """
        Gets feature names.
        An exception is raised if the given forest is not supported.

        :return: List with features' names.
        :rtype: list
        """
        if type(self.forest) is lgbm.Booster:
            return self.forest.feature_name()
        elif type(self.forest) is xgb.Booster:
            return self.forest.feature_names
        elif type(self.forest) is GradientBoostingRegressor or type(self.forest) is GradientBoostingClassifier:
            names_list = [f"x_{i}" for i in range(self.forest.n_features_)]
            return names_list
        else:
            raise Exception(f"{self.forest} not supported")

    # only features used in splits
    def feature_importances(self, imp_type: str):
        """
        Gets feature importances.
        An exception is raised if the given forest or the importance type is not supported.

        :param imp_type: The way the importance is calculated. If “split”, result contains the number of times 
            the feature is used to split the data across all trees. If “gain”, result contains the average gain 
            of the feature when it is used in splits. An exception is raised if the given type is not supported.
        :type imp_type:
        :return: List with pairs of feature names and their importances.
        :rtype: list
        """
        if type(self.forest) is lgbm.Booster:
            features_name = self.forest.feature_name()
            if imp_type == "split":
                feature_importance = [(features_name[i], value) for i, value in
                                      enumerate(self.forest.feature_importance(importance_type="split"))]
            elif imp_type == "gain":
                feature_importance = [(features_name[i], value) for i, value in
                                      enumerate(self.forest.feature_importance(importance_type="gain"))]
            else:
                raise Exception(f"Importance type {imp_type} not supported")

        elif type(self.forest) is xgb.Booster:
            if imp_type == "split":
                feature_importance = [(key, value) for key, value in
                                      self.forest.get_score(importance_type="weight").items()]
            elif imp_type == "gain":
                feature_importance = [(key, value) for key, value in
                                      self.forest.get_score(importance_type="gain").items()]
            else:
                raise Exception(f"Importance type {imp_type} not supported")
            trees_df = self.forest.trees_to_dataframe()
            feat_names = sorted(set(trees_df['Feature']))
            for feat_name in self.forest.feature_names:
                if feat_name not in feat_names:
                    feature_importance.append((feat_name, 0))

        elif type(self.forest) is GradientBoostingRegressor or type(self.forest) is GradientBoostingClassifier:
            if imp_type == "gain":
                feature_importance = [(f"x_{i}", self.forest.feature_importances_[i]) for i in
                                      range(self.forest.n_features_)]
            elif imp_type == "split":
                split_imp = self._split_aux()
                feature_importance = [(f"x_{i}", value) for i, value in
                                      split_imp.items()]
            else:
                raise Exception(f"Importance type {imp_type} not supported")

        else:
            raise Exception(f"{type(self.forest)} not supported")
        imp_sum = 0
        for pair in feature_importance:
            imp_sum += pair[1]
        final_feat = []
        for j in range(len(feature_importance)):
            # normalizing feature importances
            final_feat.append((feature_importance[j][0], feature_importance[j][1] / imp_sum))
        return final_feat

    def predict(self, data):
        """
        Makes a prediction.
        An exception is raised if the given forest is not supported.

        :param data: Data source for prediction.
        :type data: pd.DataFrame
        :return: Prediction result. 
        :rtype: np.array
        """
        if type(self.forest) is lgbm.Booster:
            return self.forest.predict(data)
        elif type(self.forest) is xgb.Booster:
            dtrain = xgb.DMatrix(data)
            return self.forest.predict(dtrain)
        elif type(self.forest) is GradientBoostingRegressor or type(self.forest) is GradientBoostingClassifier:
            return self.forest.predict(data)
        else:
            raise Exception(f"{self.forest} not supported")

    def num_features(self):
        """
        Gets number of features.
        An exception is raised if the given forest is not supported.

        :return: The number of features. 
        :rtype: int
        """
        if type(self.forest) is lgbm.Booster:
            return self.forest.num_feature()
        elif type(self.forest) is xgb.Booster:
            return self.forest.num_features()
        elif type(self.forest) is GradientBoostingRegressor or type(self.forest) is GradientBoostingClassifier:
            return self.forest.n_features_
        else:
            raise Exception(f"{self.forest} not supported")

    def get_wrapped_forest(self):
        return self.forest
