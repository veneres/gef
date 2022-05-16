import itertools
import typing
from collections import defaultdict

import numpy as np
import pandas as pd

from gamexplainer.sampling_startegies import rnd_sampling, all_sampling, equal_dist_sampling, quantile_sampling, \
    kmeans_sampling, equi_size_sampling, is_category
from gamexplainer.metaforest import MetaForest
from gamexplainer.utils import h_stat_all_pairs
from pygam import LinearGAM, LogisticGAM, s, te, f
from sklearn.metrics import mean_squared_error, accuracy_score


class GamExplainer:
    """
    Class implementation of the GAM-based Explanation of Forests (GEF).

    :param sample_n: The number of values created through a sampling technique to train the GAM.
    :type sample_n: int, optional(default=100)
    :param feat_importance_method: Is a method to get feature importance. Possible types are:

        * **gain**: the average gain of the feature when it is used in trees;
        * **split**: the number of times a feature is used to split the data across all trees.
    :type feat_importance_method: string, optional (default='gain')
    :param sample_method: Is a strategy to get a sample. Possible approaches are:

        * **random**: random selection of values from a uniform distribution;
        * **all**: uses all the threshold values range for sampling;
        * **equal**: defines subsets of thresholds for each feature with equal distance between adjacent values; 
        * **quantile**: defines subsets of quantile values;
        * **kmeans**: sampling is formed using the k-means algorithm. Return subsets of centroids of the resulting clusters.
        * **equi_size**: defines the sampling domain based on equi-sized subset of thresholds;
    :type sample_method: string, optional (default='equi_size')
    :param interaction_importance_method: Is the way to define the importance of interaction between features.
        This method exploits the forest structure. Four values are provided:

        * **gain_adj**: identifies pairs of features held in two adjacent nodes with the best importance;
        * **count_adj**: counts pairs of features that interact with each other;
        * **gain_path**: feature interaction selection based on a given distance between the nodes;
        * **count_path**: counting of identified feature interactions taking into account the given distance.
    :type interaction_importance_method: string, optional (default='gain_adj')
    :param inter_max_distance: The maximum distance between two nodes.
    :type inter_max_distance: int
    :param n_sample_gam: The number of instances on which we run the GAM.
    :type n_sample_gam: int, optional (default=1e5)
    :param portion_sample_test: The portion of the dataset to be used as test set.
    :type portion_sample_test: float, optional(default=0.3)
    :param verbose: If set to True, the explainer gives some additional information about the explanation.
    :type verbose: bool, optional(default=False)
    :param n_spline_terms: The number of splines to be used.
    :type n_spline_terms: int, optional(default=5)
    :param n_inter_terms: The number of interaction terms.
    :type n_inter_terms: int, optional(default=0)
    :param gam_params: A set of further parameters to be passed to the GAM models of the **pygam** package.
    :type gam_params: dict
    :param random_state: The random seed.
    :type random_state: int, optional(default=42)
    :param fixed_feat_inter: A set of fixed interacting features.
    :type fixed_feat_inter: list
    :param classification: If set to True, the model adopted for GAM is LogisticGam, LinearGam otherwise.
    :type classification: bool, optional(default=False)
    :param cat_feat: A set of categorical features.
    :type cat_feat: list
    """

    def __init__(self,
                 sample_n=100,
                 feat_importance_method="gain",
                 sample_method="equi_size",
                 interaction_importance_method="count_path",
                 inter_max_distance=None,
                 n_sample_gam=int(1e5),
                 portion_sample_test=0.3,
                 verbose=False,
                 n_spline_terms=5,
                 n_spline_per_term=50,
                 n_inter_terms=0,
                 gam_params=None,
                 random_state=42,
                 fixed_feat_inter=None,
                 classification=False,
                 cat_feat=None
                 ):

        self.sample_method_map = {
            "random": rnd_sampling,  # baseline
            "all": all_sampling,
            "equal": equal_dist_sampling,
            "equi_size": equi_size_sampling,
            "quantile": quantile_sampling,
            "kmeans": kmeans_sampling,
        }
        self.feat_importance_method_map = {
            "split": self._feat_importance_split,
            "gain": self._feat_importance_gain,
        }
        self.interaction_importance_method_map = {
            "gain_adj": self._gain_adj_interaction,
            "count_adj": self._count_adj_interaction,
            "gain_path": self._gain_path_interaction,
            "count_path": self._count_path_interaction,
            "h_stat": self._h_stat_interaction,
            "pair_gain": self._pair_gain_interaction
        }
        if sample_method not in self.sample_method_map:
            raise NotImplementedError(f"Sampling method not implemented {sample_method=}")

        if feat_importance_method not in self.feat_importance_method_map:
            raise NotImplementedError(f"Feat importance method not implemented {sample_method=}")
        self.sample_n = sample_n
        self.feat_importance_method = feat_importance_method
        self.interaction_importance_method = self.interaction_importance_method_map[interaction_importance_method]
        self.sample_method = sample_method
        self.n_sample_gam = n_sample_gam
        self.n_sample_test = int(n_sample_gam * portion_sample_test)
        self.verbose = verbose
        self.n_splines = n_spline_terms
        self.n_inter_terms = n_inter_terms
        self.n_spline_per_term = n_spline_per_term
        self.inter_max_distance = inter_max_distance
        self.fixed_feat_inter = fixed_feat_inter
        self.classification = classification
        self.gam_params = {}
        self.cat_feat = cat_feat

        if gam_params is not None:
            self.gam_params = gam_params
        np.random.seed(random_state)

        # variable to fill during the explanation procedure
        self.lam_search_space = None
        self.fitted = False
        self.feature_dict = None
        self.forest: typing.Optional[MetaForest] = None
        self.mif = None
        self.interaction_matrix = None
        self.interactions = None
        self.sampled = None
        self.loss_res = None
        self.feature_importances = None
        self.gam = None
        self.X_synth_train = None
        self.y_synth_train = None

        if verbose:
            print(f"Model parameteres:\n{sample_n=}\n{sample_method=}\n{n_sample_gam=}\n{portion_sample_test=}"
                  f"\n{n_spline_terms=}\n{n_inter_terms=}")

    def _collect_tree_info(self, tree):
        """
        Collects some information from the tree structure, i.e. 'split_feature', 'threshold', 'left_child' and 'right_child'.

        :param tree: The internal tree structure in JSON format.
        :type tree: dict
        :return: A dictionary containing dumped information about the given tree.
        :rtype: dict
        """
        info = {
            'split': [],
            'threshold': []
        }
        if 'split_feature' in tree:
            info['split'].append(tree['split_feature'])
            info['threshold'].append(tree['threshold'])

            left_info = self._collect_tree_info(tree['left_child'])
            right_info = self._collect_tree_info(tree['right_child'])

            info['split'] = info['split'] + left_info['split'] + right_info['split']
            info['threshold'] = info['threshold'] + left_info['threshold'] + right_info['threshold']

        return info

    def _retrieve_thresholds(self):
        """
        Returns features with their threshold values. 
        
        :return: A dictionary with feature names as keys and thresholds as values.
        :rtype: dict
        """

        forest_dump = self.forest.dump_model()
        tree_info = []
        feature_names = self.forest.feature_names()

        for tree in forest_dump['tree_info']:
            tree_info.append(self._collect_tree_info(tree['tree_structure']))

        feature_dict = defaultdict(list)

        for info in tree_info:
            for i in range(0, len(info['split'])):
                feature_dict[feature_names[info['split'][i]]].append(info['threshold'][i])

        return feature_dict

    def _add_path_interaction(self, subtree, max_distance, fun_used, root=None):
        """
        Saves into the 'interaction_matrix' class field information about features' interactions.
        
        :param subtree: The left or the right subtree of the main tree.
        :type subtree: dict
        :param max_distance: A specified distance between the nodes of the given tree.
        :type max_distance: int
        :param fun_used: Calculates the interaction value between two nodes.
        :type fun_used: function
        """
        if max_distance < 0 or 'split_feature' not in subtree:
            return
        current_feat = subtree.get('split_feature')
        if root is None:
            root = subtree
        root_feat = root.get("split_feature", None)
        # taking into account only the most important features...

        current_feat_name = self.forest.feature_names()[current_feat]

        if current_feat_name in self.mif and current_feat != root_feat:
            row = min(current_feat, root_feat)
            col = max(current_feat, root_feat)
            self.interaction_matrix[row][col] += fun_used(subtree, root)

        self._add_path_interaction(subtree['left_child'], max_distance - 1, fun_used, root)
        self._add_path_interaction(subtree['right_child'], max_distance - 1, fun_used, root)

    def _dfs_tree_dump(self, subtree, fun_to_call, *args):
        """
        Applies the given function to the subtree.

        :param subtree: A branch of a single tree.
        :type subtree: dict
        :param fun_to_call: The function to be applied to the subtree's structure.
        :type fun_to_call: function
        :param \*args: Other parameters.
        :type \*args: argument list
        """
        fun_to_call(subtree, *args)
        if "left_child" in subtree:
            self._dfs_tree_dump(subtree["left_child"], fun_to_call, *args)
        if "right_child" in subtree:
            self._dfs_tree_dump(subtree["right_child"], fun_to_call, *args)

    def _compute_path_interaction(self, max_distance, fun_used):
        """
        Initialize the interaction matrix and computes the feature interactions to put into it.
        
        :param max_distance: A specified distance between the nodes of the given tree.
        :type max_distance: int
        :param fun_used: Calculates the interaction value between two nodes.
        :type fun_used: function
        """
        n_features = len(self.forest.feature_names())
        self.interaction_matrix = np.zeros((n_features, n_features), dtype=float)

        forest_dump = self.forest.dump_model()
        forest_tree_info = forest_dump['tree_info']
        tree_dumps = [tree["tree_structure"] for tree in forest_tree_info]

        for tree_dump in tree_dumps:
            self._dfs_tree_dump(tree_dump, self._add_path_interaction, max_distance, fun_used)

    def _count_adj_interaction(self):
        """
        Computes the interaction based on the proximity of the nodes.
        Given two nodes it returns 1.
        """
        self._compute_path_interaction(1, lambda node_1, node_2: 1)

    def _gain_adj_interaction(self):
        """
        Computes the interaction based on the proximity of the nodes.
        Given two nodes it gives the minimum value of the nodes' importance.
        """
        self._compute_path_interaction(1, lambda node_1, node_2: min(node_1['split_gain'], node_2['split_gain']))

    def _count_path_interaction(self):
        """
        Computes the interaction based on the maximum distance between the nodes.
        Given two nodes it gives 1.
        """
        self._compute_path_interaction(self.inter_max_distance, lambda node_1, node_2: 1)

    def _gain_path_interaction(self):
        """
        Computes the interaction based on the maximum distance between the nodes.
        Given two nodes it gives the minimum value of the nodes' importance..
        """
        self._compute_path_interaction(self.inter_max_distance,
                                       lambda node_1, node_2: min(node_1['split_gain'], node_2['split_gain']))

    def _h_stat_interaction(self):
        h_stats = h_stat_all_pairs(self.forest.get_wrapped_forest(), self.X_synth_train, self.mif, verbose=self.verbose)
        n_features = len(self.forest.feature_names())
        feat_name_to_index = {feat: i for i, feat in enumerate(self.forest.feature_names())}
        self.interaction_matrix = np.zeros((n_features, n_features), dtype=float)
        for (feat_1, feat_2), h_stat in h_stats.items():
            feat_1 = feat_name_to_index[feat_1]
            feat_2 = feat_name_to_index[feat_2]
            row = min(feat_1, feat_2)
            col = max(feat_1, feat_2)
            self.interaction_matrix[row][col] = h_stat
        return 0

    def _pair_gain_interaction(self):
        imp_gain = self.forest.feature_importances("gain")
        feat_name_to_index = {feat: i for i, feat in enumerate(self.forest.feature_names())}
        imp_gain = [(feat_name_to_index[feat], value) for feat, value in imp_gain]
        scores = []
        n_features = len(self.forest.feature_names())
        for (f1, f1_value), (f2, f2_value) in itertools.combinations(imp_gain, 2):
            scores.append(((f1, f2), f1_value + f2_value))

        self.interaction_matrix = np.zeros((n_features, n_features), dtype=float)
        for (feat_1, feat_2), score in scores:
            row = min(feat_1, feat_2)
            col = max(feat_1, feat_2)
            self.interaction_matrix[row][col] = score
        return 0

    def _create_dataset(self, n):
        """
        Creates a dataset for the GAM.
        
        :param n: The number of the features.
        :type n: int
        :return: A dataset of randomly selected features' values.
        :rtype: pd.DataFrame
        """
        dataset = []
        columns = []
        # print(sampled_points.shape)
        for key, points in self.sampled.items():
            if points.shape[0] == 0:
                features = np.zeros(n, dtype=int)
            else:
                features = np.random.choice(points, n, replace=True)
            dataset.append(features)
            columns.append(key)

        dataset = np.array(dataset)
        dataset = dataset.T.reshape(-1, dataset.shape[0])
        dataset = pd.DataFrame(dataset, columns=columns)
        return dataset

    def _gam_train(self, X, y):
        """
        Trains a generalized additive model on the given data.
        
        :param X: Input space.
        :type X: pd.DataFrame
        :param y: Output space.
        :type y: pd.DataFrame
        :return: A Generalized Additive Model fitted with the training data.
        :rtype: GAM object
        """
        target = pd.Series(y)
        feats_name = X.columns.tolist()
        splines = []
        for feat in self.mif:
            if feat not in self.cat_feat:
                splines.append(s(feats_name.index(feat), n_splines=self.n_spline_per_term))
            else:
                splines.append(f(feats_name.index(feat)))  # add a factor term for categorical feature
        if len(splines) == 0:
            raise Exception("No spline created")
        terms = splines[0]
        for i in range(1, len(splines)):
            terms += splines[i]
        if self.interactions is not None:
            for f1_i, f2_i in self.interactions:
                terms += te(f1_i, f2_i)

        if self.classification:
            gam = LogisticGAM(terms, **self.gam_params)
        else:
            gam = LinearGAM(terms, **self.gam_params)

        gam.gridsearch(X.values, target.values, lam=self.lam_search_space)
        if self.verbose:
            gam.summary()

        return gam

    def _create_gam(self, X_train, y_train, X_test, y_test):
        """
        Creates a GAM model and computes the Root Mean Squared Error for regression tasks or the accuracy for classification tasks.

        :param X_train: Input train set.
        :type X_train: pd.DataFrame
        :param y_train: Output train set.
        :type y_train: pd.DataFrame
        :param X_test: Input test set.
        :type X_test: pd.DataFrame
        :param y_test: Output test set.
        :type y_test: pd.DataFrame
        :return:
        
            - A Generalized Additive Model fitted with the training data.
            - The MSE obtained.
        :rtype: (GAM object, float) 
        """
        gam_forest = self._gam_train(X_train, y_train)
        if self.verbose:
            print('GAM_LAMBDA', gam_forest.lam)

        if self.classification:
            response = gam_forest.predict_proba(X_test)
            loss_res = accuracy_score(y_test > 0.5, response > 0.5)
        else:
            response = gam_forest.predict(X_test)
            loss_res = mean_squared_error(y_test, response, squared=False)
        return gam_forest, loss_res

    def _feat_importance_split(self):
        """
        Computes the most important features based on 'split' importance type and returns them.

        :return: The most important features.
        :rtype: dict
        """

        self.feature_importances = self.forest.feature_importances("split")
        self.feature_importances.sort(key=lambda x: x[1], reverse=True)
        mif = [feat_key for feat_key, imp in self.feature_importances]
        if self.verbose:
            print(f"Most important features: {mif=}")
        return mif

    def _feat_importance_gain(self):
        """
        Computes the most important features based on 'gain' importance type and returns them.

        :return: The most important features.
        :rtype: dict
        """

        self.feature_importances = self.forest.feature_importances("gain")
        self.feature_importances.sort(key=lambda x: x[1], reverse=True)
        mif = [feat_key for feat_key, imp in self.feature_importances]
        if self.verbose:
            print(f"Most important features: {mif=}")
        return mif

    def get_feature_thresholds(self):
        if self.fitted:
            return self.feature_dict
        raise Exception("Call 'explain' the explainer before calling 'get_feature_thresholds'.")

    def explain(self, forest, lam_search_space=None):
        """
        Creates the GAM, runs it on a Booster and returns the resulted model.
        It also saves some further information in the class fields i.e. the resulting GAM model, the RMSE or the accuracy ecc.

        :param forest: The forest of trees to be explained.
        :type forest: lgbm.Booster
        :return: The trained Generalized Additive Model.
        :rtype: GAM object
        """
        if lam_search_space is None:
            self.lam_search_space = np.logspace(-3, 3, 11)
        else:
            self.lam_search_space = lam_search_space
        meta_forest = MetaForest(forest)
        self.forest = meta_forest
        self.feature_dict = self._retrieve_thresholds()

        if self.cat_feat is None:
            category_features = []
            for key in self.feature_dict.keys():
                if is_category(self.feature_dict[key]):
                    if self.verbose:
                        print(f"Feature: {self.feature_dict[key]} identified as categorical")
                    category_features.append(key)
            self.cat_feat = category_features
        else:
            for feat in self.cat_feat:
                if feat not in self.feature_dict.keys():
                    raise ValueError(f"'{feat}' not present in the set of features")

        # compute feat importance
        # compute the minimum between the number of features present in the forest and the number of spline
        # it is possible that we have less feature than requested splines
        min_feat = min(len(self.feature_dict.keys()), self.n_splines)

        self.mif = self.feat_importance_method_map[self.feat_importance_method]()[:min_feat]

        # sample the thresholds
        self.sampled = self.sample_method_map[self.sample_method](self.feature_dict, self.sample_n)
        self.sampled = {key: np.unique(sampled_values) for key, sampled_values in self.sampled.items()}

        for feat_name in self.forest.feature_names():
            if feat_name not in self.sampled:
                self.sampled[feat_name] = np.array([])

        # Create the training dataset for the GAM
        self.X_synth_train = self._create_dataset(self.n_sample_gam)[self.forest.feature_names()]

        # only if a feature's set of thresholds are in a string format, then they are converted to a catgorical type
        # it's a requirement for lightgbm library
        for c in self.cat_feat:
            res = all(isinstance(n, str) for n in self.feature_dict[c])
            if res:
                self.X_synth_train[c] = self.X_synth_train[c].astype('category')
        self.y_synth_train = self.forest.predict(self.X_synth_train)

        # Test the synthetic dataset
        X_synth_test = self._create_dataset(self.n_sample_test)[self.forest.feature_names()]
        for c in self.cat_feat:
            res = all(isinstance(n, str) for n in self.feature_dict[c])
            if res:
                X_synth_test[c] = X_synth_test[c].astype('category')
        y_synth_test = self.forest.predict(X_synth_test)

        # compute features interaction if needed
        if self.n_inter_terms > 0 or self.fixed_feat_inter is not None:
            if self.fixed_feat_inter is not None:
                self.interactions = self.fixed_feat_inter
            else:
                self.interaction_importance_method()
                interaction_list = [(i, j, value)
                                    for i, row in enumerate(self.interaction_matrix)
                                    for j, value in enumerate(row)]
                interaction_list.sort(key=lambda x: x[2], reverse=True)
                self.interactions = [(i, j) for i, j, value in interaction_list[:self.n_inter_terms]]

        gam_forest, loss_res = self._create_gam(self.X_synth_train, self.y_synth_train, X_synth_test, y_synth_test)
        if self.verbose:
            print(f"GAM fitted from the forest, {loss_res=}")

        self.fitted = True
        self.gam = gam_forest
        self.loss_res = loss_res

        return gam_forest

    @staticmethod
    def auto_fit(forest_to_explain, min_rmse_increment=0.1, verbose=True, **params):
        """
        Fits multiple GamExplainer instances with variable number of splines and interaction terms.
        Returns a GamExplainer with fine-tuned parameters based on the best RMSE obtained. 

        :param forest: The forest of trees to be explained.
        :type forest: lgbm.Booster
        :param min_rmse_increment: A portion of RMSE to subtract.
        :type: float, optional(deafult=0.1)
        :param verbose: If set to true, the explainer prints messages during the explanation process.
        :type verbose: bool
        :param \*\*params: Extra parameters.
        :type \*\*params: parameter list
        :return: A new instance of GamExplainer 
        :rtype: GamExplainer
        """
        explainer = GamExplainer(n_spline_terms=1, **params)
        explainer.explain(forest_to_explain)
        rmse = explainer.loss_res  # substituted rmse with loss_res

        number_of_spline = 1
        for i in range(2, forest_to_explain.num_features()):
            explainer = GamExplainer(n_spline_terms=i, **params)
            explainer.explain(forest_to_explain)
            new_rmse = explainer.loss_res

            if new_rmse >= rmse * (1 - min_rmse_increment):
                break
            number_of_spline = i
            rmse = new_rmse
            if verbose:
                print(f"{rmse=}, {number_of_spline=}")

        number_of_interactions = 0

        for i in range(1, forest_to_explain.num_features() * (forest_to_explain.num_features() - 1)):
            explainer = GamExplainer(n_spline_terms=number_of_spline, n_inter_terms=i, **params)
            explainer.explain(forest_to_explain)
            new_rmse = explainer.loss_res

            if new_rmse >= rmse * (1 - min_rmse_increment):
                break
            rmse = new_rmse
            number_of_interactions = i
            if verbose:
                print(f"{rmse=}, {number_of_spline=}, {number_of_interactions}")

        return GamExplainer(n_spline_terms=number_of_spline, n_inter_terms=number_of_interactions, **params)
