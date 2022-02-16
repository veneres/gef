import itertools
from collections import defaultdict

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

from gamexplainer.metaforest import MetaForest


def plot_thresholds_hist(explainer, feature_index, ax1=None, bins=None):
    """
    Plots a histogram of the thresholds.

    :param explainer: The explainer object whose thresholds' values have to be plotted.
    :type explainer: GamExplainer
    :param feature_index: The indexes of the features to be plotted.
    :type feature_index: int
    :param ax1: The axis on which to operate.
    :type ax1: matplotlib Axes
    :param bins: Integer or sequence or string. It is one the parameters of the *matplotlib.axes.Axes.hist* function.
    :type bins: object
    """
    plt.style.use('seaborn-paper')
    c1, c2 = sns.color_palette("Blues", 2)
    matplotlib_hist_params = {}
    if bins is not None:
        matplotlib_hist_params = {"bins": bins}
    if not explainer.fitted:
        raise Exception("No forest explained")

    if ax1 is None:
        fig, ax1 = plt.subplots()
    ax1.hist(explainer.feature_dict[feature_index], **matplotlib_hist_params, color=c1, alpha=0.7)
    ax2 = ax1.twinx()
    ax2.hist(explainer.sampled[feature_index], **matplotlib_hist_params, color=c2, alpha=0.7)

    ax1.set_title(f"Sampling method: {explainer.sample_method}")


def plot_feature_importance(explainer):
    """
    Plots the feature importances.

    :param explainer: The explainer object whose feature importances' values have to be plotted.
    :type explainer: GamExplainer
    """
    if not explainer.fitted:
        raise Exception("No forest explained")
    feats_name = [feat for feat, value in explainer.feature_importances]
    y_pos = np.arange(len(feats_name))
    feat_imp = [value for feat, value in explainer.feature_importances]

    plt.barh(y_pos, feat_imp, align='center', alpha=0.5)
    plt.yticks(y_pos, feats_name)
    plt.xlabel('Feature importance')
    plt.title('Feature importance')

    plt.show()


def plot_splines(explainer, real_fun=None):
    """
    Plots the splines of the GAM explainer.

    :param explainer: The explainer object whose splines' values have to be plotted.
    :type explainer: GamExplainer
    :param real_fun: The generating function.
    :type real_fun: function
    :return: The resulted axes.
    :rtype: matplotlib Axes
    """
    axes = []
    for i, term in enumerate(explainer.gam.terms):
        if term.isintercept:
            continue
        grid = explainer.gam.generate_X_grid(term=i, meshgrid=term.istensor)
        pdep, confi = explainer.gam.partial_dependence(term=i, X=grid, width=0.95, meshgrid=term.istensor)

        if term.istensor:
            print(term.feature)
            ax = plt.axes(projection='3d')
            ax.plot_surface(grid[0], grid[1], pdep, cmap=sns.color_palette("Blues", as_cmap=True))
        else:
            c1, c2, c3 = sns.color_palette("Blues", 3)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(grid[:, term.feature], pdep, c=c1)
            ax.plot(grid[:, term.feature], confi, c=c2, ls='--')

            if real_fun is not None and term.feature in real_fun:
                ax.plot(grid[:, term.feature], real_fun[term.feature](grid[:, term.feature]), c=c3)

        ax.set_title(repr(term))
        axes.append(ax)
    return axes


def plot_interactions_scores(explainer, **heatmap_params):
    """
    Plots the heatmap of the GAM explainer's interaction scores.
    
    :param explainer: The explainer object whose interaction scores' values have to be plotted.
    :type explainer: GamExplainer
    :param \*\*heatmap_params: All other parameters passed to *seaborn.heatmap*.
    :type \*\*heatmap_params: parameter list
    :return: Axes object with the heatmap.
    :rtype: matplotlib Axes
    """
    if not explainer.fitted:
        raise Exception("No forest explained")
    if explainer.interactions is None:
        raise Exception("Interaction importance not computed")
    # min-max scaling
    mat_range = np.max(explainer.interaction_matrix) - np.min(explainer.interaction_matrix)
    scaled_matrix = (explainer.interaction_matrix - np.min(explainer.interaction_matrix)) / mat_range

    mask = np.zeros_like(scaled_matrix)
    mask[np.triu_indices_from(mask)] = True
    mask = mask == False
    df = pd.DataFrame(scaled_matrix, columns=explainer.forest.feature_names(), index=explainer.forest.feature_names())
    ax = sns.heatmap(df, mask=mask, square=True, **heatmap_params)
    return ax


def np_where_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


# works only with one and two features
def _partial_dependence(forest: MetaForest, X: pd.DataFrame, feats:np.array, grid: [np.array]):
    if len(feats) <= 0 or len(feats) > 2:
        raise AttributeError(f"len(feats) must be 1 or 2, {len(feats)=}")

    if len(grid) == 1:
        xs = [(elem,) for elem in grid[0]]  # to be consistent with the case of two variables
    else:
        xs = itertools.product(grid[0], grid[1])

    pdp_res = {}
    for x in xs:  # for each value in the grid
        X_mod = X.copy()  # compute the average prediction from the training set X
        for i, coord in enumerate(x):  # for each feature to be changed
            X_mod[feats[i]] = coord
        pdp_res[x] = np.mean(forest.predict(X_mod))

    return pdp_res


def h_stat_all_pairs(forest: MetaForest, X: pd.DataFrame, feats: np.array, sample_size=100, random_state=42,
                     verbose=True):
    X = X.sample(sample_size, random_state=random_state)

    # Dictionary of features values, used to not repeat computations

    feat_val_dict = {}

    # Store occurrences for values of single features

    for feat in feats:
        feat_val_dict[feat] = defaultdict(int)
        values = X[feat]
        for value in values:
            feat_val_dict[feat][value] += 1

    all_pairs = list(itertools.combinations(feats, 2))

    # Store occurrences for values of pairs of features
    for feat1, feat2 in all_pairs:
        feat_val_dict[(feat1, feat2)] = defaultdict(int)
        for index, sample in X.iterrows():
            feat1_value = sample[feat1]
            feat2_value = sample[feat2]
            feat_val_dict[(feat1, feat2)][(feat1_value, feat2_value)] += 1

    # create grids

    grids = {feat: [np.array(list(feat_val_dict[feat].keys()))] for feat in feats}

    for feat1, feat2 in all_pairs:
        grid_f1 = np.array(list(feat_val_dict[feat1].keys()))
        grid_f2 = np.array(list(feat_val_dict[feat2].keys()))
        grids[(feat1, feat2)] = [grid_f1, grid_f2]

    # Compute all the PD needed.
    pdps = {}

    it_feats = feats
    if verbose:
        print("H statistic computation starts...")
        print("Partial dependence for a single feature")
        it_feats = tqdm(feats)

    for feat in it_feats:
        pdps[feat] = _partial_dependence(forest, X, [feat], grids[feat])

    it_all_pairs = all_pairs
    if verbose:
        print("Partial dependence for all pair of features")

    for feat1, feat2 in it_all_pairs:
        pdps[(feat1, feat2)] = _partial_dependence(forest, X, [feat1, feat2], grids[(feat1, feat2)])

    # centering PDPs
    pdp_singles_avg = {feats: np.mean(list(pdp.values())) for feats, pdp in pdps.items()}

    pdps_centered = {}
    for feats, pdp in pdps.items():
        pdps_centered[feats] = {}
        for x, value in pdp.items():
            pdps_centered[feats][x] = value - pdp_singles_avg[feats]

    dict_num = defaultdict(int)
    dict_den = defaultdict(int)
    for (feat1, feat2) in all_pairs:
        for (feat1_value, feat2_value), count in feat_val_dict[(feat1, feat2)].items():
            joint_pdp = pdps_centered[(feat1, feat2)][(feat1_value, feat2_value)]
            f1_pdp = pdps_centered[feat1][(feat1_value,)]
            f2_pdp = pdps_centered[feat2][(feat2_value,)]
            dict_num[(feat1, feat2)] += np.square(joint_pdp - f1_pdp - f2_pdp) * count
            dict_den[(feat1, feat2)] += np.square(joint_pdp) * count
    res = {pair: np.sqrt(num / dict_den[pair]) for pair, num in dict_num.items()}
    return res


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


# range 0.1 = 10 % of the entire range from the training set

def _generate_X_grid_centered(gam, term, n=100, meshgrid=False, range_perc=100, center=None):
    """
    Create a nice grid of X data. Updated and modified from Pygam
    https://github.com/dswah/pyGAM/blob/master/pygam/pygam.py#L1398

    array is sorted by feature and uniformly spaced,
    so the marginal and joint distributions are likely wrong

    if term is >= 0, we generate n samples per feature,
    which results in n^deg samples,
    where deg is the degree of the interaction of the term

    Parameters
    ----------
    term : int,
        Which term to process.

    n : int, optional
        number of data points to create

    meshgrid : bool, optional
        Whether to return a meshgrid (useful for 3d plotting)
        or a feature matrix (useful for inference like partial predictions)

    Returns
    -------
    if meshgrid is False:
        np.array of shape (n, n_features)
        where m is the number of
        (sub)terms in the requested (tensor)term.
    else:
        tuple of len m,
        where m is the number of (sub)terms in the requested
        (tensor)term.

        each element in the tuple contains a np.ndarray of size (n)^m

    Raises
    ------
    ValueError :
        If the term requested is an intercept
        since it does not make sense to process the intercept term.
    """
    if not gam._is_fitted:
        raise AttributeError('GAM has not been fitted. Call fit first.')

    # cant do Intercept
    if gam.terms[term].isintercept:
        raise ValueError('cannot create grid for intercept term')

    # process each subterm in a TensorTerm
    if gam.terms[term].istensor:
        Xs = []
        for term_ in gam.terms[term]:
            Xs.append(np.linspace(term_.edge_knots_[0],
                                  term_.edge_knots_[1],
                                  num=n))

        Xs = np.meshgrid(*Xs, indexing='ij')
        if meshgrid:
            return tuple(Xs)
        else:
            return gam._flatten_mesh(Xs, term=term)

    # all other Terms
    elif hasattr(gam.terms[term], 'edge_knots_'):
        x_min = gam.terms[term].edge_knots_[0]
        x_max = gam.terms[term].edge_knots_[1]
        if center is None:
            center = (x_min + x_max) / 2
        x_span = (x_max - x_min) * (range_perc / 100)
        x = np.linspace(max(center - x_span / 2, x_min),
                        min(center + x_span / 2, x_max),
                        num=n)

        if meshgrid:
            return (x,)

        # fill in feature matrix with only relevant features for this term
        X = np.zeros((n, gam.statistics_['m_features']))
        X[:, gam.terms[term].feature] = x
        if getattr(gam.terms[term], 'by', None) is not None:
            X[:, gam.terms[term].by] = 1.

        return X

    # don't know what to do here
    else:
        raise TypeError('Unexpected term type: {}'.format(gam.terms[term]))


def plot_local(gam, term_names, term_id, X_train, sample_id, range_perc=100, pdp_ax=None):
    if pdp_ax is None:
        f, (pdp_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
    term = gam.terms[term_id]
    sample = X_train[sample_id].reshape(1, -1)
    c1, c2, c3 = sns.color_palette(n_colors=3)

    sample_feat_value = sample.ravel()[term.feature]
    # Local PDP
    XX = _generate_X_grid_centered(gam, n=1000, term=term_id, center=sample_feat_value, range_perc=range_perc)
    y_pdep, confi = gam.partial_dependence(term=term_id, X=XX, width=0.95)
    x_pdep = XX[:, term.feature]
    pdp_ax.plot(x_pdep, y_pdep, color=c1)

    # Compute average effect
    XX_global = gam.generate_X_grid(term=term_id, n=1000)
    pdep_global, _ = gam.partial_dependence(term=term_id, X=XX_global, width=0.95)

    avg_effect = np.mean(pdep_global)
    max_y = np.max(pdep_global)
    min_y = np.min(pdep_global)
    diff_avg = y_pdep - avg_effect  # difference from the average
    abs_diff_avg = np.abs(diff_avg)
    max_diff_max = np.abs(max_y - avg_effect)
    max_diff_min = np.abs(min_y - avg_effect)
    pdp_less_avg = y_pdep < avg_effect
    pdp_grt_avg = y_pdep > avg_effect
    color_map_min = sns.diverging_palette(10, 240, as_cmap=True)

    shading_n = 100

    for i in range(shading_n):
        pdp_ax.fill_between(x_pdep, y_pdep, avg_effect,
                            where=np.logical_and(pdp_grt_avg, abs_diff_avg > i / shading_n * max_diff_max),
                            color=color_map_min(0.5 + i / shading_n / 2))
        pdp_ax.fill_between(x_pdep, y_pdep, avg_effect,
                            where=np.logical_and(pdp_less_avg, abs_diff_avg > i / shading_n * max_diff_min),
                            color=color_map_min(0.5 - i / shading_n / 2))

    x_point = sample[0, term.feature]  # col vector
    y_point = gam.partial_dependence(term=term_id, X=sample)

    pdp_ax.set_title(term_names[term.feature])

    color_sample = "grey"
    pdp_ax.scatter(x_point, y_point, label="Sample under investigation", color=color_sample, zorder=3)
    pdp_ax.vlines(x_point, pdp_ax.get_ylim()[0], y_point, linestyle="dashed", color=color_sample)
    pdp_ax.hlines(y_point, pdp_ax.get_xlim()[0], x_point, linestyle="dashed", color=color_sample)


def n_plot_terms(gam):
    count = 0
    for i, term in enumerate(gam.terms):
        if term.isintercept or term.istensor:
            continue
        count += 1
    return count


def plot_local_all_terms(gam, term_names, X_train, sample_id, range_perc=100, figsize=(7, 8)):
    n_terms = n_plot_terms(gam)
    fig, axs = plt.subplots(nrows=n_terms, ncols=1, figsize=figsize, tight_layout=True)

    for i, term in enumerate(gam.terms):
        if term.isintercept or term.istensor:
            continue
        # shared axis on right col
        plot_local(gam, term_names, i, X_train, sample_id, range_perc=range_perc, pdp_ax=axs[i])
