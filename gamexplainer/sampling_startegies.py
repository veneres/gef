import typing

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def unpack(list_: list):
    """
    Checks and converts the string format of categorical data into integers.

    :param list_: A set of threshold values to be checked.
    :type list_: list
    :return: A set of correctly converted values, or the input list if there is no need for a conversion.
    :rtype: list
    """
    final = []
    res = all(isinstance(n, str) for n in list_)
    if res:
        for n in list_:
            if not n or not n.strip():
                continue
            elif n.find('||'):
                a_list = n.split('||')
                map_object = list(map(int, a_list))
            else:
                map_object = list(map(int, n))
            final += map_object
        return final

    return list_


def is_category(thresholds: list, max_threshold=10):
    """
    Checks if a set of thresholds could belong to a categorical feature.

    :param thresholds: A set of threshold values to be checked.
    :type thresholds: list

    :param max_threshold: Maximum number of different modalities to consider a feature as a categorical.
    :type max_threshold: int

    :return: Returns true if the thresholds are strings in some particular format or if 
        the thresholds are integers and the number of distinct values are limited by the given max_threshold.
    :rtype: bool
    """
    res = all(isinstance(n, str) for n in thresholds)
    # if the thresholds are strings in some particular formats
    if res:
        return True
    # if the thresholds are integers and the number of distinct values are limited to a given threshold
    if pd.Series(thresholds).nunique() <= max_threshold:
        return True
    return False


def split_value(thresholds: list):
    """
    Split some threshold values if there are middle values.

    :param thresholds: A dictionary of threshold values to be split.
    :type thresholds: dict
    :return: A set of correctly split values.
    :rtype: np.array
    """
    values = []
    for t in thresholds:
        values.append(int(t))
        values.append(int(t + 1))
    values = np.unique(values)
    return values


def categories_sampling(thresholds: list):
    """
       Return a set of values sampled from a categorical feature

       :param thresholds: A dictionary of threshold values to be split.
       :type thresholds: dict
       :return: A set of values sampled from a categorical feature.
       :rtype: np.array
    """
    res = all(isinstance(n, str) for n in thresholds)
    if res:
        values = unpack(thresholds)
    else:
        values = split_value(thresholds)
    return values


def rnd_sampling(feature_dict: dict,
                 sample_size: int,
                 epsilon_perc=0.1,
                 random_state=42,
                 categories: typing.Optional[list] = None):
    """
    Creates subsets of thresholds of specified size, for each feature, drawn uniformly at random
    from a specific range. If a feature is identified as a category then its 
    thresholds are added as they are, without adopting particular sampling techniques.

    :param feature_dict: A dictionary of features, each with its own set of thresholds.
    :type feature_dict: dict
    :param sample_size: The number of instances of the dataset to be gererated.
    :type sample_size: int
    :param epsilon_perc: Allows to extend the range of features thresholds with the specified percentage.
        By default epsilon is 10% of the range of the features thresholds.
    :type epsilon_perc: int
    :param random_state: Random seed for a numpy function in order to set the seed of the random number generator.
    :type random_state: int
    :return: A dictionary of features, each with a numpy array of randomly uniform thresholds of length sample_size.
    :rtype: dict
    """
    np.random.seed(random_state)
    sampled = {}
    for feat, thresholds in feature_dict.items():
        # by default epsilon is 10% of the range of the features thresholds
        # if the feature is identified as a category then its thresholds are added as they are
        if (is_category(thresholds) and categories is None) or (categories is not None and feat in categories):
            sampled[feat] = categories_sampling(thresholds)
        else:
            epsilon = (np.max(thresholds) - np.min(thresholds)) * epsilon_perc
            low = np.min(thresholds) - epsilon
            high = np.max(thresholds) + epsilon
            sampled[feat] = np.random.uniform(low=low, high=high, size=sample_size)
    return sampled


def all_sampling(feature_dict: dict, sample_size: int, epsilon_perc=0.05, categories: typing.Optional[list] = None):
    """
    Given a sorted list of feature thresholds in ascending order, creates a sampling domain of values for each feature, 
    including the minimum, the maximum and other middle values. If a feature is identified as a category then its 
    thresholds are added as they are, without adopting particular sampling techniques.

    :param feature_dict: A dictionary of features, each with its own set of threshold values.
    :type feature_dict: dict
    :param sample_size: A dummy variable to have a consistent number of positional parameters (three)
        between all the functions.
    :type sample_size: int
    :param epsilon_perc: A percentage for a measure (epsilon) which allows to extend the sampling feature domain.
        By default epsilon is 5% of the range of the feature thresholds values.
    :type epsilon_perc: int
    :return: A dictionary of features, each with a numpy array including the minimum and the maximum threshold values, 
        and other calculated middle values.
    :rtype: dict
    """
    sampled = {}
    for feat, thresholds in feature_dict.items():
        if (is_category(thresholds) and categories is None) or (categories is not None and feat in categories):
            sampled[feat] = categories_sampling(thresholds)
        else:
            min_val = np.min(thresholds)
            max_val = np.max(thresholds)
            epsilon = (max_val - min_val) * epsilon_perc
            min_val -= epsilon
            max_val += epsilon
            thresholds.sort()
            mid_points = [(thresholds[i] + thresholds[i + 1]) / 2 for i in range(len(thresholds) - 1)]
            sampled[feat] = np.array([min_val] + mid_points + [max_val])
    return sampled


def equal_dist_sampling(feature_dict, sample_size, epsilon_perc=0.05, categories: typing.Optional[list] = None):
    """
    Returns subintervals of values, all of the same length, within a given range. 
    The start of the subinterval is the minimum threshold and the stop is the maximum threshold value. 
    The subinterval includes both the minimum and the maximum threshold values.
    If a feature is identified as a category then its 
    thresholds are added as they are, without adopting particular sampling techniques.

    :param feature_dict: A dictionary of features, each with its own set of threshold values.
    :type feature_dict: dict
    :param sample_size: The number of instances. Here it is used for spacing between values, in order to
        get a set the correct size, namely the sample_size. The distance between two adjacent values will be 
        (max_threshold - min_threshold) / (sample_size - 1). The maximum value of the set cannot be greater or 
        equal to the maximum threshold value.
    :type sample_size: int
    :param epsilon_perc: A percentage for a measure allowing to extend the sampling feature domain.
    :type epsilon_perc: int
    :return: A dictionary of features, each having a numpy array of threshold values. All of these arrays have equal size and 
        equal distance between adjacent values.
    :rtype: dict
    """
    sampled = {}
    for feat, thresholds in feature_dict.items():
        if len(feature_dict[feat]) > 0:
            if (is_category(thresholds) and categories is None) or (categories is not None and feat in categories):
                equal = categories_sampling(thresholds)
            else:
                min_val = np.min(thresholds)
                max_val = np.max(thresholds)
                epsilon = (max_val - min_val) * epsilon_perc
                min_val -= epsilon
                max_val += epsilon
                if min_val != max_val:
                    equal = np.arange(min_val, max_val, (max_val - min_val) / (sample_size - 1))
                else:
                    equal = np.array([])
                equal = np.append(equal, [max_val])
        else:
            equal = np.array([])
        sampled[feat] = equal

    return sampled


def quantile_sampling(feature_dict, sample_size, categories: typing.Optional[list] = None):
    """
    Returns subsets of quantile values, all of the same length, within a given range.
    There is a sequence of quantiles to compute, which must be between 0 and 1 inclusive.
    If a feature is identified as a category then its 
    thresholds are added as they are, without adopting particular sampling techniques.

    :param feature_dict: A dictionary of features, each with its own set of threshold values.
    :type feature_dict: dict
    :param sample_size: The number of instances. Here it is used for defining the distance between two adjacent
        quantiles.
    :type sample_size: int
    :return: A dictionary of features, each having a numpy array of quantiles for each threshold. All of these arrays have 
        equal size and equal distance between adjacent values.
    :rtype: dict
    """
    sampled = {}
    for feat, thresholds in feature_dict.items():
        if (is_category(thresholds) and categories is None) or (categories is not None and feat in categories):
            sampled[feat] = categories_sampling(thresholds)
        else:
            values = np.array(thresholds)
            if sample_size < len(values):
                quantiles = np.arange(0, 1, 1 / sample_size)
                sampled[feat] = np.quantile(values, quantiles, interpolation="nearest")
            else:
                sampled[feat] = values
    return sampled


def equi_size_sampling(feature_dict: dict, sample_size: int, categories: typing.Optional[list] = None):
    """

    :param feature_dict: A dictionary of features, each with its own set of threshold values.
    :type feature_dict: dict
    :param sample_size: The number of values to be sampled.
    :type sample_size: int
    :return: A dictionary representing the new sampling domain, where at each key (feature) its associated a list of
        sampled values.
    :rtype: dict
    """
    sampled = {}
    for feat, thresholds in feature_dict.items():
        if (is_category(thresholds) and categories is None) or (categories is not None and feat in categories):
            sampled[feat] = categories_sampling(thresholds)
        else:
            n = 0
            points = []
            step = int(len(thresholds) / sample_size)
            if step > 0:
                data = np.sort(np.array(thresholds))
                for j in range(0, sample_size - 1):
                    points.append(np.mean(data[n:n + step]))
                    n = n + step
                if data[n:].shape[0] > 0:
                    points.append(np.mean(data[n:]))
            else:
                points = thresholds
            sampled[feat] = np.array(points)
    return sampled


def kmeans_sampling(feature_dict, sample_size, categories: typing.Optional[list] = None):
    """
    Returns subsets of centroids of the clusters resulting from the use of the k-means algorithm. 
    The number of clusters is taken as the minimum between the sample_size and the length of the set of unique threshold values.
    If a feature is identified as a category then its 
    thresholds are added as they are, without adopting particular sampling techniques.

    :param feature_dict: A dictionary of features, each with its own set of threshold values.
    :type feature_dict: dict
    :param sample_size: The number of instances. Here it is used for defining the number of clusters to be formed.
    :type sample_size: int
    :return: A dictionary of features, each having a numpy array of centroids of the clusters that were formed. 
    :rtype: dict
    """
    sampled = {}

    for feat, thresholds in feature_dict.items():
        if len(thresholds) > 0:
            if (is_category(thresholds) and categories is None) or (categories is not None and feat in categories):
                sampled[feat] = categories_sampling(thresholds)
            else:
                data = np.sort(np.array(thresholds)).reshape(-1, 1)
                kmeans = KMeans(n_clusters=min(sample_size, len(set(thresholds))), random_state=0).fit(data)
                centers = np.sort(np.array(kmeans.cluster_centers_.T[0]))
                sampled[feat] = centers
    return sampled
