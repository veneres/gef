from collections.abc import Callable
from typing import Optional, Any

import numpy as np
import pandas
import pandas as pd
import os

CD = os.path.dirname(os.path.realpath(__file__))

# Datasets paths

CONCRETE_DATASET_PATH = os.path.join(CD, "datasets_csv", "concrete.csv")
YPMSD_DATASET_PATH = os.path.join(CD, "datasets_csv", "YearPredictionMSD.csv")
BREAST_CANCER_DATASET_PATH = os.path.join(CD, "datasets_csv", "wdbc.csv")
BIKE_SHARING_DATASET_PATH = os.path.join(CD, "datasets_csv", "bike_sharing.csv")
HOUSE_PRICES_DATASET_PATH = os.path.join(CD, "datasets_csv", "housing.csv")
FOREST_FIRE_DATASET_PATH = os.path.join(CD, "datasets_csv", "forestfires.csv")
PARKINSON_DATASET_PATH = os.path.join(CD, "datasets_csv", "parkinson.csv")
CREDIT_CARD_DATASET_PATH = os.path.join(CD, "datasets_csv", "creditcard.csv")
ABALONE_DATASET_PATH = os.path.join(CD, "datasets_csv", "abalone.csv")
ADULT_DATASET_PATH = os.path.join(CD, "datasets_csv", "adult.csv")
AUDIT_DATASET_PATH = os.path.join(CD, "datasets_csv", "audit.csv")
BANK_DATASET_PATH = os.path.join(CD, "datasets_csv", "bank.csv")
GARMENTS_WORKER_PRODUCTIVITY_DATASET_PATH = os.path.join(CD, "datasets_csv", "garments_worker_productivity.csv")
VOTES_DATASET_PATH = os.path.join(CD, "datasets_csv", "votes.csv")


def dataset_from_fun(n_sample: int,
                     n_features: int,
                     fun: Callable,
                     dist_gen: Optional[np.random.Generator] = None,
                     min_value: float = 0,
                     max_value: float = 1,
                     random_state: Any = 42,
                     **kwargs) -> pd.DataFrame:
    """
    Creates a synthetic dataset given a sample size, the number of features to be used and the function to create each
    label.

    :param n_sample: The number of samples to be generated.
    :type n_sample: int
    :param n_features: The number of features to be generated.
    :type n_features: int
    :param fun: The function to be used to create the label.
    :type fun: function
    :param dist_gen: The distribution generator to use to draw the samples. If None, numpy.random.uniform will be used.
    :type dist_gen: Generator
    :param min_value: Minimum value to draw from the distribution.
    :type min_value: int
    :param max_value: Maximum value to draw from the distribution.
    :type max_value: int
    :param random_state: random state used from the numpy random generator
    :type random_state: int or anything suitable as seed for a numpy random generator
    :return: A pandas dataframe of shape (n_sample, n_features + 1).
    :rtype: pd.DataFrame
    """
    if n_sample <= 0 or n_features <= 0:
        raise ValueError("n_sample and n_features must be positive")
    if dist_gen is None:
        dist_gen = np.random.default_rng(seed=random_state).uniform

    samples = dist_gen(min_value, max_value, size=(n_sample, n_features))

    label = np.atleast_2d(np.apply_along_axis(fun, 1, samples, **kwargs)).T
    cols = [f"x_{i}" for i in range(n_features)] + ["y"]
    res = pd.DataFrame(np.append(samples, label, axis=1), columns=cols)

    return res


# real-world datasets as pandas dataframe

concrete = pandas.read_csv(CONCRETE_DATASET_PATH, sep=";")
breast_cancer = pandas.read_csv(BREAST_CANCER_DATASET_PATH, sep=",")
bike_sharing = pandas.read_csv(BIKE_SHARING_DATASET_PATH, sep=",")
house_prices = pandas.read_csv(HOUSE_PRICES_DATASET_PATH, delim_whitespace=True)
forest_fire = pandas.read_csv(FOREST_FIRE_DATASET_PATH, sep=",")
parkinson = pandas.read_csv(PARKINSON_DATASET_PATH, sep=",")
credit_card = pandas.read_csv(CREDIT_CARD_DATASET_PATH, sep=",")
abalone = pandas.read_csv(ABALONE_DATASET_PATH, sep=",")
adult = pandas.read_csv(ADULT_DATASET_PATH, sep=",")
audit = pandas.read_csv(AUDIT_DATASET_PATH, sep=",")
bank = pandas.read_csv(BANK_DATASET_PATH, sep=",")
garments_worker_productivity = pandas.read_csv(GARMENTS_WORKER_PRODUCTIVITY_DATASET_PATH, sep=",")
votes = pandas.read_csv(VOTES_DATASET_PATH, sep=",")
