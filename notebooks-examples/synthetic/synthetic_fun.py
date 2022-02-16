import numpy as np


def base_fun(key):
    if key == 0:
        return lambda x: x
    if key == 1:
        return lambda x: np.sin(x * 20)
    if key == 2:
        return lambda x: np.exp((x - 0.5) * 50) / (np.exp((x - 0.5) * 50) + 1)
    if key == 3:
        return lambda x: (np.arctan(x * 10) + np.sin(x * 10)) / 2
    if key == 4:
        return lambda x: 2.0 / (x + 1)
    raise AttributeError("Key not associated to any function")


def base_fun_inter(x, y):
    return 2 * (np.sqrt(2 * np.pi)) * np.exp(-1 * (((x - 0.5) ** 2 + (y - 0.5) ** 2) / 2))


def fun_without_interaction(x: np.array, rnd_gen: np.random.Generator):
    r"""
        TODO: fix the description
        Simple function with interaction defined as follows:

        .. math::
            f(\bm{x}) & =  \bm{x}_1 - \sin\left(20\bm{x}_1\right) \\
            & + \frac{\exp\left(50(\bm{x}_2 -0.5)\right)}{\exp\left(50(\bm{x}_2 -0.5)\right) + 1} \\
            & + \frac{\arctan\left(10\bm{x}_3\right)- \sin\left(10\bm{x}_3\right)}{2} \\
            & + 30 \exp \left(-\frac{1}{\sqrt{2 \pi}}\frac{\bm{x}_4^2  + \bm{x}_5^2}{2}\right)\\
            &  + \frac{2}{\bm{x}_6 +1} + \mathcal{N}(0,0.1^2)

        where :math:`\mathcal{N}(0,0.1^2)` is the normally distributed noise from the ``rnd_gen``.

        :param x: A numpy array of 6 elements.
        :type x: np.array
        :param rnd_gen: the numpy random number generator to be used
        :type rnd_gen: numpy.random.Generator
        :return: The value representing :math:`f(\bm{x})`.
        :rtype: float
        """
    noise = rnd_gen.normal(0, 0.1, size=1)[0]
    no_inter = base_fun(0)(x[0]) + base_fun(1)(x[1]) + base_fun(2)(x[2]) + base_fun(3)(x[3]) + base_fun(4)(x[4])

    return no_inter + noise


def fun_interaction(x: np.array, rnd_gen: np.random.Generator, real_interactions: [tuple]):
    r"""
        TODO: fix the description
        Simple function with interaction defined as follows:

        .. math::
            f(\bm{x}) & =  \bm{x}_1 - \sin\left(20\bm{x}_1\right) \\
            & + \frac{\exp\left(50(\bm{x}_2 -0.5)\right)}{\exp\left(50(\bm{x}_2 -0.5)\right) + 1} \\
            & + \frac{\arctan\left(10\bm{x}_3\right)- \sin\left(10\bm{x}_3\right)}{2} \\
            & + 30 \exp \left(-\frac{1}{\sqrt{2 \pi}}\frac{\bm{x}_4^2  + \bm{x}_5^2}{2}\right)\\
            &  + \frac{2}{\bm{x}_6 +1} + \mathcal{N}(0,0.1^2)

        where :math:`\mathcal{N}(0,0.1^2)` is the normally distributed noise from the ``rnd_gen``.

        :param x: A numpy array of 6 elements.
        :type x: np.array
        :param rnd_gen: the numpy random number generator to be used
        :type rnd_gen: numpy.random.Generator
        :return: The value representing :math:`f(\bm{x})`.
        :rtype: float
        """
    inter = 0
    for i, j in real_interactions:
        inter += base_fun_inter(x[i], x[j])

    return fun_without_interaction(x, rnd_gen) + inter
