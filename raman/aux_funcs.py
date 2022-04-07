from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import chisquare


def mean_squared_deviation(x: np.array, s: np.array) -> float:
    return np.sqrt(np.mean(((x - s) / s) ** 2))


def mean_deviation(x: np.array, s: np.array) -> np.ndarray:
    return np.mean((x - s) / s)


def diff_(y: np.array, x: np.array, window: int = 1, *args, **kwargs):
    diff = np.gradient(y[::window]) / np.gradient(x[::window])

    new_diff = []
    for ele in diff:
        for _ in range(window):
            new_diff.append(ele)

            if len(new_diff) == len(x):
                return np.array(new_diff)


def diff_linear_regression(y: np.array, x: np.array, window: int = 5, weights: np.array = None):
    def fit(init, final):
        y_fit = y[init: final].reshape(-1, 1)
        x_fit = x[init: final].reshape(-1, 1)

        if weights is None:
            linear_regession = LinearRegression().fit(x_fit, y_fit)
        else:
            weight_fit = weights[init: final]
            linear_regession = LinearRegression().fit(x_fit, y_fit, sample_weight=weight_fit)

        return linear_regession.coef_[0][0]

    if window % 2 == 0:
        raise ValueError("window must be odd.")

    win = window // 2
    diff_y = []
    for i in range(win, len(y) - win - 10 - 1):
        diff_y.append(fit(i - win, i + win + 1))
    #        if (i % 20 == 0) & (win <= window // 2 + 10):
    #            win += 2

    for i in range(window // 2):
        # diff_y.insert(i, fit(None, i + window // 2))
        diff_y.insert(0, diff_y[0])

    while len(diff_y) != len(y):
        diff_y += [diff_y[-1]]

    return np.array(diff_y)


def diff_polyfit(y: np.array, x: np.array, window: int = 5, weights: np.array = None):
    def fit(init, final):
        y_fit = y[init: final]
        x_fit = x[init: final]

        return np.poly1d(np.polyfit(x_fit, y_fit, 3))

    if window % 2 == 0:
        raise ValueError("window must be odd.")

    win = window // 2
    diff_y = []
    for i in range(win, len(y) - win - 10 - 1):
        a, b, c, _ = fit(i - win, i + win + 1)
        diff_y.append(3 * a * x[i] ** 2 + 2 * b * x[i] + c)

    for i in range(window // 2):
        diff_y.insert(0, diff_y[0])

    while len(diff_y) != len(y):
        diff_y += [diff_y[-1]]

    print(np.array(diff_y).shape)

    return np.array(diff_y)


def diff_chi_squared(y: np.array, x: np.array, window=5):
    win = window // 2

    diff_y = []
    params = []
    for i in range(win, len(y) - win - 1):
        y_fit = y[i - win: i + win + 1]
        x_fit = x[i - win: i + win + 1]

        poly_fits = [np.poly1d(np.polyfit(y=y_fit, x=x_fit, deg=i)) for i in range(1, 4)]

        chi_square = np.array([chisquare(poly(x_fit), y_fit)[0] for poly in poly_fits])

        chosen_ind = (np.abs(chi_square - 0.5)).argmin()

        params.append(chosen_ind)

        chosen_func = poly_fits[chosen_ind]

        diff_y.append((chosen_func(x[i] + 1e6) - chosen_func(x[i])) / 1e6)

    for i in range(win):
        diff_y.insert(0, diff_y[0])

    while len(diff_y) != len(y):
        diff_y += [diff_y[-1]]

    return np.array(diff_y), params


def diff_chi_squared_2(y: np.array, x: np.array, params, window=5):
    win = window // 2
    diff_y = []
    for i, param in zip(range(win, len(y) - win - 1), params):
        y_fit = y[i - win: i + win + 1]
        x_fit = x[i - win: i + win + 1]

        func = np.poly1d(np.polyfit(y=y_fit, x=x_fit, deg=param + 1))

        diff_y.append((func(x[i] + 1e6) - func(x[i])) / 1e6)

    for i in range(win):
        diff_y.insert(0, diff_y[0])

    while len(diff_y) != len(y):
        diff_y += [diff_y[-1]]

    return np.array(diff_y)

