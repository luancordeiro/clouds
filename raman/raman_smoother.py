import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def smooth_sliding_average(window):
    def smooth(y):
        return pd.Series(y).rolling(window).mean().to_numpy()

    return smooth


def smooth_ewm(window):
    def smooth(y):
        return pd.Series(y).ewm(window).mean().to_numpy()

    return smooth


def smooth_savgol(window, degree):
    def smooth(y):
        return savgol_filter(y, window, degree)

    return smooth


def smooth_gaussian_filter(sigma, mode="reflect", truncate=4):
    def smooth(y):
        return gaussian_filter1d(y, sigma, mode=mode, truncate=truncate)

    return smooth
