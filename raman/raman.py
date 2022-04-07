import numpy as np
import pandas as pd
from alpha_beta_mol import AlphaBetaMolecular
from scipy.integrate import cumtrapz, trapz

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from raman_data import RamanData
from aux_funcs import *

plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams.update({'font.family': 'serif', 'font.size': 14, 'font.weight': 'light'})


class Raman:
    """
    Input:
    z                - Altitude                                          [m]
    elastic_signal   -
    inelastic_signal -
    lidar_wavelength - the wavelength of the lidar                       [m]
    raman_wavelength - the wavelength of the inelastic backscatter foton [m]
    angstrom_coeff   - dependence of                                     [u.a]
    p_air            - pressure                                          [Pa]
    t_air            - temperature                                       [K]
    co2ppmv          - co2 concentration                                 [ppmv]
    diff_strategy    - differenciation strategy for the calculation of aerosol extinction

    """
    _alpha = dict()
    _beta = dict()
    _diff_window = 5
    _diff_strategy = diff_linear_regression

    def __init__(self,
                 z: np.array,
                 elastic_signal: np.array,
                 inelastic_signal: np.array,
                 lidar_wavelength: float,
                 raman_wavelength: float,
                 angstrom_coeff: float,
                 p_air: np.array,
                 t_air: np.array,
                 ref: int = 12000,
                 beta_ref: int = None,
                 co2ppmv: int = 392,
                 df_solutions: pd.DataFrame = None):

        self.raman_data = RamanData(z, elastic_signal, inelastic_signal, p_air, t_air)
        self.lidar_wavelength = lidar_wavelength
        self.raman_wavelength = raman_wavelength
        self.angstrom_coeff = angstrom_coeff
        self.co2ppmv = co2ppmv
        self.z_ref = ref
        self.beta_ref = beta_ref

        self.df_solutions = df_solutions

    def get_alpha(self) -> dict:
        return self._alpha.copy()

    def get_beta(self) -> dict:
        return self._beta.copy()

    def get_lidar_ratio(self) -> np.array:
        return self._alpha["elastic_aer"] / self._beta["elastic_aer"]

    def set_beta_ref(self, beta_ref):
        self.beta_ref = beta_ref
        return self

    def set_diff_strategy(self, diff_strategy):
        self._diff_strategy = diff_strategy
        return self

    def _alpha_beta_molecular(self, co2ppmv) -> None:
        elastic_alpha_beta = AlphaBetaMolecular(self.p_air,
                                                self.t_air,
                                                self.lidar_wavelength,
                                                co2ppmv)

        inelastic_alpha_beta = AlphaBetaMolecular(self.p_air,
                                                  self.t_air,
                                                  self.raman_wavelength,
                                                  co2ppmv)

        self._alpha['elastic_mol'], self._beta['elastic_mol'], _ = elastic_alpha_beta.get_params()
        self._alpha['inelastic_mol'], self._beta['inelastic_mol'], _ = inelastic_alpha_beta.get_params()

    def _raman_scatterer_numerical_density(self) -> np.array:
        atm_numerical_density = self.p_air / (1.380649e-23 * self.t_air)
        return atm_numerical_density * 78.08e-2

    def _ranged_corrected_signal(self) -> np.array:
        return self.inelastic_signal * self.z ** 2

    def _diff(self, y, x) -> np.array:
        y = y if self._smooth_strategy is None else self._smooth_strategy(y)

        print(self._diff_window)
        return self._diff_strategy(y, x, window=self._diff_window)

    def _alpha_elastic_aer(self) -> np.array:
        diff_num_signal = self._diff(np.log(self._raman_scatterer_numerical_density() /
                                            self._ranged_corrected_signal()),
                                     self.z)

        return (diff_num_signal - self._alpha['elastic_mol'] - self._alpha['inelastic_mol']) / \
               (1 + (self.lidar_wavelength / self.raman_wavelength) ** self.angstrom_coeff)

    def _alpha_elastic_total(self) -> np.array:
        return self._alpha["elastic_aer"] + self._alpha["elastic_mol"]

    def _alpha_inelastic_total(self) -> np.array:
        return self._alpha["inelastic_aer"] + self._alpha["inelastic_mol"]

    def _beta_elastic_total(self, smoother) -> np.array:
        scatterer_numerical_density = self._raman_scatterer_numerical_density()

        elastic_signal = self.elastic_signal if smoother is None else smoother(self.elastic_signal)

        inelastic_signal = self.inelastic_signal if smoother is None else smoother(self.inelastic_signal)

        signal_ratio = ((inelastic_signal[self._ref] * elastic_signal * scatterer_numerical_density) /
                        (elastic_signal[self._ref] * inelastic_signal *
                         scatterer_numerical_density[self._ref]))

        attenuation_ratio = (np.exp(-cumtrapz(x=self.z, y=self._alpha_inelastic_total(), initial=0) +
                                    trapz(x=self.z[:self._ref], y=self._alpha_inelastic_total()[:self._ref])) /
                             np.exp(-cumtrapz(x=self.z, y=self._alpha_elastic_total(), initial=0) +
                                    trapz(x=self.z[:self._ref], y=self._alpha_elastic_total()[:self._ref])))

        beta_ref = self._beta["elastic_mol"][self._ref] if self.beta_ref is None else self.beta_ref

        return beta_ref * signal_ratio * attenuation_ratio

    def _data_preprocessing(self, vertical_window=None, diff_window=5, smooth_strategy=None) -> None:
        (self.z,
         self.elastic_signal,
         self.inelastic_signal,
         self.p_air,
         self.t_air) = self.raman_data.data_binning(vertical_window)

        self._diff_window = diff_window

        self._smooth_strategy = smooth_strategy

        self._alpha_beta_molecular(self.co2ppmv)

        self._ref = np.where(abs(self.z - self.z_ref) == min(abs(self.z - self.z_ref)))[0][0]

    def fit(self, vertical_average_window=None, smooth_diff_strategy=None, profile_smoother=None, diff_window=5):

        self._data_preprocessing(vertical_average_window, diff_window, smooth_diff_strategy)

        self._alpha["elastic_aer"] = self._alpha_elastic_aer()

        self._alpha["inelastic_aer"] = self._alpha["elastic_aer"] / (
                self.raman_wavelength / self.lidar_wavelength) ** self.angstrom_coeff

        self._beta["elastic_aer"] = self._beta_elastic_total(profile_smoother) - self._beta["elastic_mol"]

        return self

    def print_performance(self, alt_base=None, alt_top=None):
        print(
            f"Entre {min(self.z) if alt_base is None else alt_base}m e {max(self.z) if alt_top is None else alt_top}m")
        alt_base = None if alt_base is None else np.where(abs(self.z - alt_base) == min(abs(self.z - alt_base)))[0][0]
        alt_top = None if alt_top is None else np.where(abs(self.z - alt_top) == min(abs(self.z - alt_top)))[0][0]

        extinction = interp1d(self.df_solutions.Altitude, self.df_solutions.Extinction)
        backscatter = interp1d(self.df_solutions.Altitude, self.df_solutions.Backscatter)
        lidar_ratio = interp1d(self.df_solutions.Altitude, self.df_solutions.Lidarratio)

        mean_dev_extinction = mean_deviation(self._alpha["elastic_aer"][alt_base:alt_top],
                                             extinction(self.z[alt_base:alt_top]))
        mean_sqr_dev_extinction = mean_squared_deviation(self._alpha["elastic_aer"][alt_base:alt_top],
                                                         extinction(self.z[alt_base:alt_top]))

        mean_dev_backscatter = mean_deviation(self._beta["elastic_aer"][alt_base:alt_top],
                                              backscatter(self.z[alt_base:alt_top]))
        mean_sqr_dev_backscatter = mean_squared_deviation(self._beta["elastic_aer"][alt_base:alt_top],
                                                          backscatter(self.z[alt_base:alt_top]))

        mean_dev_lidar_ratio = mean_deviation(self.get_lidar_ratio()[alt_base:alt_top],
                                              lidar_ratio(self.z[alt_base:alt_top]))
        mean_sqr_lidar_ratio = mean_squared_deviation(self.get_lidar_ratio()[alt_base:alt_top],
                                                      lidar_ratio(self.z[alt_base:alt_top]))

        print("##################### Extinction #####################")

        print(f"mean deviation: {np.round(mean_dev_extinction * 100, 3)}%")
        print(f"mean squared deviation: {np.round(mean_sqr_dev_extinction * 100, 3)}%")

        print()
        print("##################### Backscatter #####################")

        print(f"mean deviation: {np.round(mean_dev_backscatter * 100, 3)}%")
        print(f"mean squared deviation: {np.round(mean_sqr_dev_backscatter * 100, 3)}%")

        print()
        print("##################### Lidar Ratio #####################")

        solution_mean_value = np.mean(self.get_lidar_ratio()[alt_base:alt_top])
        computed_mean_value = np.mean(lidar_ratio(self.z[alt_base:alt_top]))

        print(f"computed mean value: = {np.round(solution_mean_value, 2)}")
        print(f"true mean value = {np.round(computed_mean_value, 2)}")
        print(f"relative error = "
              f"{np.round(abs(computed_mean_value - solution_mean_value) * 100 / solution_mean_value, 2)}%")
        print()
        print(f"mean deviation: {np.round(mean_dev_lidar_ratio * 100, 3)}%")
        print(f"mean squared deviation: {np.round(mean_sqr_lidar_ratio * 100, 3)}%")
        print()

        return self

    def plot_performance(self, alt_base=None, alt_top=None):
        alt_base_sol = None if alt_base is None else \
            np.where(abs(self.raman_data.height - alt_base) == min(abs(self.raman_data.height - alt_base)))[0][0]
        alt_top_sol = None if alt_top is None else \
            np.where(abs(self.raman_data.height - alt_top) == min(abs(self.raman_data.height - alt_top)))[0][0]

        alt_base = None if alt_base is None else np.where(abs(self.z - alt_base) == min(abs(self.z - alt_base)))[0][0]
        alt_top = None if alt_top is None else np.where(abs(self.z - alt_top) == min(abs(self.z - alt_top)))[0][0]
        
        print(alt_base_sol)
        print(alt_top_sol)
        print(alt_base)
        print(alt_top)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        ax1.set_ylabel("Height (km)")
        fig.set_figwidth(15)
        fig.set_figheight(9)

        printter(self.df_solutions.Extinction * 1e4,
                 self.df_solutions.Altitude / 1e3,
                 self._alpha["elastic_aer"] * 1e4,
                 self.z / 1e3,
                 r"Extinction coefficient (10$^{-4}$ m$^{-1}$)",
                 alt_base_sol,
                 alt_top_sol,
                 alt_base,
                 alt_top,
                 ax1)

        printter(self.df_solutions.Backscatter * 1e6,
                 self.df_solutions.Altitude / 1e3,
                 self._beta["elastic_aer"] * 1e6,
                 self.z / 1e3,
                 r"Backscatter coefficient (10$^{-6}$ m$^{-1}$ sr$^{-1}$)",
                 alt_base_sol,
                 alt_top_sol,
                 alt_base,
                 alt_top,
                 ax2)

        printter(self.df_solutions.Lidarratio,
                 self.df_solutions.Altitude / 1e3,
                 self.get_lidar_ratio(),
                 self.z / 1e3,
                 "Lidar ratio (sr)",
                 alt_base_sol,
                 alt_top_sol,
                 alt_base,
                 alt_top,
                 ax3,
                 self.lidar_wavelength)

        ax3.legend()
        plt.savefig(f"figs/{alt_top}_{int(self.lidar_wavelength * 1e9)}.png", dpi=300)

        return self


def printter(x_solution, y_solution, x_computed, y_computed, x_label, alt_base_sol, alt_top_sol, alt_base, alt_top, ax,
             wavelength=None):
    ax.plot(x_solution[alt_base_sol: alt_top_sol],
            y_solution[alt_base_sol: alt_top_sol],
            "-",
            linewidth=3,
            alpha=0.7,
            color="red",
            # label=f"exact solution"
            )

    if wavelength is not None:
        ax.plot(x_solution[alt_base_sol: alt_top_sol],
                y_solution[alt_base_sol: alt_top_sol],
                "-",
                linewidth=3,
                alpha=0.7,
                color="red",
                label=f"exact solution ({int(wavelength * 1e9)} nm)"
                )

    ax.plot(x_computed[alt_base: alt_top],
            y_computed[alt_base: alt_top],
            "--",
            color="black",
            label="computed solution")

    # plt.gca().spines["top"].set_visible(False)
    # plt.gca().spines["right"].set_visible(False)
    ax.grid(alpha=0.65)

    # plt.legend()
    # plt.ylabel("Height (km)")
    ax.set_xlabel(x_label)
    # plt.show()


def print_data_info(j):
    import netCDF4 as nc

    print('Elastic Signal')
    print(nc.Dataset(f"synthetic_signals/Elastic_signals/nctd/sigelwv{j}.000"))

    print()
    print('Raman Signal')
    print(nc.Dataset(f"synthetic_signals/Raman_signals/nctd/sigrawv{j}.000"))

    print()
    print('Pressure and Temperature')
    print(nc.Dataset("synthetic_signals/Temperature and Pressure/nctd/PTsim.txt"))

    print()
    print('Solutions')
    print(nc.Dataset(f"synthetic_signals/Solutions/nctd/aerowv{j}.000"))
    print()


def open_data(j):
    import os

    os.chdir('synthetic_signals')

    df_temp_pressure = pd.read_csv("Temperature and Pressure/txt/PTsim.txt.txt", sep=' ').drop(columns='Length')

    df_sol = pd.read_csv(f"Solutions/txt/aerowv{j}.000.txt", sep=' ').drop(columns='Length')

    df_temp_pressure['Pressure'] *= 100
    df_temp_pressure['Temperature'] += 273.15

    ######

    os.chdir("Elastic_signals/txt/")
    dfs_elastic_signals = [pd.read_csv(file_name, sep=' ').drop(columns='Length')
                           for file_name in os.listdir()
                           if file_name.startswith(f'sigelwv{j}.')]

    df_elastic_signals = pd.DataFrame()

    for i, df in enumerate(dfs_elastic_signals):
        df_elastic_signals[f"Signal{i}"] = df["Signal"]

    df_elastic_signals["MeanSignal"] = df_elastic_signals.mean(axis=1)
    df_elastic_signals["Std"] = df_elastic_signals.drop(columns="MeanSignal").std(axis=1)

    df_elastic_signals["Altitude"] = dfs_elastic_signals[0]["Altitude"]

    del dfs_elastic_signals

    #####

    os.chdir("../../Raman_signals/txt/")

    dfs_raman_signals = [pd.read_csv(file_name, sep=' ').drop(columns='Length')
                         for file_name in os.listdir()
                         if file_name.startswith(f'sigrawv{j}.')]

    df_raman_signals = pd.DataFrame()

    for i, df in enumerate(dfs_raman_signals):
        df_raman_signals[f"Signal{i}"] = df["Signal"]

    df_raman_signals["MeanSignal"] = df_raman_signals.mean(axis=1)
    df_raman_signals["Std"] = df_raman_signals.drop(columns="MeanSignal").std(axis=1)

    df_raman_signals["Altitude"] = dfs_raman_signals[0]["Altitude"]

    del dfs_raman_signals

    os.chdir('../../..')

    return (df_elastic_signals,
            df_raman_signals,
            df_temp_pressure,
            df_sol)
