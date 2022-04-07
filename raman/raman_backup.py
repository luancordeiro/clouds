import numpy as np
import pandas as pd
from alpha_beta_mol import AlphaBetaMolecular
from scipy.integrate import cumtrapz, trapz

# from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams.update({'font.family': 'serif', 'font.size': 14, 'font.weight': 'light'})


def diff_sliding_average(y, x, window=3, degree=None):
    return np.gradient(pd.Series(y).rolling(window).mean()) / np.gradient(x)


def diff_ewm(y, x, window=3, degree=None):
    return np.gradient(pd.Series(y).ewm(span=window).mean()) / np.gradient(x)


def diff_uniform_filter(y, x, window=3, degree=None):
    return np.gradient(uniform_filter1d(y, size=window)) / np.gradient(x)


def diff_savgol_filter(y, x, window=3, degree=2):
    return np.gradient(savgol_filter(y, window, degree)) / np.gradient(x)


def diff_sliding_lsq_fit(y, x, window=3, degree=None):
    diff_y = []
    for i in range(window, len(y) + 1):
        x_fit = x[i - window:i]
        y_fit = y[i - window:i]

        a_ = np.vstack([x_fit, np.ones(len(x_fit))]).T
        a, _ = np.linalg.lstsq(a_, y_fit, rcond=None)[0]

        if i != window:
            diff_y.append(a)
        else:
            diff_y += [a] * window

    return np.array(diff_y)


def diff_sliding_polynomial_fit(y, x, window, degree=2):
    h = 1e-5
    diff_y = []
    for i in range(window, len(y) + 1):
        x_fit = x[i - window:i]
        y_fit = y[i - window:i]

        p = np.poly1d(np.polyfit(x_fit, y_fit, degree))

        if i != window:
            diff_y.append((p(x_fit[i - 1] + h) - p(x_fit[i - 1])) / h)
        else:
            for x_ in x_fit:
                diff_y.append((p(x_ + h) - p(x_)) / h)

    return np.array(diff_y)


def diff_sliding_polynomial_fit_new_new_new(y, x, window=None, degree=None):
    def find_new_top_index(base_index):
        min_top = 10
        max_top = 45

        correlations = [0] * min_top
        index = range(base_index + min_top, base_index + max_top)
        for i in index:
            correlations.append(np.corrcoef(y[base_index: i], x[base_index: i])[0, 1])

        new_index = np.where(np.array(correlations) == max(correlations))[0][0]

        return base_index + new_index

    y = pd.Series(y).ewm(alpha=0.1).mean()

    base_index = 0
    top_index = find_new_top_index(base_index)

    y_diff = []
    while True:
        x_fit = x[base_index:top_index]
        y_fit = y[base_index:top_index]

        a_ = np.vstack([x_fit, np.ones(len(x_fit))]).T
        a, _ = np.linalg.lstsq(a_, y_fit, rcond=None)[0]

        if top_index is None:
            y_diff += [a] * (len(x) - base_index)
        else:
            y_diff += [a] * (top_index - base_index)

        base_index = top_index
        if base_index is None:
            break

        top_index = find_new_top_index(base_index)
        if len(x) - top_index < 20:
            top_index = None

    return np.array(y_diff)


def mean_squared_deviation(x: np.array, s: np.array) -> float:
    return np.sqrt(np.mean(((x - s) / s) ** 2))


def mean_deviation(x: np.array, s: np.array) -> np.ndarray:
    return np.mean((x - s) / s)


class OriginalData:
    def __init__(self,
                 height: np.array,
                 elastic_signal: np.array,
                 inelastic_signal: np.array,
                 p_air: np.array,
                 t_air: np.array):
        self.height = np.array(height)
        self.elastic_signal = np.array(elastic_signal)
        self.inelastic_signal = np.array(inelastic_signal)
        self.p_air = np.array(p_air)
        self.t_air = np.array(t_air)

    def get_data(self):
        return (self.height.copy(),
                self.elastic_signal.copy(),
                self.inelastic_signal.copy(),
                self.p_air.copy(),
                self.t_air.copy())

    def get_data_frame(self):
        return pd.DataFrame(data=np.array(self.get_data()).T,
                            columns=["z", "elastic_signal", "inelastic_signal", "p_air", "t_air"])


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
    _diff_window = 10
    _diff_strategy = diff_sliding_lsq_fit
    _diff_degree = 1
    _vertical_window = None

    def __init__(self,
                 z: np.array,
                 elastic_signal: np.array,
                 inelastic_signal: np.array,
                 lidar_wavelength: float,
                 raman_wavelength: float,
                 angstrom_coeff: float,
                 p_air: np.array,
                 t_air: np.array,
                 ref: int = 4000,
                 beta_ref: int = None,
                 co2ppmv: int = 392,
                 df_solutions: pd.DataFrame = None):

        self.original_data = OriginalData(z, elastic_signal, inelastic_signal, p_air, t_air)
        self.z, self.elastic_signal, self.inelastic_signal, self.p_air, self.t_air = self.original_data.get_data()

        self.lidar_wavelength = lidar_wavelength
        self.raman_wavelength = raman_wavelength
        self.angstrom_coeff = angstrom_coeff
        self.co2ppmv = co2ppmv
        self.z_ref = ref
        self.beta_ref = beta_ref

        self.df_solutions = df_solutions

    def set_beta_ref(self, beta_ref):
        self.beta_ref = beta_ref
        return self

    def get_alpha(self):
        return self._alpha.copy()

    def get_beta(self):
        return self._beta.copy()

    def get_lidar_ratio(self):
        return self._alpha["elastic_aer"] / self._beta["elastic_aer"]

    def _alpha_beta_molecular(self, co2ppmv):
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

        self._ref = np.where(abs(self.z - self.z_ref) == min(abs(self.z - self.z_ref)))[0][0]
        self._beta_ref = self._beta["elastic_mol"][self._ref] if self.beta_ref is None else self.beta_ref

    def _raman_scatterer_numerical_density(self) -> np.array:
        """Com base na eq. dos gases ideais e na razao de nitrogenio na atmosfera, calcula o perfil da densidade
        numerica utilizando os perfis de temperatura e pressao"""
        atm_numerical_density = self.p_air / (1.380649e-23 * self.t_air)
        return atm_numerical_density * 78.08e-2

    def _diff(self, y, x) -> np.array:
        """Realiza a suavizacao da curva e, em seguida, calcula a derivada necessaria para o calculo do coeficiente de
        extincao dos aerossois com base na estrategia escolhida."""
        return self._diff_strategy(y, x, self._diff_window, self._diff_degree)

    def _alpha_elastic_aer(self) -> np.array:
        """Retorna o coeficiente de extincao de aerossois."""
        ranged_corrected_signal = self.inelastic_signal * self.z ** 2

        y = np.log(self._raman_scatterer_numerical_density() / ranged_corrected_signal)

        diff_num_signal = self._diff(y, self.z)

        return (diff_num_signal - self._alpha['elastic_mol'] - self._alpha['inelastic_mol']) / \
               (1 + (self.lidar_wavelength / self.raman_wavelength) ** self.angstrom_coeff)

    def _alpha_elastic_total(self) -> np.array:
        return self._alpha["elastic_aer"] + self._alpha["elastic_mol"]

    def _alpha_inelastic_total(self) -> np.array:
        return self._alpha["inelastic_aer"] + self._alpha["inelastic_mol"]

    def _beta_elastic_total(self):
        """Calcula o coeficiente de retroespalhamento total de moleculas e particulas."""

        scatterer_numerical_density = self._raman_scatterer_numerical_density()

        signal_ratio = ((self.inelastic_signal[self._ref] * self.elastic_signal * scatterer_numerical_density) /
                        (self.elastic_signal[self._ref] * self.inelastic_signal * scatterer_numerical_density[
                            self._ref]))

        attenuation_ratio = (np.exp(-cumtrapz(x=self.z, y=self._alpha_inelastic_total(), initial=0) +
                                    trapz(x=self.z[:self._ref], y=self._alpha_inelastic_total()[:self._ref])) /
                             np.exp(-cumtrapz(x=self.z, y=self._alpha_elastic_total(), initial=0) +
                                    trapz(x=self.z[:self._ref], y=self._alpha_elastic_total()[:self._ref])))

        return self._beta_ref * signal_ratio * attenuation_ratio

    def _set_vertical_average_window(self, vertical_window=None) -> None:
        if vertical_window is not None:
            df = self.original_data.get_data_frame()

            df = df.groupby(df.index // vertical_window).mean()

            self.z, self.elastic_signal, self.inelastic_signal, self.p_air, self.t_air = (df.z.to_numpy(),
                                                                                          df.elastic_signal.to_numpy(),
                                                                                          df.inelastic_signal.to_numpy(),
                                                                                          df.p_air.to_numpy(),
                                                                                          df.t_air.to_numpy())
            del df
        else:
            self.z, self.elastic_signal, self.inelastic_signal, self.p_air, self.t_air = self.original_data.get_data()

        self._alpha_beta_molecular(self.co2ppmv)

    def fit(self, vertical_average_window=None, diff_strategy=diff_sliding_lsq_fit, diff_window=58, diff_degree=1):
        self._diff_window = diff_window
        self._diff_strategy = diff_strategy
        self._diff_degree = diff_degree
        self._set_vertical_average_window(self._vertical_window if vertical_average_window is None
                                          else vertical_average_window)

        self._alpha["elastic_aer"] = self._alpha_elastic_aer()

        self._alpha["inelastic_aer"] = self._alpha["elastic_aer"] / (
                self.raman_wavelength / self.lidar_wavelength) ** self.angstrom_coeff

        self._beta["elastic_aer"] = self._beta_elastic_total() - self._beta["elastic_mol"]

        return self

    def print_performance(self, alt_base=None, alt_top=None):
        print(f"Entre {min(self.z) if alt_base is None else alt_base}m e {max(self.z) if alt_top is None else alt_top}m")
        alt_base = None if alt_base is None else np.where(abs(self.z - alt_base) == min(abs(self.z - alt_base)))[0][0]
        alt_top = None if alt_top is None else np.where(abs(self.z - alt_top) == min(abs(self.z - alt_top)))[0][0]

        extinction = interp1d(self.df_solutions.Altitude, self.df_solutions.Extinction)
        backscatter = interp1d(self.df_solutions.Altitude, self.df_solutions.Backscatter)
        lidar_ratio = interp1d(self.df_solutions.Altitude, self.df_solutions.Lidarratio)
        
        mean_dev_extinction = mean_deviation(self._alpha["elastic_aer"][alt_base:alt_top], extinction(self.z[alt_base:alt_top]))
        mean_sqr_dev_extinction = mean_squared_deviation(self._alpha["elastic_aer"][alt_base:alt_top], extinction(self.z[alt_base:alt_top]))

        mean_dev_backscatter = mean_deviation(self._beta["elastic_aer"][alt_base:alt_top], backscatter(self.z[alt_base:alt_top]))
        mean_sqr_dev_backscatter = mean_squared_deviation(self._beta["elastic_aer"][alt_base:alt_top], backscatter(self.z[alt_base:alt_top]))

        mean_dev_lidar_ratio = mean_deviation(self.get_lidar_ratio()[alt_base:alt_top], lidar_ratio(self.z[alt_base:alt_top]))
        mean_sqr_lidar_ratio = mean_squared_deviation(self.get_lidar_ratio()[alt_base:alt_top], lidar_ratio(self.z[alt_base:alt_top]))

        print("##################### Extinction #####################")

        print(f"mean deviation: {np.round(mean_dev_backscatter, 3) * 100}%")
        print(f"mean squared deviation: {np.round(mean_sqr_dev_backscatter, 3) * 100}%")

        print()
        print("##################### Backscatter #####################")

        print(f"mean deviation: {np.round(mean_dev_extinction, 3) * 100}%")
        print(f"mean squared deviation: {np.round(mean_sqr_dev_extinction, 3) * 100}%")

        print()
        print("##################### Lidar Ratio #####################")

        solution_mean_value = np.mean(self.get_lidar_ratio()[alt_base:alt_top])
        computed_mean_value = np.mean(lidar_ratio(self.z[alt_base:alt_top]))

        print(f"computed mean value: = {np.round(solution_mean_value, 2)}")
        print(f"true mean value = {np.round(computed_mean_value, 2)}")
        print(f"relative error = {np.round(abs(computed_mean_value - solution_mean_value) / solution_mean_value, 2) * 100}%")
        print(f"mean deviation: {np.round(mean_dev_lidar_ratio, 3) * 100}%")
        print(f"mean squared deviation: {np.round(mean_sqr_lidar_ratio, 3) * 100}%")

        return self


    def plot_performance(self, alt_base=None, alt_top=None):
        alt_base_sol = None if alt_base is None else np.where(abs(self.original_data.height - alt_base) == min(abs(self.original_data.height - alt_base)))[0][0]
        alt_top_sol = None if alt_top is None else np.where(abs(self.original_data.height - alt_top) == min(abs(self.original_data.height - alt_top)))[0][0]

        alt_base = None if alt_base is None else np.where(abs(self.z - alt_base) == min(abs(self.z - alt_base)))[0][0]
        alt_top = None if alt_top is None else np.where(abs(self.z - alt_top) == min(abs(self.z - alt_top)))[0][0]


        printter(self.df_solutions.Extinction * 1e4,
                 self.df_solutions.Altitude / 1e3,
                 self._alpha["elastic_aer"] * 1e4,
                 self.z / 1e3,
                 r"Extinction coefficient (10$^{-4}$ m$^{-1}$)",
                 alt_base_sol,
                 alt_top_sol,
                 alt_base,
                 alt_top)

        printter(self.df_solutions.Backscatter * 1e6,
                 self.df_solutions.Altitude / 1e3,
                 self._beta["elastic_aer"] * 1e6,
                 self.z / 1e3,
                 r"Backscatter coefficient (10$^{-6}$ m$^{-1}$ sr$^{-1}$)",
                 alt_base_sol,
                 alt_top_sol,
                 alt_base,
                 alt_top)

        printter(self.df_solutions.Lidarratio,
                 self.df_solutions.Altitude / 1e3,
                 self.get_lidar_ratio(),
                 self.z / 1e3,
                 "Lidar ratio (sr)",
                 alt_base_sol,
                 alt_top_sol,
                 alt_base,
                 alt_top)

        return self


def printter(x_solution, y_solution, x_computed, y_computed, x_label, alt_base_sol, alt_top_sol, alt_base, alt_top):
    plt.plot(x_solution[alt_base_sol: alt_top_sol],
             y_solution[alt_base_sol: alt_top_sol],
             "-",
             linewidth=3,
             alpha=0.7,
             color="red",
             label="exact solution")

    plt.plot(x_computed[alt_base: alt_top],
             y_computed[alt_base: alt_top],
             "--",
             color="black",
             label="computed solution")

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.grid(alpha=0.65)

    plt.legend()
    plt.ylabel("Height (km)")
    plt.xlabel(x_label)
    plt.show()


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

    df_raman_signals["Altitude"] = dfs_raman_signals[0]["Altitude"]

    del dfs_raman_signals

    os.chdir('../../..')

    return (df_elastic_signals[['Altitude', 'MeanSignal']],
            df_raman_signals[['Altitude', 'MeanSignal']],
            df_temp_pressure,
            df_sol)


if __name__ == "__main__":
    import netCDF4 as nc
    import os
    import matplotlib.pyplot as plt

    n = 1  # 1 ou 2

    wavelength_lidar = 355e-9 if n == 1 else 532e-9
    wavelength_raman = 387e-9 if n == 1 else 607e-9

    # print_data_info(n)

    df_elastic, df_inelastic, df_met, df_solutions = open_data(n)
    # print(df_elastic.head())

    ind_min = 23
    ind_max = 500

    raman_test = Raman(df_elastic.Altitude[ind_min:ind_max],
                       df_elastic.MeanSignal[ind_min:ind_max],
                       df_inelastic.MeanSignal[ind_min:ind_max],
                       wavelength_lidar,
                       wavelength_raman,
                       1.8,
                       df_met.Pressure[ind_min:ind_max],
                       df_met.Temperature[ind_min:ind_max],
                       ref=4000,
                       extinction_solution=df_solutions.Extinction[ind_min:ind_max])

    # raman_test.set_window(58).set_diff_strategy(diff_sliding_lsq_fit)

    msqr_dev, m_dev = raman_test.get_metrics()

    print(f'mean square deviation = {msqr_dev}')
    print(f'mean deviation = {m_dev}')

    print()

    plt.plot(raman_test.extinction_solution, raman_test.z, '-', linewidth=3, color='red', label="exact solution")
    plt.plot(raman_test.aer_extinction(), raman_test.z, '--', color='black', label="computed solution")
    plt.legend()
    plt.ylabel('Altitude (m)')
    plt.xlabel('Extinction')
    plt.show()
