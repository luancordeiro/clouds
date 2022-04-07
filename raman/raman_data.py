import numpy as np
import pandas as pd


class RamanData:
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
        df = pd.DataFrame(data=[self.height,
                                self.elastic_signal,
                                self.inelastic_signal,
                                self.p_air,
                                self.t_air]).T

        df.columns = ["z", "elastic_signal", "inelastic_signal", "p_air", "t_air"]

        return df

    def data_binning(self, vertical_window):
        if vertical_window is None:
            return self.get_data()

        df = self.get_data_frame()

        height = df.z.groupby(df.index // vertical_window).mean().to_numpy()
        p_air = df.p_air.groupby(df.index // vertical_window).mean().to_numpy()
        t_air = df.t_air.groupby(df.index // vertical_window).mean().to_numpy()
        elastic_signal = df.elastic_signal.groupby(df.index // vertical_window).sum().to_numpy()
        inelastic_signal = df.inelastic_signal.groupby(df.index // vertical_window).sum().to_numpy()

        return (height,
                elastic_signal,
                inelastic_signal,
                p_air,
                t_air)
