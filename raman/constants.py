class Constants:
    def __init__(self, co2ppmv=392):
        self.co2ppmv = co2ppmv

        # Atmospheric Constituents Concentration                [ppv]
        # ref: Seinfeld and Pandis (1998)
        self.N2ppv = 0.78084
        self.O2ppv = 0.20946
        self.Arppv = 0.00934
        self.Neppv = 1.80 * 1e-5
        self.Heppv = 5.20 * 1e-6
        self.Krppv = 1.10 * 1e-6
        self.H2ppv = 5.80 * 1e-7
        self.Xeppv = 9.00 * 1e-8
        self.CO2ppv = co2ppmv * 1e-6

        # Atmospheric Constituents Molecular weight             [g mol^-1]
        # ref: Handbook of Physics and Chemistry (CRC 1997)
        self.N2mwt = 28.013
        self.O2mwt = 31.999
        self.Armwt = 39.948
        self.Nemwt = 20.18
        self.Hemwt = 4.003
        self.Krmwt = 83.8
        self.H2mwt = 2.016
        self.Xemwt = 131.29
        self.CO2mwt = 44.01

        self.Airmwt = (self.N2ppv * self.N2mwt + self.O2ppv * self.O2mwt + self.Arppv * self.Armwt +
                       self.Neppv * self.Nemwt + self.Heppv * self.Hemwt + self.Krppv * self.Krmwt +
                       self.H2ppv * self.H2mwt + self.Xeppv * self.Xemwt + self.CO2ppv * self.CO2mwt) / \
                      (self.N2ppv + self.O2ppv + self.Arppv + self.Neppv + self.Heppv + self.Krppv +
                       self.H2ppv + self.Xeppv + self.CO2ppv)

        # Physic Constats
        self.k = 1.3806503e-23   # Boltzmann Constant                                                [J K^-1]
        self.Na = 6.0221367e+23  # Avogadro's number                                                 [# mol^-1]

        # Wallace and Hobbs, p. 65
        self.Rgas = self.k * self.Na               # Universal gas constant                          [J K^-1 mol^-1]
        self.Rair = self.Rgas / self.Airmwt * 1e3  # Dry air gas constant                            [J K^-1 kg^-1]

        # Standard Atmosphere Reference Values
        self.T0 = 273.15    # zero deg celcius                                                       [K]
        self.Tstd = 288.15  # Temperature                                                            [K]
        self.Pstd = 101325  # Pressure                                                               [Pa]

        # Bodhaine et al, 1999
        self.Mvol = 22.4141e-3                                     # Molar volume at Pstd and T0     [m^3 mol^-1]
        self.Nstd = (self.Na / self.Mvol) * (self.T0 / self.Tstd)  # Molec density at Tstd and Pstd  [# m^-3]


if __name__ == '__main__':
    c = Constants()
