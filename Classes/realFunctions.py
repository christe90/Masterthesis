import pandas as pd
import numpy as np
import math

class realFunctions:
    def __init__(self, realFunctionName):
        self.realFunctionName = realFunctionName
        self.switcher={
            'Balkenbiegung':self._Balkenbiegung,
            'BeschlBewegungMitV0':self._BeschlBewegungMitV0,
            'Rapididaet':self._Rapididaet,
            'FermiDirac':self._FermiDirac,
            'Plank':self._Zerfallsgesetz,
            'SchwingungFederpendel':self._SchwingungFederpendel,
            'WaermeUebergang':self._Waermeuebergang
        }
        self.realFunction = self.switcher.get(self.realFunctionName)

    def _Balkenbiegung(self,**x_Data):
        y_Data = ((1/3) *x_Data['P'] * x_Data['L'] ** 3 / (x_Data['E'] * x_Data['I']))
        return y_Data

    def _BeschlBewegungMitV0(self, **x_Data):
        y_Data = (x_Data['v0'] * x_Data['t']) - (0.5)* (x_Data['g'] * x_Data['t']**2)
        return y_Data

    def _Rapididaet(self, **x_Data):
        y_Data = (x_Data['c'] * np.tanh(x_Data['theta']))
        return y_Data

    def _FermiDirac(self, **x_Data):
        y_Data = 1 / (math.exp((x_Data['E'] - x_Data['mue']) / (x_Data['k'] * x_Data['T'])) + 1)
        return y_Data

    def _Zerfallsgesetz(self, **x_Data):
        y_Data = x_Data['N0'] * (math.exp(-1 * (x_Data['lambda'] * x_Data['t'])))
        return y_Data

    def _SchwingungFederpendel(self, **x_Data):
        y_Data = (x_Data['y0'] * math.cos(math.sqrt(x_Data['D'] / x_Data['m']) * x_Data['t']))
        return y_Data

    def _Waermeuebergang(self, **x_Data):
        if x_Data['ri'] >= x_Data['ra']:
            x_Data['ra'] = x_Data['ri'] + 1
        y_Data = 2 * math.pi * x_Data['lambda'] *x_Data['upsilon'] / math.log(x_Data['ra'] / x_Data['ri'])
        return y_Data
