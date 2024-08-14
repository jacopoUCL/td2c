from d2c.benchmark.base import BaseCausalInference

from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
from statsmodels.tools.sm_exceptions import InfeasibleTestError

import pandas as pd
import pickle

class Granger(BaseCausalInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, single_ts,**kwargs):
        results = {}

        for x1 in range(single_ts.shape[1]):
            gc2 = {}
            for x2 in range(single_ts.shape[1]):
                try:
                    gc_res = grangercausalitytests(single_ts[:,[x1,x2]], self.maxlags, verbose=False)
                    gc_res_lags = {}
                    for lag in range(1,self.maxlags+1):
                        gc_res_lags[lag] = gc_res[lag][0]['ssr_ftest'][1]
                except InfeasibleTestError:
                    gc_res_lags = {lag:np.nan for lag in range(1,self.maxlags+1)}
                gc2[int(x2)] = gc_res_lags 
            results[int(x1)] = gc2 

        return results
    
    def build_causal_df(self, results, n_variables):
        """
        grangercausalitytests returns a this structure for each pair of variables:
        {1: ({'ssr_ftest': (3.3884921764815416e-14, 0.9999998674716435, 249.0, 1),
            'ssr_chi2test': (3.4157089811119155e-14, 0.9999998525378735, 1),
            'lrtest': (-0.0, 1.0, 1),
            'params_ftest': (49.20154796128821, 2.1709506448984477e-11, 249.0, 1.0)},
            [<statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7fc2a04b7e50>,
            <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7fc2a021bfa0>,
            array([[0., 1., 0.]])]),
        2: [...]

        where the first key is the lag.
        We only store the p-value of the ssr_ftest which is in the second position of the first tuple after accessing the lag. So gc_res[lag][0]['ssr_ftest'][1]

        We check the p-value of the ssr_ftest to build the causal df.
        """
        #initialization
        pairs = [(source, effect) for source in range(n_variables, n_variables * self.maxlags + n_variables) for effect in range(n_variables)]
        multi_index = pd.MultiIndex.from_tuples(pairs, names=['from', 'to'])
        causal_dataframe = pd.DataFrame(index=multi_index, columns=['effect', 'p_value', 'probability', 'is_causal'])

        for lag in range(self.maxlags):
            for source in range(n_variables):
                for effect in range(n_variables):
                    current_pvalue = results[source][effect][lag+1]
                    
                    is_causal = 0 if current_pvalue > 0.05 else 1
                    
                    causal_dataframe.loc[(n_variables + source+lag*n_variables, effect)] = None, current_pvalue, None, is_causal 

        #break the multiindex into columns (from and to)
        causal_dataframe.reset_index(inplace=True)

        return causal_dataframe
