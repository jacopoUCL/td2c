from d2c.benchmark.base import BaseCausalInference

from tigramite.pcmci import PCMCI as PCMCI_
from tigramite.independence_tests.parcorr import ParCorr
# from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.gpdc import GPDC

import tigramite.data_processing as pp

import pandas as pd
import pickle

class PCMCI(BaseCausalInference):
    def __init__(self, *args, **kwargs):
        self.CI = kwargs.pop('ci', 'ParCorr')

        super().__init__(*args, **kwargs)
        self.returns_proba = True

    def infer(self, single_ts,**kwargs):
        dataframe = pp.DataFrame(single_ts)
        if self.CI == 'ParCorr':
            cond_ind_test = ParCorr()
        elif self.CI == 'CMIknn':
            # cond_ind_test = CMIknn()
            raise NotImplementedError('This CI is not yet implemented')

        elif self.CI == 'GPDC':
            cond_ind_test = GPDC()
        else:
            raise NotImplementedError('This CI is not yet implemented')

        pcmci = PCMCI_(dataframe=dataframe, cond_ind_test=cond_ind_test)
        results = pcmci.run_pcmci(tau_max=self.maxlags)
        return results
    
    def build_causal_df(self, results, n_variables):
        """
        PCMCI returns undirected contemporaneous links and directed past links.
        A graph object, a p-value matrix, and a values matrix. 
        The graph object can be best used with PCMCI methods (eg. to plot).
        The other objects have shape (from, to, maxlags + 1).
        We can filter out the contemporaneous links and convert the results to a DataFrame.
        """
        pvalues = results['p_matrix']
        values = results['val_matrix']

        #initialization
        pairs = [(source, effect) for source in range(n_variables, n_variables * self.maxlags + n_variables) for effect in range(n_variables)]
        multi_index = pd.MultiIndex.from_tuples(pairs, names=['from', 'to'])
        causal_dataframe = pd.DataFrame(index=multi_index, columns=['effect', 'p_value', 'probability', 'is_causal'])

        for lag in range(1, self.maxlags + 1): # ignore undirected contemporaneous lags
            for source in range(n_variables):
                for effect in range(n_variables):
                    current_pvalue = pvalues[source][effect][lag]
                    current_value = values[source][effect][lag]

                    is_causal = 0 if current_pvalue > 0.05 else 0 if abs(current_value) < 0.1 else 1
                    
                    causal_dataframe.loc[(source + lag*n_variables, effect)] = current_value, current_pvalue, None, is_causal

        #break the multiindex into columns (from and to)
        causal_dataframe.reset_index(inplace=True)
        return causal_dataframe