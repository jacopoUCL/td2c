from d2c.benchmark.base import BaseCausalInference

from causalnex.structure.dynotears import from_pandas_dynamic

import pandas as pd
import pickle

class DYNOTEARS(BaseCausalInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.returns_proba = True

    def infer(self, single_ts,**kwargs):
        return from_pandas_dynamic(pd.DataFrame(single_ts), p=self.maxlags)
    
    def build_causal_df(self, results, n_variables):
        """
        Dynotears returns a graph object.
        OutEdgeView([('0_lag0', '2_lag0'), ('0_lag2', '3_lag0'), ('1_lag0', '4_lag0'), ('2_lag0', '0_lag0'), ('3_lag1', '1_lag0'), ('3_lag2', '1_lag0'), ('4_lag0', '1_lag0'), ('4_lag1', '1_lag0'), ('4_lag2', '0_lag0'), ('4_lag2', '2_lag0')])
        This can be parsed and easily converted to a DataFrame.
        """
        #initialization
        pairs = [(source, effect) for source in range(n_variables, n_variables * self.maxlags + n_variables) for effect in range(n_variables)]
        multi_index = pd.MultiIndex.from_tuples(pairs, names=['from', 'to'])
        causal_dataframe = pd.DataFrame(index=multi_index, columns=['effect', 'p_value', 'probability', 'is_causal'])

        causal_dataframe['is_causal'] = 0
        causal_dataframe['probability'] = None
        causal_dataframe['p_value'] = 0
        causal_dataframe['value'] = 0

        for edge in results.edges:
            source = int(edge[0].split('_')[0])
            effect = int(edge[1].split('_')[0])
            lag = int(edge[0].split('lag')[1])
            if lag > 0: #we ignore contemporaneous relations for the moment
                value = results.get_edge_data(*edge)['weight']
                causal_dataframe.loc[(source+lag*n_variables, effect), 'is_causal'] = 1
                causal_dataframe.loc[(source+lag*n_variables, effect), 'effect'] = value
                causal_dataframe.loc[(source+lag*n_variables, effect), 'p_value'] = None

        causal_dataframe.reset_index(inplace=True)

        assert len([edge for edge in results.edges if 'lag0' not in edge[0]]) == causal_dataframe.is_causal.sum()
        return causal_dataframe