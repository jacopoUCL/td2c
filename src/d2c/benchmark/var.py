
import pandas as pd
import pickle
import statsmodels.tsa.api as tsa

from d2c.benchmark.base import BaseCausalInference

class VAR(BaseCausalInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.returns_proba = True

    def infer(self, single_ts,**kwargs):
        model = tsa.var.var_model.VAR(single_ts)
        results = model.fit(maxlags=self.maxlags)
        return results
    
    def build_causal_df(self, results, n_variables):

        """
        coefs[i, j, k] gives the effect of the k-th variable's lag i on the j-th variable. 

        eg. y_1 = 0.5*y_1(t-1) + 0.3*y_2(t-1) + 0.2*y_1(t-2) + 0.1*y_2(t-2)
        y_2 = 0.1*y_1(t-1) + 0.3*y_2(t-1) + 0.2*y_1(t-2) + 0.1*y_2(t-2)

        will have 
        coefs[0, 0, 0] = 0.5
        coefs[0, 0, 1] = 0.3
        coefs[1, 0, 0] = 0.2
        coefs[1, 0, 1] = 0.1
        coefs[0, 1, 0] = 0.1
        coefs[0, 1, 1] = 0.3
        [...]

        We consider the link causal if the p-value is below 0.05 and the effect is above 0.1.
        """
        pvalues = results.pvalues
        values = results.coefs

        #initialization
        pairs = [(source, effect) for source in range(n_variables, n_variables * self.maxlags + n_variables) for effect in range(n_variables)]
        multi_index = pd.MultiIndex.from_tuples(pairs, names=['from', 'to'])
        causal_dataframe = pd.DataFrame(index=multi_index, columns=['effect', 'p_value', 'probability', 'is_causal'])

        for lag in range(self.maxlags):
            for source in range(n_variables):
                for effect in range(n_variables):
                    current_pvalue = pvalues[source+lag*n_variables, effect]
                    current_value = values[lag][effect][source]

                    is_causal = 0 if current_pvalue > 0.05 else 0 if abs(current_value) < 0.1 else 1

                    # this is ok because 
                    # lag is 0-based
                    # so it's equivalent to 
                    # source+(lag+1)*n_variables
                    causal_dataframe.loc[(n_variables + source+lag*n_variables, effect)] = current_value, current_pvalue, None, is_causal

        #break the multiindex into columns (from and to)
        causal_dataframe.reset_index(inplace=True)

        return causal_dataframe

