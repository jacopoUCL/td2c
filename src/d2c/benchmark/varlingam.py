
from d2c.benchmark.base import BaseCausalInference
import pandas as pd
import pickle
from lingam import VARLiNGAM as VARLiNGAM_



class VARLiNGAM(BaseCausalInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.returns_proba = True
        
    def infer(self, single_ts,**kwargs):
        """
        VARLiNGAM inference method.
        A simple wrapper around the VARLiNGAM class.
        We use default parameters. 
        We use the bootstrap method to estimate the causal effects.
        """
        model = VARLiNGAM_(lags=self.maxlags)
        results = model.bootstrap(single_ts,n_sampling=10)
        return results.get_total_causal_effects(min_causal_effect=None)
    
    def build_causal_df(self, total_causal_effects, n_variables):
        """
        VARLiNGAM already returns a dictionary 
        very similar to the causal_df convention that we adopt here.
        It can be converted to a DataFrame.
        N.B. It contains also contemporaneous effects, which we exclude.
        
        '|    |   from |   to |    effect |   probability |\n
         |---:|-------:|-----:|----------:|--------------:|\n
         |  0 |      4 |    2 |  0.491077 |          0.98 |\n
         |  1 |      5 |    0 |  0.370244 |          0.84 |\n
         |  2 |     10 |    0 | -0.698416 |          0.84 |\n
         |  3 |      9 |    2 | -0.724122 |          0.81 |\n
         |  4 |      7 |    2 | -0.277903 |          0.81 |'
        """

        df = pd.DataFrame(total_causal_effects) 
        df['is_causal'] = df['probability'] > 0.5
        df = df.loc[df['from'] >= n_variables] #exclude contemporaneous effects
        df['p-value'] = None
        return df[['from', 'to', 'effect', 'p-value', 'probability', 'is_causal']]