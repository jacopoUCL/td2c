import pandas as pd
from d2c.benchmark.base import BaseCausalInference
from d2c.descriptors.loader import DataLoader
from d2c.descriptors.d2c import D2C as D2C_


class D2CWrapper(BaseCausalInference):
    """
    D2C class wrapper for causal inference using the D2C algorithm.

    Parameters:
    - n_variables (int): Number of variables in the dataset. Default is 6.
    - model: The associated model used for prediction. Must implement the fit and predict methods.

    Attributes:
    - n_variables (int): Number of variables in the dataset.
    - model: The associated model used for prediction.
    - returns_proba (bool): Flag indicating whether the model returns probabilities.

    Methods:
    - infer(single_ts, **kwargs): Performs causal inference on a single time series.
    - build_causal_df(results): Builds the causal dataframe from the results.

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the D2C class.

        Parameters:
        - n_variables (int): Number of variables in the dataset. Default is 6.
        - model: The associated model used for prediction. Must implement the fit and predict methods.

        """
        self.n_variables = kwargs.pop('n_variables', 6)
        
        self.full = kwargs.pop('full', True)
        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise ValueError('model is required for D2C inference')
        
        super().__init__(*args, **kwargs)
        self.returns_proba = True

    def infer(self, single_ts, **kwargs):
        """
        Perform causal inference on a single time series.

        Parameters:
        - single_ts: The input time series data.

        Returns:
        - results: The results of the causal inference in a dataframe format.

        """
        data_for_d2c = DataLoader._create_lagged_single_ts(single_ts, self.maxlags)

        d2c = D2C_(dags = None, 
                    observations = [data_for_d2c], 
                    maxlags=self.maxlags,
                    n_variables=self.n_variables,
                    full=self.full)
        
        descriptors = d2c.compute_descriptors_without_dag(n_variables=self.n_variables,maxlags=self.maxlags)

        descriptors = pd.DataFrame(descriptors)
        X_test = descriptors.drop(['graph_id', 'edge_source', 'edge_dest','is_causal'], axis=1)

        y_pred_proba = self.model.predict_proba(X_test)[:,1]
        y_pred = y_pred_proba > 0.5

        descriptors['probability'] = y_pred_proba
        descriptors['is_causal'] = y_pred
        results = descriptors[['edge_source','edge_dest','probability','is_causal']]

        return results
        
    
    def build_causal_df(self, results, n_variables):
        """
        Build the causal dataframe from the results.

        Parameters:
        - results: The results of the causal inference.

        Returns:
        - causal_df: The causal dataframe with the expected format.

        """
        results.rename(columns={'edge_source':'from', 'edge_dest':'to'}, inplace=True)

        results['p_value'] = None
        results['effect'] = None

        causal_df = results[['from','to','effect','p_value','probability','is_causal']]

        return causal_df
