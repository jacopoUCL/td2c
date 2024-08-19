import pandas as pd
from d2c.benchmark.base import BaseCausalInference
from d2c.descriptors_generation.loader import DataLoader
from d2c.descriptors_generation.d2c import D2C as D2C_


class D2CWrapper(BaseCausalInference):
    """
    D2C class wrapper for causal inference using the D2C algorithm.
    This is used as a standalone class when running the method against competitors. 
    This method - differently from the D2C class - can be executed directly on a raw time series and will perform 
    complete causal discovery of all possible edges. 

    Notice that this works on a single time series and not on a list. (Why??)

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

    Returns:
    - results: The results of the causal inference in a dataframe format.
    """
# It gives back the edges only, in the format: edge_source, edge_dest, probability, is_causal.
# Could add a function to return a DAG from the edges.

    def __init__(self, *args, **kwargs):
        """
        Initialize the D2C class.

        Parameters:
        - n_variables (int): Number of variables in the dataset. Default is 6.
        - model: The associated model used for prediction. Must implement the fit and predict methods.

        """
        self.n_variables = kwargs.pop('n_variables', 6)
        
        self.full = kwargs.pop('full', True)
        self.quantiles = kwargs.pop('quantiles', True)
        self.model = kwargs.pop('model', None)
        self.cmi = kwargs.pop('cmi', 'cmiknn_3')
        self.mb_estimator = kwargs.pop('mb_estimator', 'ts')
        self.normalize = kwargs.pop('normalize', True)
        self.filename = kwargs.pop('filename', None)
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

        #get ts_index from kwargs
        ts_index = kwargs.get('ts_index', None)

        data_for_d2c = DataLoader._create_lagged_single_ts(single_ts, self.maxlags)

        d2c = D2C_(dags = None, 
                    observations = [data_for_d2c], 
                    maxlags=self.maxlags,
                    n_variables=self.n_variables,
                    full=self.full,
                    quantiles=self.quantiles,
                    cmi=self.cmi,
                    normalize=self.normalize,
                    mb_estimator=self.mb_estimator)
        
        descriptors = d2c.compute_descriptors_without_dag(n_variables=self.n_variables,maxlags=self.maxlags)

        descriptors = pd.DataFrame(descriptors)
        if self.filename is not None:
            descriptors.to_csv(self.filename+'_'+str(ts_index)+'.csv')
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
