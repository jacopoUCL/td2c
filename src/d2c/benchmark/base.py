from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pickle

class BaseCausalInference:
    def __init__(self, ts_list, maxlags=3, ground_truth=None, n_jobs=1, suffix=''):

        if n_jobs < 1:
            raise ValueError('n_jobs must be at least 1')
        self.ts_list = ts_list
        self.maxlags = maxlags
        self.causal_dfs = {}
        self.ground_truth = ground_truth
        self.returns_proba = False #TODO: in subclasses
        self.n_jobs = n_jobs
        self.suffix = suffix

        self.causal_matrices = {}
        self.p_value_matrices = {}
        self.lag_matrices = {}
        

    def standardize(self, single_ts):
        # Standardize data
        single_ts -= single_ts.mean(axis=0)
        single_ts /= single_ts.std(axis=0)
        return single_ts

    def infer(self, single_ts,**kwargs):
        pass
    
    def build_causal_df(self, results, n_variables):
        pass

    def process_ts(self, ts_tuple):
        ts_index, ts = ts_tuple
        results = self.infer(ts, ts_index=ts_index)
        causal_df = self.build_causal_df(results, ts.shape[1])
        return ts_index, causal_df

    def run(self):

        ts_tuples = list(enumerate(self.ts_list))
        if self.n_jobs == 1:
            for ts_index, ts in ts_tuples:
                _, causal_df = self.process_ts((ts_index, ts)) 
                self.causal_dfs[ts_index] = causal_df
        else:
            with Pool(processes=self.n_jobs) as pool:
                results = pool.map(self.process_ts, ts_tuples)
            for ts_index, causal_df in results:
                self.causal_dfs[ts_index] = causal_df

        return self
    

    def build_causeme_matrices(self, n_variables):
        # assumes causal dfs are available
        for ts_index, causal_df in self.causal_dfs.items():
            causal_df_2 = causal_df.reset_index(drop=False) 
            self.causal_matrices[ts_index] = np.zeros((n_variables, n_variables))
            self.p_value_matrices[ts_index] = np.zeros((n_variables, n_variables))
            self.lag_matrices[ts_index] = np.zeros((n_variables, n_variables))
            for source_variable_index in range(n_variables):
                #filter by source
                source_df = causal_df_2.loc[causal_df_2['source'] % n_variables == source_variable_index]
                for target_variable_index in range(n_variables):
                    #filter by target
                    target_df = source_df.loc[source_df['target'] == target_variable_index].reset_index(drop=True)

                    #make 'pvalue' column numeric
                    target_df['pvalue'] = pd.to_numeric(target_df['pvalue'], errors='coerce')

                    #find line wheere column p-value is lowest
                    lowest_p_value_index = target_df['pvalue'].idxmin()
                    #get the corresponding row
                    lowest_p_value_row = target_df.loc[lowest_p_value_index]
                    #get the lag 
                    lag = lowest_p_value_row['source'] // n_variables
                    if lowest_p_value_row['is_causal'] == 1:
                        self.causal_matrices[ts_index][source_variable_index, target_variable_index] = lowest_p_value_row['value']
                        self.p_value_matrices[ts_index][source_variable_index, target_variable_index] = lowest_p_value_row['pvalue']
                        self.lag_matrices[ts_index][source_variable_index, target_variable_index] = lag

    def get_causal_matrices(self):
        return self.causal_matrices

    def get_p_value_matrices(self):
        return self.p_value_matrices
    
    def get_lag_matrices(self):
        return self.lag_matrices
    
    def get_causeme_matrices(self):
        return self.causal_matrices, self.p_value_matrices, self.lag_matrices

    def get_causal_dfs(self):
        return self.causal_dfs

    def set_causal_dfs(self, causal_dfs):
        self.causal_dfs = causal_dfs
    
    def set_ground_truth(self, causal_dfs):
        self.ground_truth = causal_dfs
    
    def filter_causal_dfs(self, d2c_causal_dfs):
        indexes = {}
        for i, df in d2c_causal_dfs.items():
            indexes[i] = list(df.index)

        updated_causal_dfs = {}
        for i, idx_i in indexes.items():
            updated_causal_dfs[i] = self.causal_dfs[i].loc[idx_i]

        updated_ground_truths = {}
        for i, idx_i in indexes.items():
            updated_ground_truths[i] = self.ground_truth[i].loc[idx_i]

        self.set_causal_dfs(updated_causal_dfs)
        self.set_ground_truth(updated_ground_truths)
 
    def get_predictions(self):
        """
        Returns a dictionary with the predictions of the causal relationships for each time series.
        This allows for the evaluation of the causal inference methods beyond the existing evaluate method.
        """
        predictions = {}
        for ts_idx in range(len(self.ts_dict)):
            y_test = self.ground_truth[ts_idx]['is_causal'].astype(int)
            y_hat = self.causal_dfs[ts_idx]['is_causal'].astype(int)
            y_test = y_test.loc[y_hat.index] #needed because of the filtering 
            y_prob = self.causal_dfs[ts_idx]['value'].values.astype(float)
            predictions[ts_idx] = (y_test,y_prob)

        return predictions



    def evaluate(self):
        
        metrics = []
        for ts_idx in range(len(self.ts_dict)):
            # print('\revaluating', ts_idx, 'of', len(self.ts_list), end='', flush=True)

            method_name = self.__class__.__name__ + self.suffix
            y_test = self.ground_truth[ts_idx]['is_causal'].astype(int)
            
            y_hat = self.causal_dfs[ts_idx]['is_causal'].astype(int)
            y_test = y_test.loc[y_hat.index]

            metrics.append([method_name, 'accuracy', accuracy_score(y_test, y_hat)])
            metrics.append([method_name, 'precision', precision_score(y_test, y_hat, zero_division=np.nan)])
            metrics.append([method_name, 'recall', recall_score(y_test, y_hat, zero_division=np.nan)])
            metrics.append([method_name, 'f1', f1_score(y_test, y_hat, zero_division=np.nan)])
            metrics.append([method_name, 'balanced_error', 1 - balanced_accuracy_score(y_test, y_hat)])   
            if self.returns_proba: 
                y_prob = self.causal_dfs[ts_idx]['value'].values.astype(float)
                if len(np.unique(y_test)) > 1:
                    metrics.append([method_name, 'auc', roc_auc_score(y_test, y_prob)])
        return metrics

    def save_causal_dfs(self, name):
        method_name = self.__class__.__name__ + self.suffix
        with open('../data/'+name+'_'+method_name+'_causal_dfs.pkl', 'wb') as f:
            pickle.dump(self.causal_dfs, f)