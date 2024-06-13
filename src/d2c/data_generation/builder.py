"""
This module is responsible for generating time series data based on the specified parameters.
"""

import numpy as np
import warnings
import pickle
import networkx as nx
from d2c.data_generation.models import model_registry

class TSBuilder():
    def __init__(self, observations_per_time_series=200, maxlags=4, n_variables=5, time_series_per_process=10, processes_to_use=list(range(1, 21)), noise_std=0.1, max_neighborhood_size = 6, seed=42, max_attempts=20, verbose=True):
        """
        Initializes the TSBuilder object with the specified parameters.

        Args:
            observations_per_time_series (int): Number of observations per time series.
            maxlags (int): Maximum number of time lags.
            n_variables (int): Number of variables in the time series.
            time_series_per_process (int): Number of time series per process.
            processes_to_use (list): List of process IDs to use.
            noise_std (float): Standard deviation of the noise.
            max_neighborhood_size (int): Maximum size of the neighborhood.
            seed (int): Seed for random number generation.
            verbose (bool): Whether to print verbose output.
        """
        self.observations_per_time_series = observations_per_time_series
        self.maxlags = maxlags #This is only used to build the corresponding DAGs
        self.n_variables = n_variables
        self.ts_per_process = time_series_per_process
        self.noise_std = noise_std
        self.seed = seed
        self.max_attempts = max_attempts
        self.verbose = verbose
        self.processes_to_use = processes_to_use
        self.max_neighborhood_size = min(max_neighborhood_size, n_variables)

        self.generated_observations = {}
        self.generated_dags = {}
        self.neighbors = {}     


    # def _generate_random_excluding_zero(self, max_time_lag, n_variables, exclusion_zone=0.1):
    #     result = np.zeros((max_time_lag, n_variables))
    #     for i in range(max_time_lag):
    #         for j in range(n_variables):
    #             # Randomly choose between the two ranges
    #             if np.random.rand() > 0.5:
    #                 result[i, j] = np.random.uniform(-1, -exclusion_zone)
    #             else:
    #                 result[i, j] = np.random.uniform(exclusion_zone, 1)
    #     return result

    def build(self):
        """
        Builds the time series data.

        Args:
            max_attempts (int): Maximum number of attempts to generate valid time series.

        Raises:
            ValueError: If a non-finite value is detected in the generated time series.

        """
        np.random.seed(self.seed)
        for process_id in self.processes_to_use:

            self.generated_observations[process_id] = {}
            self.generated_dags[process_id] = {}
            self.neighbors[process_id] = {}

            model_instance = model_registry.get_model(process_id)()
            
            lines_to_initialize = model_instance.get_maximum_time_lag() + 1 
            #each model has a different time lag
            # max_time_lag t --> t+1 is 0, t-1 --> t+1 is 1, t-2 --> t+1 is 2, etc. [CONVENTION]
            # but if 0, we need to initialize one line, if 1, we need to initialize two lines, etc. 
            # for this reason, we take max_time_lag + 1
            total_ts_lines = self.observations_per_time_series + lines_to_initialize

            for ts_index in range(self.ts_per_process):
                #The seeding is here because attempts can vary across executions, likely due to floating point operations
                
                attempted_series = []
                attempted_neighbors = []
                attempts = 0
                while attempts < self.max_attempts:
                        
                    W = np.random.normal(0, self.noise_std, (total_ts_lines, self.n_variables))
                    # noise to zero 
                    # W = np.zeros((total_ts_lines, self.n_variables))
                    size_N_j = np.random.randint(1, self.max_neighborhood_size + 1, self.n_variables)

                                            
                    #fill up the neighborhood of each variable, starting from the variables itself
                    N_j = [[j] for j in range(self.n_variables)]
                    for j in range(self.n_variables):
                        remaining_size = size_N_j[j] - 1
                        remaining = np.setdiff1d(range(self.n_variables), N_j[j])
                        if remaining_size > 0:
                            N_j[j] = np.append(N_j[j], np.random.choice(remaining, remaining_size, replace=False))
                    Y_n = np.full((total_ts_lines, self.n_variables), np.nan)
                    
                    # Initialize the first `lines_to_initialize` rows with starting values if needed
                    #Random
                    Y_n[:lines_to_initialize] = np.random.uniform(-1, 1, (lines_to_initialize, self.n_variables))

                    
                    #the update functions expect to be given the last 'available' line to compute following values
                    #so we start from lines_to_initialize - 1 and go up to total_ts_lines - 1
                    for t in range(lines_to_initialize-1, total_ts_lines-1): 
                        for j in range(self.n_variables):
                            Y_n[t+1, j] = model_instance.update(Y_n, t, j, N_j[j], W)
                            

                    # attempted_series.append(Y_n[max_time_lag:])       
                    attempted_series.append(Y_n)
                    attempted_neighbors.append(N_j)
                    attempts += 1
                               
                chosen_series = None
                chosen_neighbors = None
                threshold = 1e-6  # Define a threshold for extremely small numbers in absolute value
                for series_idx, Y_n in enumerate(attempted_series):
                    if not np.any(np.isnan(Y_n)) and not np.any(np.isinf(Y_n)):
                        if np.min(Y_n) > -1e6 and np.max(Y_n) < 1e6:
                            if not np.any(np.abs(Y_n) < threshold):  # Check if all values are above the threshold
                                chosen_series = Y_n
                                chosen_neighbors = attempted_neighbors[series_idx]
                                break 
                
                if chosen_series is None:
                    raise ValueError(f"Failed to generate valid TS for model {process_id}, TS index {ts_index} after {self.max_attempts} attempts. Try again with a different seed.")

                self.generated_dags[process_id][ts_index] = model_instance.build_dag(T=self.maxlags,N_j=chosen_neighbors, N=self.n_variables)
                self.generated_observations[process_id][ts_index] = chosen_series[lines_to_initialize:]
                self.neighbors[process_id][ts_index] = chosen_neighbors


    def _prepare_dags(self):
        """
        This method renames the nodes in the generated DAGs based on a different naming convention.
        Specifically, the nodes are renamed from "Y[t][j]" to "{j}_t-{maxlags-t}".
        It also assigns an index to each DAG based on the process ID and time series index.
        # TODO: handle the case when not all processes are considered! 
        Returns:
            None
        """
        rename_dict = {f"Y[{t}][{j}]": f"{j}_t-{self.maxlags-t}" for t in range(self.maxlags+1) for j in range(self.n_variables)}
        for process_id in self.generated_dags:
            for ts_index in self.generated_dags[process_id]:
                self.generated_dags[process_id][ts_index] = nx.relabel_nodes(self.generated_dags[process_id][ts_index], rename_dict)
                # self.generated_dags[process_id][ts_index].graph['index'] = (process_id - 1) * self.ts_per_process + ts_index

    def get_generated_observations(self):
        """
        Returns the generated observations.

        Returns:
            dict: A dictionary containing the generated observations.
        """
        return self.generated_observations
    
    def get_generated_dags(self):
        """
        Returns the generated DAGs.

        Returns:
            dict: A dictionary containing the generated DAGs.
        """
        self._prepare_dags()
        return self.generated_dags
    
    def get_neighbors(self):
        """
        Returns the generated neighbors.

        Returns:
            dict: A dictionary containing the generated neighbors.
        """
        return self.neighbors

    def to_pickle(self, path):
        """
        Saves the generated data to a pickle file.

        Args:
            path (str): Path to save the data.
        """
        self._prepare_dags()
        with open(path, 'wb') as f:
            pickle.dump((self.generated_observations, self.generated_dags, self.neighbors), f)