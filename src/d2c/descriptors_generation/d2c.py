from multiprocessing import Pool
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

from d2c.descriptors_generation.utils import coeff, HOC
from d2c.descriptors_generation.estimators import MarkovBlanketEstimator, MutualInformationEstimator

# class for D2C and TD2C
class D2C:
    """
    D2C class for computing descriptors in a time series dataset.

    Args:
        dags (list): List of directed acyclic graphs (DAGs) representing causal relationships.
        observations (list): List of observations (pd.DataFrame) corresponding to each DAG.
        n_variables (int, optional): Number of variables in the time series. Defaults to 3.
        maxlags (int, optional): Maximum number of lags in the time series. Defaults to 3.
        mutual_information_proxy (str, optional): Method to use for mutual information computation. Defaults to "Ridge".
        proxy_params (dict, optional): Parameters for the mutual information computation. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        n_jobs (int, optional): Number of parallel jobs to run. Defaults to 1.

    Attributes:
        DAGs (list): List of DAGs representing causal relationships.
        dag_to_observation (dict): Mapping of DAG index to corresponding observation.
        x_y (None): Placeholder for computed descriptors.
        n_variables (int): Number of variables in the time series.
        maxlags (int): Maximum number of lags in the time series.
        test_couples (list): List of couples for which descriptors have been computed.
        mutual_information_proxy (str): Method used for mutual information computation.
        proxy_params (dict): Parameters for the mutual information computation.
        family (dict): Family of descriptors to compute.
        verbose (bool): Whether to print verbose output.
        n_jobs (int): Number of parallel jobs to run.
        seed (int): Random seed for reproducibility.
    """
    def __init__(self, 
                 dags, 
                 observations, 
                 couples_to_consider_per_dag = 20, 
                 MB_size = 5, 
                 n_variables=3, 
                 maxlags=3, 
                 mutual_information_proxy="Ridge", 
                 proxy_params=None,
                 full = False, 
                 quantiles = True,
                 normalize=False,
                 verbose=False, 
                 cmi = 'cmiknn_3',
                 mb_estimator = 'original',
                 seed= 42, 
                 n_jobs=1,
                 top_vars = 3) -> None:
        
        self.DAGs = dags
        self.observations = observations
        self.couples_to_consider_per_dag = couples_to_consider_per_dag
        self.n_variables = n_variables 
        self.maxlags = maxlags 
        self.mutual_information_proxy = mutual_information_proxy  
        self.proxy_params = proxy_params
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.seed = seed
        self.cmi = cmi
        self.normalize = normalize
        self.top_vars = top_vars

        self.x_y = None # Placeholder for computed descriptors, list of dictionaries
        self.test_couples = []  # List of couples for which descriptors have been computed

        self.markov_blanket_estimator = MarkovBlanketEstimator(size=min(MB_size, n_variables - 2), n_variables = n_variables, maxlags=maxlags, top_vars = self.top_vars)

        self.mb_estimator = mb_estimator
        self.mutual_information_estimator = MutualInformationEstimator(proxy=mutual_information_proxy, proxy_params=proxy_params, k=int(self.cmi.split('_')[-1]) if self.cmi.startswith('cmiknn') else None)

        self.quantiles = quantiles
        self.full = full

        np.random.seed(seed)

    def initialize(self) -> None:
        """
        Initialize the D2C object by computing descriptors in parallel for all observations.

        """
        if self.couples_to_consider_per_dag == -1:
            num_samples = -1
        else:
            num_samples = self.couples_to_consider_per_dag // 3 # because 1/3 is causal A->B, 1/3 is B->A, 1/3 is other non-causal that respect time 
        
        if self.n_jobs == 1:
            results = [self.compute_descriptors_with_dag(dag_idx, dag, self.n_variables, self.maxlags, num_samples=num_samples) for dag_idx, dag in enumerate(self.DAGs)]

        else:
            args = [(dag_idx, dag, self.n_variables, self.maxlags, num_samples) for dag_idx, dag in enumerate(self.DAGs)]
            with Pool(processes=self.n_jobs) as pool:
                results = pool.starmap(self.compute_descriptors_with_dag, args)

        #merge lists into a single list
        results = [item for sublist in results for item in sublist]
        self.x_y = results

    def compute_descriptors_without_dag(self, n_variables, maxlags) -> list:
        """ 
        Compute all descriptors when a directed acyclic graph (DAG) is not available.
        This is useful for real testing data, but not for synthetic training data.
        So far it's one D2C object per synthetic dataset, so we don't need to pass the DAGs. 
        We only have one set of observations, so we place them as if the dag index was 0.
        Synthetic labeled data is handled by the compute_descriptors_with_dag method.
        TODO: not clear, refactor, remove the dag_idx=0
        """

        all_possible_links = {(i, j) for i in range(n_variables, n_variables + n_variables * maxlags) for j in range(n_variables) if i != j}

        if self.n_jobs == 1:
            results = [self.compute_descriptors_for_couple(dag_idx=0, ca=a, ef=b, label=np.nan) for a, b in all_possible_links]

        else:
            args = [(0, a, b, np.nan) for a, b in all_possible_links]
            with Pool(processes=self.n_jobs) as pool:
                results = pool.starmap(self.compute_descriptors_for_couple, args)

        return pd.DataFrame(results)

    def compute_descriptors_with_dag(self, dag_idx, dag, n_variables, maxlags, num_samples=20) -> list:
        """
        Compute all descriptors associated to a directed acyclic graph (DAG).
        This is useful for synthetic training data, but not for real testing data.
        In this method, we can select specific couples based on their nature (causal or non-causal).
        Real unlabeled data is handled by the compute_descriptors_without_dags method.

        Args:
            dag (networkx.DiGraph): The directed acyclic graph.
            n_variables (int): The number of variables in the graph.
            maxlags (int): The maximum number of lags.
            num_samples (int, optional): The number of samples to consider. Defaults to 20.

        Returns:
            List of couples contains the computed descriptors, and the labels (1 for causal links, 0 for non-causal links).
        """

        x_y_couples = []

        all_possible_links = {(i, j) for i in range(n_variables, n_variables + n_variables * maxlags) for j in range(n_variables) if i != j}

        causal_links = list(set([(int(parent), int(child)) for parent, child in dag.edges]).intersection(all_possible_links))
        non_causal_links = list(all_possible_links - set(causal_links))

        if num_samples == -1:
            for parent, child in causal_links:
                x_y_couples.append(self.compute_descriptors_for_couple(dag_idx, parent, child, label=1)) # causal
            for node_a, node_b in non_causal_links:
                x_y_couples.append(self.compute_descriptors_for_couple(dag_idx, node_a, node_b, label=0)) # noncausal, time ordered

            self.test_couples.extend(causal_links)
            self.test_couples.extend(non_causal_links)
        
        else: 

            subset_causal_links = np.random.permutation(causal_links)[:min(len(causal_links), num_samples)].astype(int)
            subset_non_causal_links = np.random.permutation(non_causal_links)[:min(len(non_causal_links), num_samples)].astype(int)

            # dag_idx = dag.graph['index']

            for parent, child in subset_causal_links:
                x_y_couples.append(self.compute_descriptors_for_couple(dag_idx, parent, child, label=1)) # causal
                x_y_couples.append(self.compute_descriptors_for_couple(dag_idx, child, parent, label=0)) # noncausal, not time ordered (yet informative)
            for node_a, node_b in subset_non_causal_links:
                x_y_couples.append(self.compute_descriptors_for_couple(dag_idx, node_a, node_b, label=0)) # noncausal, time ordered

            self.test_couples.extend(subset_causal_links)
            self.test_couples.extend(subset_non_causal_links)
        
        return x_y_couples

    def get_markov_blanket(self, dag, node):
        """
        Computes the REAL Markov Blanket of a node in a specific DAG.
        It uses the dag structure, therefore this is not an estimate.
        This is not applicable in realistic settings, because it won't
        be possible to use it on observational data (unknown dag).

        Parameters:
        dag (networkx.DiGraph): The DAG.
        node (int): The node index.

        Returns:
        list: The Markov Blanket of the node.
        """
        parents = list(dag.predecessors(node))
        children = list(dag.successors(node))
        parents_of_children = []
        for child in children:
            parents_of_children.extend(list(dag.predecessors(child)))
        parents_of_children = list(set(parents_of_children))

        return list(set(parents + parents_of_children + children))

    def standardize_data(self, observations):
        """Standardizes the observation DataFrame."""
        return (observations - observations.mean()) / observations.std()

    def check_data_validity(self, observations):
        """
        For synthetic data, check is performed at data creation only. 
        For real data, check is performed at the beginning of the process, not at each iteration.
        
        Deprecated: This function is not used anymore, as it is not necessary to check the data at each iteration.
        TODO: perform this check only at the beginning of the process.
        """
        if np.any(np.isnan(observations)) or np.any(np.isinf(observations)):
            raise ValueError("Error: NA or Inf in data")

    def update_dictionary_quantiles(self, dictionary, name, quantiles):
        """
        Update the given dictionary with quantiles.

        Args:
            dictionary (dict): The dictionary to update.
            name (str): The name of the quantiles.
            quantiles (list): A list of quantiles to add to the dictionary.

        Returns:
            None
        """
        # Add quantiles to the dictionary
        for i, q in enumerate(quantiles):
            dictionary[f'{name}_q{i}'] = q

    def update_dictionary_distribution(self, dictionary, name, values):
        """
        Update the given dictionary moments of the distribution.

        Args:
            dictionary (dict): The dictionary to update.
            name (str): The name of the quantiles.
            values (list): A list of values to compute the distirbution moments.

        Returns:
            None
        """
        # from scipy.stats import kurtosis, skew
        dictionary[f'{name}_mean'] = np.mean(values)
        dictionary[f'{name}_std'] = np.std(values)
        # dictionary[f'{name}_skew'] = skew(values)
        # dictionary[f'{name}_kurtosis'] = kurtosis(values)

    def update_dictionary_actual_values(self, dictionary, name, values):
        """
        Update the given dictionary with actual values. No quantiles, no moments, just the actual descriptors.

        Args:
            dictionary (dict): The dictionary to update.
            name (str): The name of the quantiles.
            values (list): A list of values to compute the distirbution moments.

        Returns:
            None
        """
        for i, q in enumerate(values):
            dictionary[f'{name}_{i}'] = q

    def compute_descriptors_for_couple(self, dag_idx, ca, ef, label): # , sp?
        """
        Compute descriptors for a given couple of nodes in a directed acyclic graph (DAG).

        Args:
            dag_idx (int): The index of the DAG.
            ca (int): The index of the cause node.
            ef (int): The index of the effect node.
            sp (int): The index of the spouse node.
            label (bool): The label indicating whether the edge between the cause, effect and spouse nodes is causal.

        Returns:
            dict: A dictionary containing the computed descriptors.

        """
        # pq=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        pq=[0.25, 0.5, 0.75]

        if self.normalize:
            observations = self.standardize_data(self.observations[dag_idx])
        else:
            observations = self.observations[dag_idx]

        if self.mb_estimator=='original':
            MBca = self.markov_blanket_estimator.estimate(observations, node=ca)
            MBef = self.markov_blanket_estimator.estimate(observations, node=ef)
#           Mbsp = self.markov_blanket_estimator.estimate(observations, node=sp) 
        elif self.mb_estimator=='ts':
            MBca = self.markov_blanket_estimator.estimate_time_series(observations, node=ca)
            MBef = self.markov_blanket_estimator.estimate_time_series(observations, node=ef)
#           Mbsp = self.markov_blanket_estimator.estimate_time_series(observations, node=sp)
        elif self.mb_estimator=='ts_rank':
            MBca = self.markov_blanket_estimator.estimate_time_series_ranking(observations, node=ca)
            MBef = self.markov_blanket_estimator.estimate_time_series_ranking(observations, node=ef)
#           Mbsp = self.markov_blanket_estimator.estimate_time_series(observations, node=sp)
            
        common_causes_eff = list(set(MBca).intersection(MBef))
        mbca_mbef_couples = [(i, j) for i in range(len(MBca)) for j in range(len(MBef))]
        mbca_mbca_couples = [(i, j) for i in range(len(MBca)) for j in range(len(MBca)) if i != j]
        mbef_mbef_couples = [(i, j) for i in range(len(MBef)) for j in range(len(MBef)) if i != j]

#       common_causes_sp = list(set(MBca).intersection(MBsp))
#       mbca_mbsp_couples = [(i, j) for i in range(len(MBca)) for j in range(len(MBsp))]
        # mbca_mbca_couples = [(i, j) for i in range(len(MBca)) for j in range(len(MBca)) if i != j]
        # mbsp_mbsp_couples = [(i, j) for i in range(len(MBsp)) for j in range(len(MBsp)) if i != j]

        
        # e, c = observations[:, ef], observations[:, ca] #aliases 'e' and 'c' for brevity
        e,c  = ef,ca
        # s = sp
        if 'cmiknn' in self.cmi:
            CMI = self.mutual_information_estimator.estimate_knn_cmi # alias for mutual information estimator function, for brevity
        elif self.cmi == 'original':
            CMI = self.mutual_information_estimator.estimate_original

        values = {}
        values['graph_id'] = dag_idx
        values['edge_source'] = ca
        values['edge_dest'] = ef
#       values['edge_spouse'] = sp
        values['is_causal'] = label

        # b: ef = b * (ca + mbef)
        values['coeff_cause'] = coeff(observations[:, e], observations[:, c], observations[:, MBef])

        # b: ca = b * (ef + mbca)
        values['coeff_eff'] = coeff(observations[:, c], observations[:, e], observations[:, MBca])

        # b: add spouses?
        

        values['HOC_3_1'] = HOC(observations[:, c], observations[:, e], 3, 1)
        values['HOC_1_2'] = HOC(observations[:, c], observations[:, e], 1, 2)
        values['HOC_2_1'] = HOC(observations[:, c], observations[:, e], 2, 1)
        values['HOC_1_3'] = HOC(observations[:, c], observations[:, e], 1, 3)
        # add spouses?

        values['kurtosis_ca'] = kurtosis(observations[:, c])
        values['kurtosis_ef'] = kurtosis(observations[:, e])
        # add spouses?


        # I(mca ; mef | cause) for (mca,mef) in mbca_mbef_couples
        # mca_mef_cau = [0] if not len(mbca_mbef_couples) else [CMI(observations[:,i], observations[:,j], c) for i, j in mbca_mbef_couples]
        mca_mef_cau = [0] if not len(mbca_mbef_couples) else [CMI(observations, i,j,c) for i, j in mbca_mbef_couples]
        if self.quantiles: self.update_dictionary_quantiles(values, 'mca_mef_cau', np.quantile(mca_mef_cau, pq))
        else: self.update_dictionary_actual_values(values, 'mca_mef_cau', mca_mef_cau)

        # I(mca ; mef| effect) for (mca,mef) in mbca_mbef_couples
        # mca_mef_eff = [0] if not len(mbca_mbef_couples) else [CMI(observations[:,i], observations[:,j], e) for i, j in mbca_mbef_couples]
        mca_mef_eff = [0] if not len(mbca_mbef_couples) else [CMI(observations, i,j, e) for i, j in mbca_mbef_couples]
        if self.quantiles: self.update_dictionary_quantiles(values, 'mca_mef_eff', np.quantile(mca_mef_eff, pq))
        else: self.update_dictionary_actual_values(values, 'mca_mef_eff', mca_mef_eff)

        # I(cause; m | effect) for m in MBef
        # cau_m_eff = [0] if not len(MBef) else [CMI(c, observations[:, m], e) for m in MBef]
        cau_m_eff = [0] if not len(MBef) else [CMI(observations, c, m, e) for m in MBef]
        if self.quantiles: self.update_dictionary_quantiles(values, 'cau_m_eff', np.quantile(cau_m_eff, pq))
        else: self.update_dictionary_actual_values(values, 'cau_m_eff', cau_m_eff)

        # I(effect; m | cause) for m in MBca
        # eff_m_cau = [0] if not len(MBca) else [CMI(e, observations[:, m], c) for m in MBca]
        eff_m_cau = [0] if not len(MBca) else [CMI(observations,e, m, c) for m in MBca]
        if self.quantiles: self.update_dictionary_quantiles(values, 'eff_m_cau', np.quantile(eff_m_cau, pq))
        else: self.update_dictionary_actual_values(values, 'eff_m_cau', eff_m_cau)

        # I(msp; mca | effect) for (mca,msp) in mbca_mbsp_couples ??
        # mca_msp_eff = [0] if not len(mbca_mbsp_couples) else [CMI(observations[:,i], observations[:,j], e) for i, j in mbca_mbsp_couples]
#       mca_msp_eff = [0] if not len(mbca_mbsp_couples) else [CMI(observations, i,j, e) for i, j in mbca_mbsp_couples]
#       if self.quantiles: self.update_dictionary_quantiles(values, 'mca_msp_eff', np.quantile(mca_msp_eff, pq))
#       else: self.update_dictionary_actual_values(values, 'mca_msp_eff', mca_msp_eff)


        if self.full:

            # I(m; cause) for m in MBef
            # m_cau = [0] if not len(MBef) else [CMI(c, observations[:, m]) for m in MBef]
            m_cau = [0] if not len(MBef) else [CMI(observations, c, m) for m in MBef]
            if self.quantiles: self.update_dictionary_quantiles(values, 'm_cau', np.quantile(m_cau, pq))
            else: self.update_dictionary_actual_values(values, 'm_cau', m_cau)

            # I(cause; effect | common_causes)
            # values['com_cau'] = CMI(e, c, observations[:, common_causes])
            values['com_cau'] = CMI(observations, e, c, common_causes_eff)

            # I(cause; effect)
            # values['cau_eff'] = CMI(e, c)
            values['cau_eff'] = CMI(observations, e, c)

            # I(effect; cause)
            # values['eff_cau'] = CMI(c, e)
            values['eff_cau'] = CMI(observations, c, e)

            # I(effect; cause | MBeffect)
            # values['eff_cau_mbeff'] = CMI(c, e, observations[:, MBef])
            values['eff_cau_mbeff'] = CMI(observations, c, e, MBef)

            # I(cause; effect | MBcause)
            # values['cau_eff_mbcau'] = CMI(e, c, observations[:, MBca])
            values['cau_eff_mbcau'] = CMI(observations, e, c, MBca)

            # I(effect; cause | arrays_m_plus_MBca)
            # eff_cau_mbcau_plus = [0] if not len(MBef) else [CMI(c, e, observations[:,np.unique(np.concatenate(([m], MBca)))]) for m in MBef]
            eff_cau_mbcau_plus = [0] if not len(MBef) else [CMI(observations, c, e, np.unique(np.concatenate(([m], MBca)))) for m in MBef]
            if self.quantiles: self.update_dictionary_quantiles(values, 'eff_cau_mbcau_plus', np.quantile(eff_cau_mbcau_plus, pq))
            else: self.update_dictionary_actual_values(values, 'eff_cau_mbcau_plus', eff_cau_mbcau_plus)
            
            # I(cause; effect | arrays_m_plus_MBef)
            # cau_eff_mbeff_plus = [0] if not len(MBca) else [CMI(e, c, observations[:,np.unique(np.concatenate(([m], MBef)))]) for m in MBca]
            cau_eff_mbeff_plus = [0] if not len(MBca) else [CMI(observations, e, c, np.unique(np.concatenate(([m], MBef)))) for m in MBca]
            if self.quantiles: self.update_dictionary_quantiles(values, 'cau_eff_mbeff_plus', np.quantile(cau_eff_mbeff_plus, pq))
            else: self.update_dictionary_actual_values(values, 'cau_eff_mbeff_plus', cau_eff_mbeff_plus)

            # I(m; effect) for m in MBca
            # m_eff = [0] if not len(MBca) else [CMI(e, observations[:, m]) for m in MBca]
            m_eff = [0] if not len(MBca) else [CMI(observations, e, m) for m in MBca]
            if self.quantiles: self.update_dictionary_quantiles(values, 'm_eff', np.quantile(m_eff, pq))
            else: self.update_dictionary_actual_values(values, 'm_eff', m_eff)

            #I(mca ; mca| cause) - I(mca ; mca) for (mca,mca) in mbca_couples
            # mca_mca_cau = [0] if not len(mbca_mbca_couples) else [CMI(observations[:,i], observations[:,j], c) - CMI(observations[:,i], observations[:,j]) for i, j in mbca_mbca_couples]
            mca_mca_cau = [0] if not len(mbca_mbca_couples) else [CMI(observations,i,j, c) - CMI(observations,i,j) for i, j in mbca_mbca_couples]
            if self.quantiles: self.update_dictionary_quantiles(values, 'mca_mca_cau', np.quantile(mca_mca_cau, pq))
            else: self.update_dictionary_actual_values(values, 'mca_mca_cau', mca_mca_cau)

            # I(mbe ; mbe| effect) - I(mbe ; mbe) for (mbe,mbe) in mbef_couples
            # mbe_mbe_eff = [0] if not len(mbef_mbef_couples) else [CMI(observations[:,i], observations[:,j], e) - CMI(observations[:,i], observations[:,j]) for i, j in mbef_mbef_couples]
            mbe_mbe_eff = [0] if not len(mbef_mbef_couples) else [CMI(observations, i,j, e) - CMI(observations,i,j) for i, j in mbef_mbef_couples]
            if self.quantiles: self.update_dictionary_quantiles(values, 'mbe_mbe_eff', np.quantile(mbe_mbe_eff, pq))
            else: self.update_dictionary_actual_values(values, 'mbe_mbe_eff', mbe_mbe_eff)

            # I(mbca ; mbsp| effect) for (mbca,mbsp) in mbca_mbsp_couples ??
            # mbca_mbsp_eff = [0] if not len(mbca_mbsp_couples) else [CMI(observations[:,i], observations[:,j], e) for i, j in mbca_mbsp_couples]
#           mbca_mbsp_eff = [0] if not len(mbca_mbsp_couples) else [CMI(observations, i,j, e) for i, j in mbca_mbsp_couples]
#           if self.quantiles: self.update_dictionary_quantiles(values, 'mbca_mbsp_eff', np.quantile(mbca_mbsp_eff, pq))
#           else: self.update_dictionary_actual_values(values, 'mbca_mbsp_eff', mbca_mbsp_eff)

            values['n_samples'] = observations.shape[0]
            values['n_features'] = observations.shape[1]
            values['n_features/n_samples'] = observations.shape[1] / observations.shape[0]
            values['skewness_ca'] = skew(observations[:, c])
            values['skewness_ef'] = skew(observations[:, e])

        return values

    def get_descriptors_df(self) -> pd.DataFrame:
        """
        Get the concatenated DataFrame of X and Y.

        Returns:
            pd.DataFrame: The concatenated DataFrame of X and Y.

        """
        return pd.DataFrame(self.x_y)

    def get_test_couples(self):
        return self.test_couples

# class for iterative TD2C

# class for TD2C helping

# class for competitors helping