
# PACKAGE IMPORTS #################################################################################
import pandas as pd
from itertools import permutations
from tqdm import tqdm
import pickle 
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_auc_score

from d2c.descriptors_generation import D2C, DataLoader


class IterativeTD2C():
    """
    This class contains the function iterative_td2c() that performs the iterative TD2C method and the functions plot_results() 
    and df_scores() to plot the results and save the ROC-AUC scores in a csv file.
    
    This function requires data already generated and stored in the data folder

    Parameters:
    - k: is the number of top variables to keep at each iteration for each DAG (keep = 1 if treshold = False)
    - it: is the limit for the number of iterations to perform
    - top_vars: is the number of top variables to keep in case of TD2C Ranking
    - treshold: is a boolean to determine if we want to keep in the causal df the variables with a pred.proba higher than treshold_value
    - treshold_value: is the value to keep the variables in the causal df
    - size_causal_df: is the number of variables to keep in the causal_df in case of treshold = False
    - data_folder: is the folder where the data is stored
    - descr_folder: is the folder where the descriptors are stored
    - results_folder: is the folder where the results are stored
    - COUPLES_TO_CONSIDER_PER_DAG: is the number of couples to consider for each DAG
    - maxlags: is the maximum number of lags to consider
    - SEED: is the seed for the random state
    - MB_SIZE: is the size of the Markov Blanket
    - max_neighborhood_size_filter: is the maximum neighborhood size to consider
    - noise_std_filter: is the noise standard deviation to consider
    - N_JOBS: is the number of jobs to run in parallel
    - adaptive: determines if the size of the causal_df is adjusted based on the comparison of ROC scores
    - adaptive_mode: is the mode for the adaptive parameter
        . "Adding": to add 1 to the size of the causal_df if the ROC score is lower than the previous one
        . "Subtracting": to subtract 1 to the size of the causal_df if the ROC score is lower than the previous one
        . "Balancing": to add 1 to the size of the causal_df if the ROC score is lower than the previous one and subtract 1 if the ROC score is higher than the previous one
    - performative: determines if the size of the causal_df is adjusted based on the comparison of ROC scores
    - performative_mode: is the mode for the performative parameter
        . "More_Less": to stop when the ROC score is > 0.05 then the first one and when the ROC score is < 0.05 then the first one
        . "More": to stop when the avg. ROC score is > then the first one
        . "Tail": to stop after self.it iterations, if the ROC score is < the first one, goes to the highest ROC score, adds 1 to the size of the causal_df and goes for other self.it/2 iterations
    - arbitrary: determines if the size of the causal_df is adjusted based on the comparison of ROC scores
    - arbitrary_mode: is the mode for the arbitrary parameter
        . "Increasing": to add 1 at each iteration
        . "Decreasing": to subtract 1 at each iteration
        . "Common": Keeps only the edges with pred.prob > 0.7 and considered by at least 3 DAGs
    - Methods:
        . 'ts' = for classic TD2C
        . 'original' = for original D2C
        . 'ts_rank' = for TD2C with ranking
        . 'ts_rank_2' = for TD2C with ranking 2
        . 'ts_rank_3' = for TD2C with ranking 3
        . 'ts_rank_4' = for TD2C with ranking 4
        . 'ts_rank_5' = for TD2C with ranking 5
        . 'ts_rank_6' = for TD2C with ranking 6
        . 'ts_rank_7' = for TD2C with ranking 7
        . 'ts_past' = for TD2C with past and future nodes
        . 'ts_rank_no_count' = for TD2C with ranking with no contemporaneous nodes

    Stopping Criteria:
    1. if average ROC-AUC score does not improve or is the same as the previous iteration for 5 consecutive iterations
    2. if the first iteration has an average ROC-AUC score lower than 0.5
    3. if the average ROC-AUC score is more than 0.2 points lower than the first iteration
    4. if causal df is the same as the previous one for 3 consecutive iterations

    Output:
    1. Plot of average ROC-AUC scores (saved in results folder as pdf file)
    2. Average ROC-AUC scores for each iteration (saved in results folder as csv file)
    """

    def __init__(self, method = 'ts', k = 1, it = 6, top_vars = 3, treshold = False, N_JOBS = 50, noise_std_filter = 0.01,
                 treshold_value = 0.9, size_causal_df = 3, COUPLES_TO_CONSIDER_PER_DAG = -1, arbitrary = False, 
                 performative = False, adaptive = False, adaptive_mode = "Balancing", performative_mode = "More_less", 
                 arbitrary_mode = "Increasing", maxlags = 5, SEED = 42, MB_SIZE = 2, max_neighborhood_size_filter = 2, 
                 data_folder = 'home/data/', descr_folder = 'home/descr/', results_folder = 'home/results/'):
        
        self.method = method
        self.k = k
        self.it = it
        self.top_vars = top_vars
        self.treshold = treshold
        self.treshold_value = treshold_value
        self.size_causal_df = size_causal_df
        self.data_folder = data_folder
        self.descr_folder = descr_folder
        self.results_folder = results_folder
        self.COUPLES_TO_CONSIDER_PER_DAG = COUPLES_TO_CONSIDER_PER_DAG
        self.maxlags = maxlags
        self.SEED = SEED
        self.MB_SIZE = MB_SIZE
        self.max_neighborhood_size_filter = max_neighborhood_size_filter
        self.noise_std_filter = noise_std_filter
        self.N_JOBS = N_JOBS
        self.adaptive = adaptive
        self.adaptive_mode = adaptive_mode
        self.performative = performative
        self.performative_mode = performative_mode
        self.arbitrary = arbitrary
        self.arbitrary_mode = arbitrary_mode

        np.random.seed(self.SEED)

    def param_check(self):
        # verify parameters
        if self.method == None or self.k == None or self.it == None or self.top_vars == None or self.treshold == None or self.treshold_value == None or self.size_causal_df == None or self.data_folder == None or self.descr_folder == None or self.results_folder == None or self.COUPLES_TO_CONSIDER_PER_DAG == None or self.maxlags == None or self.SEED == None or self.MB_SIZE == None or self.max_neighborhood_size_filter == None or self.noise_std_filter == None or self.N_JOBS == None or self.adaptive == None:
            print('Please provide the correct parameters')
            return
        if self.method not in ['ts', 'original', 'ts_rank', 'ts_rank_2', 'ts_rank_3', 'ts_rank_4', 'ts_rank_5', 'ts_rank_6', 'ts_rank_7', 'ts_past', 'ts_rank_no_count']:
            print('Please provide the correct method')
            return
        if self.k < 1:
            print('Please provide a value greater than 0 for k')
            return
        if self.it < 1:
            print('Please provide a value greater than 0 for it')
            return
        if self.top_vars < 1:
            print('Please provide a value greater than 0 for top_vars')
            return
        if self.treshold not in [True, False]:
            print('Please provide a boolean value for treshold')
            return
        if self.treshold == True and self.treshold_value < 0.5:
            print('Please provide a value greater than 0.5 for treshold_value')
            return
        if self.size_causal_df < 1:
            print('Please provide a value greater than 0 for size_causal_df')
            return
        if self.data_folder == None:
            print('Please provide the correct data_folder')
            return
        if self.descr_folder == None:
            print('Please provide the correct descr_folder')
            return
        if self.results_folder == None:
            print('Please provide the correct results_folder')
            return
        if self.COUPLES_TO_CONSIDER_PER_DAG < 5 and self.COUPLES_TO_CONSIDER_PER_DAG != -1:
            print('Please provide a value greater than 5 for COUPLES_TO_CONSIDER_PER_DAG')
            return
        if self.maxlags < 1:
            print('Please provide a value greater than 0 for maxlags')
            return
        if self.MB_SIZE < 1:
            print('Please provide a value greater than 0 for MB_SIZE')
            return
        if self.max_neighborhood_size_filter < 1:
            print('Please provide a value greater than 0 for max_neighborhood_size_filter')
            return
        if self.noise_std_filter < 0:
            print('Please provide a value greater than 0 for noise_std_filter')
            return
        if self.N_JOBS < 1:
            print('Please provide a value greater than 0 for N_JOBS')
            return
        if self.adaptive not in ['Adding', 'Subtracting', 'Balancing', False,  True]:
            print('Please provide the correct value for adaptive')
            return
        if self.performative not in ['More_Less', 'More', 'Tail', False, True]:
            print('Please provide the correct value for performative')
            return
        if self.arbitrary not in ['Increasing', 'Decreasing', 'Common', False, True]:
            print('Please provide the correct value for arbitrary')
            return
        if self.adaptive == 'Adding' and self.size_causal_df  > 5:
            print('Please provide a value less than 5 for size_causal_df if adaptive is set to Adding')
            return
        if self.adaptive == 'Subtracting' and self.size_causal_df  < 3:
            print('Please provide a value greater than 3 for size_causal_df if adaptive is set to Subtracting')
            return
        if self.arbitrary == 'Increasing' and self.size_causal_df  > 5:
            print('Please provide a value less than 5 for size_causal_df if arbitrary is set to Increasing')
            return
        if self.arbitrary == 'Decreasing' and self.size_causal_df  < 5:
            print('Please provide a value greater than 5 for size_causal_df if arbitrary is set to Decreasing')
            return
        # if any combination of adaptive, performative, arbitrary and trheshold is True stop the function
        if self.adaptive == True and self.performative == True:
            print('Please provide only one parameter between adaptive and performative')
            return
        if self.adaptive == True and self.arbitrary == True:
            print('Please provide only one parameter between adaptive and arbitrary')
            return
        if self.adaptive == True and self.treshold == True:
            print('Please provide only one parameter between adaptive and treshold')
            return
        if self.performative == True and self.arbitrary == True:
            print('Please provide only one parameter between performative and arbitrary')
            return
        if self.performative == True and self.treshold == True:
            print('Please provide only one parameter between performative and treshold')
            return
        if self.arbitrary == True and self.treshold == True:
            print('Please provide only one parameter between arbitrary and treshold')
            return
        if self.adaptive == True and self.performative == True and self.arbitrary == True and self.treshold == True:
            print('Please provide only one parameter between adaptive, performative, arbitrary and treshold')
            return
        if self.adaptive == True and self.performative == True and self.arbitrary == True:
            print('Please provide only one parameter between adaptive, performative and arbitrary')
            return
        if self.adaptive == True and self.performative == True and self.treshold == True:
            print('Please provide only one parameter between adaptive, performative and treshold')
            return
        if self.adaptive == True and self.arbitrary == True and self.treshold == True:
            print('Please provide only one parameter between adaptive, arbitrary and treshold')
            return
        if self.performative == True and self.arbitrary == True and self.treshold == True:
            print('Please provide only one parameter between performative, arbitrary and treshold')
            return
        if self.adaptive == True and self.performative == True and self.arbitrary == True and self.treshold == True:
            print('Please provide only one parameter between adaptive, performative, arbitrary and treshold')
            return

    def start(self):

        # Description of the iteration
        print()
        print(f'Iterative TD2C - Method: {self.method} - Max iterations: {self.it} - Variables to keep per DAG: {self.k} - Top Variables: {self.top_vars} - Treshold: {self.treshold} - Size of Causal DF: {self.size_causal_df}')
        print()
    
        # Computational time estimation
        if self.performative == True and self.performative_mode == "More_Less":
            print('This iteration could last forefer...')
        elif self.performative == True and self.performative_mode == "Tail":
            if self.COUPLES_TO_CONSIDER_PER_DAG == -1 and self.treshold == False:
                print('Using all couples for each DAG')
                print(f'This iteration will take approximately {8.5*self.it + 0.3*(8.5*self.it)} minutes')
                print()
            elif self.COUPLES_TO_CONSIDER_PER_DAG == -1 and self.treshold == True:
                print(f'Using all couples for each DAG and a pred.proba higher than {self.treshold_value}')
                print(f'This iteration will take approximately {10.5*self.it + 0.3*(10.5*self.it)} minutes')
                print()
            elif self.COUPLES_TO_CONSIDER_PER_DAG != -1 and self.size_causal_df == 5:
                print(f'Using the top {self.COUPLES_TO_CONSIDER_PER_DAG} couples for each DAG')
                print(f'This iteration will take approximately {4*self.it + 0.3*(4*self.it)} minutes')
                print()
            elif self.COUPLES_TO_CONSIDER_PER_DAG != -1 and self.size_causal_df < 5:
                print(f'Using the top {self.COUPLES_TO_CONSIDER_PER_DAG} couples for each DAG')
                print(f'This iteration will take approximately {3.5*self.it + 0.3*(3.5*self.it)} minutes')
                print()
            elif self.COUPLES_TO_CONSIDER_PER_DAG != -1 and self.size_causal_df > 5:
                print(f'Using the top {self.COUPLES_TO_CONSIDER_PER_DAG} couples for each DAG')
                print(f'This iteration will take approximately {4.5*self.it + 0.3*(4.5*self.it)} minutes')
                print()
        else:
            if self.COUPLES_TO_CONSIDER_PER_DAG == -1 and self.treshold == False:
                print('Using all couples for each DAG')
                print(f'This iteration will take approximately {8.5*self.it} minutes')
                print()
            elif self.COUPLES_TO_CONSIDER_PER_DAG == -1 and self.treshold == True:
                print(f'Using all couples for each DAG and a pred.proba higher than {self.treshold_value}')
                print(f'This iteration will take approximately {10.5*self.it} minutes')
                print()
            elif self.COUPLES_TO_CONSIDER_PER_DAG != -1 and self.size_causal_df == 5:
                print(f'Using the top {self.COUPLES_TO_CONSIDER_PER_DAG} couples for each DAG')
                print(f'This iteration will take approximately {4*self.it} minutes')
                print()
            elif self.COUPLES_TO_CONSIDER_PER_DAG != -1 and self.size_causal_df < 5:
                print(f'Using the top {self.COUPLES_TO_CONSIDER_PER_DAG} couples for each DAG')
                print(f'This iteration will take approximately {3.5*self.it} minutes')
                print()
            elif self.COUPLES_TO_CONSIDER_PER_DAG != -1 and self.size_causal_df > 5:
                print(f'Using the top {self.COUPLES_TO_CONSIDER_PER_DAG} couples for each DAG')
                print(f'This iteration will take approximately {4.5*self.it} minutes')
                print()

        return

    def strategy(self):

        # Description of the method used
        if self.performative == True and self.performative_mode == "More_Less":
            print('Method: Performative - Mode: More_Less')
            print()
            strategy = "Performative-More_Less"
        elif self.performative == True and self.performative_mode == "More":
            print('Method: Performative - Mode: More')
            print()
            strategy = "Performative-More"
        elif self.performative == True and self.performative_mode == "Tail":
            print('Method: Performative - Mode: Tail')
            print()
            strategy = "Performative-Tail"
        elif self.adaptive == True and self.adaptive_mode == "Adding":
            print('Method: Adaptive - Mode: Adding')
            print()
            strategy = "Adaptive-Adding"
        elif self.adaptive == True and self.adaptive_mode == "Subtracting":
            print('Method: Adaptive - Mode: Subtracting')
            print()
            strategy = "Adaptive-Subtracting"
        elif self.adaptive == True and self.adaptive_mode == "Balancing1":
            print('Method: Adaptive - Mode: Balancing1')
            print()
            strategy = "Adaptive-Balancing1"
        elif self.adaptive == True and self.adaptive_mode == "Balancing2":
            print('Method: Adaptive - Mode: Balancing2')
            print()
            strategy = "Adaptive-Balancing2"
        elif self.arbitrary == True and self.arbitrary_mode == "Increasing":
            print('Method: Arbitrary - Mode: Increasing')
            print()
            strategy = "Arbitrary-Increasing"
        elif self.arbitrary == True and self.arbitrary_mode == "Decreasing":
            print('Method: Arbitrary - Mode: Decreasing')
            print()
            strategy = "Arbitrary-Decreasing"
        elif self.arbitrary == True and self.arbitrary_mode == "Common":
            print('Method: Arbitrary - Mode: Common')
            print()
            strategy = "Arbitrary-Common"
        else:
            print('Method: Classic')
            print()
            strategy = "Classic"
        
        return strategy

    def response(self):
        # Start the iteration?
        if self.it < 6:
            if self.performative == True and self.performative_mode == "Tail":
                print()
                print('########################################################################################')
                print("  WARNING: The number of iterations is less than 6. The results might not be accurate.  ")
                print('########################################################################################')
                print()
                response = input("Do you want to continue with the rest of the function? (y/n): ").strip().lower()
                print()
            else:
                print()
                response = input("Do you want to continue with the rest of the function? (y/n): ").strip().lower()
                print()
        else:
            response = input("Do you want to continue with the rest of the function? (y/n): ").strip().lower()
            print()

        return response

    def iterative_td2c(self, response):

        if response in ['yes', 'y', 'Yes', 'Y']:
            print()
            print("Ok! Let's start the iteration.")
            print()

            stop_1 = 0
            stop_2 = 0
            roc_scores = []
            causal_df_unified = {}
            causal_df_mid = {}
            causal_df_permut = []
            roc_0 = 0
            tail = 0
            operation = ""
            operation_reg = []
            if self.performative == True and self.performative_mode == "More_Less":
                self.it = 1000
            elif self.performative == True and self.performative_mode == "More":
                self.it = 1000
            elif self.performative == True and self.performative_mode == "Tail":
                it_tail = self.it
                if self.it < 6:
                    self.it = self.it + 3
                else:
                    self.it = self.it + int(self.it/3)
            np.random.seed(self.SEED)
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            # Estimation For Cycle
            for i in range(0,self.it+1):

                print()
                print(f'----------------------------  Estimation {i}  ----------------------------')
                print()

                
                input_folder = self.data_folder
                output_folder = self.descr_folder + f'estimate_{i}/'
                
                # Descriptors Generation #############################################################################
                # List of files to process
                to_process = []

                # Filtering the files to process
                for file in sorted(os.listdir(input_folder)):
                    gen_process_number = int(file.split('_')[0][1:])
                    n_variables = int(file.split('_')[1][1:])
                    max_neighborhood_size = int(file.split('_')[2][2:])
                    noise_std = float(file.split('_')[3][1:-4])

                    if noise_std != self.noise_std_filter or max_neighborhood_size != self.max_neighborhood_size_filter:
                        continue

                    to_process.append(file)

                # Create output folder if it does not exist
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                print('Making Descriptors...')

                # Process each file and create new DAGs based on causal paths
                for file in tqdm(to_process):
                    gen_process_number = int(file.split('_')[0][1:])
                    n_variables = int(file.split('_')[1][1:])
                    max_neighborhood_size = int(file.split('_')[2][2:])
                    noise_std = float(file.split('_')[3][1:-4])

                    dataloader = DataLoader(n_variables=n_variables, maxlags=self.maxlags)
                    dataloader.from_pickle(input_folder + file)

                    # First iteration
                    if i  ==  0:
                        d2c = D2C(
                            observations=dataloader.get_observations(),
                            dags=dataloader.get_dags(),
                            couples_to_consider_per_dag= self.COUPLES_TO_CONSIDER_PER_DAG,
                            MB_size= self.MB_SIZE,
                            n_variables=n_variables,
                            maxlags= self.maxlags,
                            seed= self.SEED,
                            n_jobs= self.N_JOBS,
                            full=True,
                            quantiles=True,
                            normalize=True,
                            cmi='original',
                            mb_estimator=self.method,
                            top_vars=self.top_vars
                        )

                    # i > 0 iterations
                    else:
                        d2c = D2C(
                            observations=dataloader.get_observations(),
                            dags=dataloader.get_dags(),
                            couples_to_consider_per_dag= self.COUPLES_TO_CONSIDER_PER_DAG,
                            MB_size= self.MB_SIZE,
                            n_variables=n_variables,
                            maxlags= self.maxlags,
                            seed= self.SEED,
                            n_jobs= self.N_JOBS,
                            full=True,
                            quantiles=True,
                            normalize=True,
                            cmi='original',
                            mb_estimator= 'iterative',
                            top_vars=self.top_vars,
                            causal_df_list=dataloader.from_pickle_causal_df(os.path.join(self.results_folder, f'metrics/estimate_{i-1}/', f'causal_df_top_{self.k}_td2c_R_N5.pkl'))
                        )

                    d2c.initialize()  # Initializes the D2C object
                    descriptors_df = d2c.get_descriptors_df()  # Computes the descriptors

                    # Save the descriptors along with new DAGs if needed
                    descriptors_df.insert(0, 'process_id', gen_process_number)
                    descriptors_df.insert(2, 'n_variables', n_variables)
                    descriptors_df.insert(3, 'max_neighborhood_size', max_neighborhood_size)
                    descriptors_df.insert(4, 'noise_std', noise_std)

                    descriptors_df.to_pickle(output_folder + f'Estimate_{i}_P{gen_process_number}_N{n_variables}_Nj{max_neighborhood_size}_n{noise_std}_MB{self.MB_SIZE}.pkl')

                # Set Classifier #################################################################################
                data_root = self.data_folder

                to_dos = []

                # This loop gets a list of all the files to be processed
                for testing_file in sorted(os.listdir(data_root)):
                    if testing_file.endswith('.pkl'):
                        gen_process_number = int(testing_file.split('_')[0][1:])
                        n_variables = int(testing_file.split('_')[1][1:])
                        max_neighborhood_size = int(testing_file.split('_')[2][2:])
                        noise_std = float(testing_file.split('_')[3][1:-4])
                        
                        if noise_std != 0.01: # if the noise is different we skip the file
                            continue

                        if max_neighborhood_size != 2: # if the max_neighborhood_size is different we skip the file
                            continue

                        to_dos.append(testing_file) # we add the file to the list (to_dos) to be processed

                # sort to_dos by number of variables
                to_dos_5_variables = [file for file in to_dos if int(file.split('_')[1][1:]) == 5]
                # to_dos_10_variables = [file for file in to_dos if int(file.split('_')[1][1:]) == 10]
                # to_dos_25_variables = [file for file in to_dos if int(file.split('_')[1][1:]) == 25]

                # we create a dictionary with the lists of files to be processed
                todos = {'5': to_dos_5_variables} # , '10': to_dos_10_variables, '25': to_dos_25_variables

                # we create a dictionary to store the results
                dfs = []
                descriptors_root = self.descr_folder + f'estimate_{i}/'

                # Create the folder if it doesn't exist
                if not os.path.exists(descriptors_root):
                    os.makedirs(descriptors_root)

                # Re-save pickle files with protocol 4
                for testing_file in sorted(os.listdir(descriptors_root)):
                    if testing_file.endswith('.pkl'):
                        file_path = os.path.join(descriptors_root, testing_file)
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        
                        # Re-save with protocol 4
                        with open(file_path, 'wb') as f:
                            pickle.dump(data, f, protocol=4)

                # This loop gets the descriptors for the files to be processed
                for testing_file in sorted(os.listdir(descriptors_root)):
                    if testing_file.endswith('.pkl'):
                        df = pd.read_pickle(os.path.join(descriptors_root, testing_file))
                        if isinstance(df, pd.DataFrame):
                            dfs.append(df)

                # we concatenate the descriptors
                descriptors_training = pd.concat(dfs, axis=0).reset_index(drop=True)

                # Classifier & Evaluation Metrics #################################################################

                print('Classification & Evaluation Metrics')

                for n_vars, todo in todos.items():

                    m1 = f'Estimate_{i}_rocs_process'
                    # m2 = f'Estimate_{i}_precision_process'
                    # m3 = f'Estimate_{i}_recall_process'
                    # m4 = f'Estimate_{i}_f1_process'

                    globals()[m1] = {}
                    # globals()[m2] = {}
                    # globals()[m3] = {}
                    # globals()[m4] = {}
                    causal_df_1 = {}

                    for testing_file in tqdm(todo):
                        gen_process_number = int(testing_file.split('_')[0][1:])
                        n_variables = int(testing_file.split('_')[1][1:])
                        max_neighborhood_size = int(testing_file.split('_')[2][2:])
                        noise_std = float(testing_file.split('_')[3][1:-4])

                        # split training and testing data
                        training_data = descriptors_training.loc[descriptors_training['process_id'] != gen_process_number]
                        X_train = training_data.drop(columns=['process_id', 'graph_id', 'n_variables', 'max_neighborhood_size','noise_std', 'edge_source', 'edge_dest', 'is_causal',])
                        y_train = training_data['is_causal']

                        testing_data = descriptors_training.loc[(descriptors_training['process_id'] == gen_process_number) & (descriptors_training['n_variables'] == n_variables) & (descriptors_training['max_neighborhood_size'] == max_neighborhood_size) & (descriptors_training['noise_std'] == noise_std)]

                        model = BalancedRandomForestClassifier(n_estimators=100, random_state=0, n_jobs=self.N_JOBS, max_depth=None, sampling_strategy='auto', replacement=True, bootstrap=False)
                        # model = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=50, max_depth=10)

                        model = model.fit(X_train, y_train)

                        rocs = {}
                        # precisions = {}
                        # recalls = {}
                        # f1s = {}
                        causal_dfs = {}
                        for graph_id in range(40):
                            #load testing descriptors
                            test_df = testing_data.loc[testing_data['graph_id'] == graph_id]
                            test_df = test_df.sort_values(by=['edge_source','edge_dest']).reset_index(drop=True) # sort for coherence

                            X_test = test_df.drop(columns=['process_id', 'graph_id', 'n_variables', 'max_neighborhood_size','noise_std', 'edge_source', 'edge_dest', 'is_causal',])
                            y_test = test_df['is_causal']

                            y_pred_proba = model.predict_proba(X_test)[:,1]
                            y_pred = model.predict(X_test)

                            roc = roc_auc_score(y_test, y_pred_proba)
                            # precision = precision_score(y_test, y_pred)
                            # recall = recall_score(y_test, y_pred)
                            # f1 = f1_score(y_test, y_pred)
                            
                            rocs[graph_id] = roc
                            # precisions[graph_id] = precision
                            # recalls[graph_id] = recall
                            # f1s[graph_id] = f1
                            
                            # add to causal_df test_df, y_pred_proba and y_pred
                            causal_dfs[graph_id] = test_df
                            causal_dfs[graph_id]['y_pred_proba'] = y_pred_proba
                            causal_dfs[graph_id]['y_pred'] = y_pred

                        causal_df_1[gen_process_number] = causal_dfs
                        globals()[m1][gen_process_number] = rocs
                        # globals()[m2][gen_process_number] = precisions
                        # globals()[m3][gen_process_number] = recalls
                        # globals()[m4][gen_process_number] = f1s

                # pickle everything
                output_folder = self.results_folder + f'journals/estimate_{i}/'

                # Create the folder if it doesn't exist
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                with open(os.path.join(output_folder, f'journal_results_td2c_R_N5.pkl'), 'wb') as f:
                    everything = (globals()[m1], causal_df_1) #, globals()[m2], globals()[m3], globals()[m4]
                    pickle.dump(everything, f)

                # Load results #####################################################################################
                input_folder = self.results_folder + f'journals/estimate_{i}/'

                with open(os.path.join(input_folder, f'journal_results_td2c_R_N5.pkl'), 'rb') as f:
                    TD2C_1_rocs_process, causal_df_1 = pickle.load(f) # , TD2C_1_precision_process, TD2C_1_recall_process, TD2C_1_f1_process


                # STOPPING CRITERIA 2: Using ROC-AUC score
                roc = pd.DataFrame(TD2C_1_rocs_process).mean().mean()
                roc_scores.append(roc)
            
                print()
                print(f'ROC-AUC score: {roc}')
                print()

                if i == 0:
                    if roc < 0.5:
                        print('ROC-AUC is too low, let\'s stop here.')
                        return
                elif i > 0:
                    if roc <= roc_0:
                        stop_2 = stop_2 + 1
                        if stop_2 == 5:
                            print()
                            print('Estimations are not improving, let\'s stop here.')
                            print()
                            return

                    else:
                        stop_2 = 0
                    
                    if roc <= roc_0 - 0.1:
                        print()
                        print(f'ROC-AUC score: {roc}')
                        print()
                        print('Estimations are not improving, let\'s stop here.')
                        print()
                        return

                roc_0 = roc

                # Reshape causal_df #################################################################################
                # keep only rows for top k y_pred_proba

                # forse meglio creando una copia prima di iterare
                for process_id, process_data in causal_df_1.items():
                    for graph_id, graph_data in process_data.items():
                        causal_df_1[process_id][graph_id] = graph_data.nlargest(self.k, 'y_pred_proba')
                        # drop rows with y_pred_proba < 0.8
                        causal_df_1[process_id][graph_id] = graph_data[graph_data['y_pred_proba'] >= 0.8]

                # for each causal_df keep only process_id, graph_id, edge_source, edge_dest and y_pred_proba
                for process_id, process_data in causal_df_1.items():
                    for graph_id, graph_data in process_data.items():
                        causal_df_1[process_id][graph_id] = graph_data[['process_id', 'graph_id', 'edge_source', 'edge_dest', 'y_pred_proba']]
                        causal_df_1[process_id][graph_id].reset_index(drop=True, inplace=True)

                # first non-empty element in causal_df_1
                for process_id, process_data in causal_df_1.items():
                    for graph_id, graph_data in process_data.items():
                        if graph_data.shape[0] > 0:
                            causal_df_1_non_empty = causal_df_1[process_id][graph_id]
                            break

                print(f'Example causal_df_1: {causal_df_1_non_empty}')

                # # Unify causal_df #################################################################################
                # input_folder = self.results_folder + f'metrics/estimate_{i}/'

                # with open(os.path.join(input_folder, f'causal_df_top_{self.k}_td2c_R_N5.pkl'), 'rb') as f:
                #     causal_df_1 = pickle.load(f)

                # # create a dataframe with all the causal_df
                # dfs = []
                # for process_id, process_data in causal_df_1.items():
                #     for graph_id, graph_data in process_data.items():
                #         dfs.append(graph_data)

                # causal_df_unif_1_mid = pd.concat(dfs, axis=0).reset_index(drop=True)

                # # sort in ascending order by process_id, graph_id, edge_source and edge_dest
                # causal_df_unif_1_mid.sort_values(by=['process_id', 'graph_id', 'edge_source', 'edge_dest'], inplace=True)

                # # unique of causal_df_unif for couples of edge_source and edge_dest
                # causal_df_unif_1 = causal_df_unif_1_mid.drop_duplicates(subset=['edge_source', 'edge_dest'])

                # causal_df_mid.append(causal_df_unif_1)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
                for process_id, process_data in causal_df_1.items():
                    for graph_id, graph_data in process_data.items():
                        causal_df_unif_1 = graph_data

                        # KEEP VARIABLES IN CAUSAL_DF FOR A TRESHOLD OR TOP N 
                        if self.treshold == True:
                            # drop rows with y_pred_proba < treshold_value
                            causal_df_unif_1 = causal_df_unif_1[causal_df_unif_1['y_pred_proba'] >= self.treshold_value]
                            if causal_df_unif_1.shape[0] > 0:
                                causal_df_unif_1 = causal_df_unif_1.nlargest(10, 'y_pred_proba')

                        elif self.adaptive == True:
                            if self.adaptive_mode == "Adding":
                                # Initialize `previous_size` as the starting size for each iteration.
                                if i == 0:
                                    previous_size = self.size_causal_df
                                    causal_df_unif_1 = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                else:
                                    # Adjust size based on the comparison of ROC scores
                                    if roc_scores[i] < roc_scores[i-1]:
                                        previous_size += 1
                                    # elif roc_scores[i] >= roc_scores[i-1]:
                                    #     previous_size = previous_size # + 1

                                    # Ensure size boundaries
                                    previous_size = max(1, min(previous_size, 10))

                                    # Select the top rows according to the adjusted size
                                    causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')

                                    # index reset
                                    causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                    # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size 
                                    # if it is until it is different
                                    while any(causal_df_unif_provv.equals(df) for df in causal_df_permut):
                                        previous_size += 1
                                        causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                        causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                    causal_df_unif_1 = causal_df_unif_provv

                            elif self.adaptive_mode == "Subtracting":
                                # Initialize `previous_size` as the starting size for each iteration.
                                if i == 0:
                                    previous_size = self.size_causal_df
                                    causal_df_unif_1 = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                else:
                                    # Adjust size based on the comparison of ROC scores
                                    if roc_scores[i] < roc_scores[i-1]:
                                        previous_size = previous_size - 1
                                    # elif roc_scores[i] >= roc_scores[i-1]:
                                    #     previous_size = previous_size # + 1

                                    # Ensure size boundaries
                                    previous_size = max(1, min(previous_size, 15))

                                    # Select the top rows according to the adjusted size
                                    causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')

                                    # index reset
                                    causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                    # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size 
                                    # if it is until it is different
                                    while any(causal_df_unif_provv.equals(df) for df in causal_df_permut):
                                        previous_size += 1
                                        causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                        causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                    causal_df_unif_1 = causal_df_unif_provv

                            elif self.adaptive_mode == "Balancing1":
                                # Initialize `previous_size` as the starting size for each iteration.
                                if i == 0:
                                    previous_size = self.size_causal_df
                                    causal_df_unif_1 = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                else:
                                    # if i is odd
                                    if i % 2 != 0:
                                        if roc_scores[i] < roc_scores[i-1]:
                                            previous_size = previous_size + 1
                                    elif i % 2 == 0:
                                        # Adjust size based on the comparison of ROC scores
                                        if roc_scores[i] < roc_scores[i-1]:
                                            previous_size = previous_size - 1

                                    # Ensure size boundaries
                                    previous_size = max(1, min(previous_size, 10))

                                    # Select the top rows according to the adjusted size
                                    causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')

                                    # index reset
                                    causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                    # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size 
                                    # if it is until it is different
                                    while any(causal_df_unif_provv.equals(df) for df in causal_df_permut):
                                        previous_size += 1
                                        causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                        causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                    causal_df_unif_1 = causal_df_unif_provv

                            elif self.adaptive_mode == "Balancing2":
                                # for i ==1
                                # Add 1 if the ROC is worse than the one before
                                # for i > 1
                                # Add 1 if the ROC is worse than the one before and:
                                # The last operation was an addition, and the ROC improved
                                # The last operation was a subtraction, and the ROC decreased.
                                # Remove 1 if the ROC is worse than the one before and:
                                # The last operation was an addition, and the ROC decreased.
                                # The last operation was a subtraction, and the ROC improved.
                                if i == 0:
                                    previous_size = self.size_causal_df
                                    causal_df_unif_1 = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                    operation_reg.append(None)
                                elif i == 1:
                                    if roc_scores[i] < roc_scores[i-1]:
                                            previous_size = previous_size + 1
                                            operation = "add"
                                            operation_reg.append(operation)
                                    else:
                                        operation_reg.append(None)

                                    previous_size = max(1, min(previous_size, 10))

                                    # Select the top rows according to the adjusted size
                                    causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')

                                    # index reset
                                    causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                    # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size 
                                    # if it is until it is different
                                    while any(causal_df_unif_provv.equals(df) for df in causal_df_permut):
                                        previous_size += 1
                                        causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                        causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                    causal_df_unif_1 = causal_df_unif_provv
                                else:
                                    if roc_scores[i] < roc_scores[i-1]:
                                        # index of the element in operation_reg != None with the highest index
                                        last_operation_index = len(operation_reg) - 1 - operation_reg[::-1].index(next(filter(None, operation_reg)))
                                        last_operation = operation_reg[last_operation_index]
                                        if last_operation == "add" and roc_scores[last_operation_index] < roc_scores[last_operation_index + 1]:
                                            previous_size += 1
                                            operation = "add"
                                        elif last_operation == "add" and roc_scores[last_operation_index] >= roc_scores[last_operation_index + 1]:
                                            previous_size -= 1
                                            operation = "sub"
                                        elif last_operation == "sub" and roc_scores[last_operation_index] < roc_scores[last_operation_index + 1]:
                                            previous_size -= 1
                                            operation = "sub"
                                        elif last_operation == "sub" and roc_scores[last_operation_index] >= roc_scores[last_operation_index + 1]:
                                            previous_size += 1
                                            operation = "add"
                                        operation_reg.append(operation)
                                    else:
                                        operation_reg.append(None)

                                    # Ensure size boundaries
                                    previous_size = max(1, min(previous_size, 15))

                                    # Select the top rows according to the adjusted size
                                    causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')

                                    # index reset
                                    causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                    # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size
                                    # if it is until it is different
                                    while any(causal_df_unif_provv.equals(df) for df in causal_df_permut):
                                        previous_size += 1
                                        causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                        causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                    causal_df_unif_1 = causal_df_unif_provv

                            elif self.adaptive_mode == "try": # just to see if with same causal_df the roc score is the same, it is not because of the random forest
                                if i == 0:
                                    previous_size = self.size_causal_df
                                    causal_df_unif_1 = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                else:
                                    causal_df_unif_1 = causal_df_unified[i-1]

                        elif self.performative == True:
                            if self.performative_mode == "More_Less":
                                if i == 0:
                                    previous_size = self.size_causal_df
                                    causal_df_unif_1 = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                else:
                                    if roc_0 - 0.04 < roc < roc_0 + 0.04:
                                        causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                        # index reset
                                        causal_df_unif_provv.reset_index(drop=True, inplace=True)
                                        # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size 
                                        # if it is until it is different
                                        while any(causal_df_unif_provv.equals(df) for df in causal_df_permut):
                                            previous_size += 1
                                            causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                            causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                        causal_df_unif_1 = causal_df_unif_provv
                                    else:
                                        return
                            
                            elif self.performative_mode == "More":
                                if i == 0:
                                    previous_size = self.size_causal_df
                                    causal_df_unif_1 = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                elif i < 5:
                                        causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                        # index reset
                                        causal_df_unif_provv.reset_index(drop=True, inplace=True)
                                        # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size 
                                        # if it is until it is different
                                        while any(causal_df_unif_provv.equals(df) for df in causal_df_permut):
                                            previous_size += 1
                                            causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                            causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                        causal_df_unif_1 = causal_df_unif_provv
                                else:
                                    if roc_scores.mean() < roc_0:
                                        causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                        # index reset
                                        causal_df_unif_provv.reset_index(drop=True, inplace=True)
                                        # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size 
                                        # if it is until it is different
                                        while any(causal_df_unif_provv.equals(df) for df in causal_df_permut):
                                            previous_size += 1
                                            causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                            causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                        causal_df_unif_1 = causal_df_unif_provv
                                    else:
                                        return
                        
                            elif self.performative_mode == "Tail":
                                # after self.it iterations, if avg. roc score is < the first one, goes to the highest roc score, 
                                # adds 1 to the size of the causal_df and goes for other self.it/2 iterations
                                if i == 0:
                                    previous_size = self.size_causal_df
                                    causal_df_unif_1 = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                elif i != it_tail:
                                    causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                    # index reset
                                    causal_df_unif_provv.reset_index(drop=True, inplace=True)
                                    # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size 
                                    # if it is until it is different
                                    while any(causal_df_unif_provv.equals(df) for df in causal_df_permut):
                                        previous_size += 1
                                        causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                        causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                    causal_df_unif_1 = causal_df_unif_provv
                                else:
                                    if tail == 0:
                                        mean_roc_score = sum(roc_scores) / len(roc_scores)
                                        print(f'mean of roc-auc scores so far is: {mean_roc_score}')
                                        if mean_roc_score < roc_scores[0]:
                                            tail = tail + 1
                                            index = roc_scores.index(max(roc_scores))
                                            # set size to the highest roc score
                                            previous_size = causal_df_unified[index].shape[0] + 1
                                            causal_df_unif_provv = causal_df_mid[index].nlargest(previous_size, 'y_pred_proba')

                                            # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size
                                            # if it is until it is different
                                            while any(causal_df_unif_1.equals(df) for df in causal_df_permut):
                                                previous_size += 1
                                                causal_df_unif_1 = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                                causal_df_unif_1.reset_index(drop=True, inplace=True)

                                            causal_df_unif_provv = causal_df_unif_1
                                        else:
                                            causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                            # index reset
                                            causal_df_unif_provv.reset_index(drop=True, inplace=True)
                                            # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size 
                                            # if it is until it is different
                                            while any(causal_df_unif_provv.equals(df) for df in causal_df_permut):
                                                previous_size += 1
                                                causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                                causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                            causal_df_unif_1 = causal_df_unif_provv
                                    else: 
                                        causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                        # index reset
                                        causal_df_unif_provv.reset_index(drop=True, inplace=True)
                                        # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size 
                                        # if it is until it is different
                                        while any(causal_df_unif_provv.equals(df) for df in causal_df_permut):
                                            previous_size += 1
                                            causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                            causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                        causal_df_unif_1 = causal_df_unif_provv

                        elif self.arbitrary == True:
                            if self.arbitrary_mode == "Increasing":
                                # adds 1 at each iteration
                                if i == 0:
                                    previous_size = self.size_causal_df
                                    causal_df_unif_1 = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                else:
                                    previous_size = previous_size + 1

                                    # Ensure size boundaries
                                    previous_size = max(1, min(previous_size, 15))

                                    # Select the top rows according to the adjusted size
                                    causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')

                                    # index reset
                                    causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                    # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size 
                                    # if it is until it is different
                                    while any(causal_df_unif_provv.equals(df) for df in causal_df_permut):
                                        previous_size += 1
                                        causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                        causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                    causal_df_unif_1 = causal_df_unif_provv

                            elif self.arbitrary_mode == "Decreasing":
                                # subtracts 1 at each iteration
                                if i == 0:
                                    previous_size = self.size_causal_df
                                    causal_df_unif_1 = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                else:
                                    previous_size = previous_size - 1

                                    # Ensure size boundaries
                                    previous_size = max(1, min(previous_size, 15))

                                    # Select the top rows according to the adjusted size
                                    causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')

                                    # index reset
                                    causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                    # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size 
                                    # if it is until it is different
                                    while any(causal_df_unif_provv.equals(df) for df in causal_df_permut):
                                        previous_size += 1
                                        causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                        causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                    causal_df_unif_1 = causal_df_unif_provv

                            # elif self.arbitrary_mode == "Common":
                            #     # keep couples edge_dest,edge_source present for at least 3 DAGS with prob.pred > 0.7
                            #     causal_df_unif_1 = causal_df_unif_1_mid[causal_df_unif_1_mid['y_pred_proba'] >= 0.7]

                            #     # keep only couples edge_dest,edge_source present for at least 3 DAGS
                            #     causal_df_unif_1 = causal_df_unif_1[causal_df_unif_1.duplicated(subset=['edge_source', 'edge_dest'], keep=False)]
                            #     # we don't care about the order because edges can only go from higher to lower nodes

                        else:
                            if i == 0:
                                previous_size = self.size_causal_df
                                causal_df_unif_1 = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')

                            else:
                                previous_size = max(1, min(previous_size, 15))

                                # Select the top rows according to the adjusted size
                                causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')

                                # index reset
                                causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size 
                                # if it is until it is different
                                while any(causal_df_unif_provv.equals(df) for df in causal_df_permut):
                                    previous_size += 1
                                    causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                                    causal_df_unif_provv.reset_index(drop=True, inplace=True)

                                causal_df_unif_1 = causal_df_unif_provv

                        # reset index
                        causal_df_unif_1.reset_index(drop=True, inplace=True)

                        causal_df_1[process_id][graph_id] = causal_df_unif_1 # this is a list of lists of dataframes

                        # save the causal_df as a pkl file alone
                        output_folder = self.results_folder + f'metrics/estimate_{i}/'

                        # Create the folder if it doesn't exist
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)

                        with open(os.path.join(output_folder, f'causal_df_top_{self.k}_td2c_R_N5.pkl'), 'wb') as f:
                            pickle.dump(causal_df_1, f)

                        # create loist of dataframes of permutations of causal_df_1 for couples of edge_source and edge_dest
                        df = pd.DataFrame(causal_df_unif_1)

                        # Get all permutations of the rows in the DataFrame
                        row_permutations = list(permutations(df.index))  # Get permutations of the row indices

                        # Create a list to store the permuted DataFrames
                        permuted_dfs = []

                        # Iterate over each permutation of row indices
                        for perm in row_permutations:
                            # Use the permuted index to create a new DataFrame
                            permuted_df = df.loc[list(perm)].reset_index(drop=True)
                            permuted_dfs.append(permuted_df) # this is a list of dataframes

                        causal_df_permut.append(permuted_dfs) # this is a list of lists of dataframes


                        # # STOPPING CRITERIA 1: if causal df is the same as the previous one for 3 consecutive iterations
                        if any(causal_df_unif_1.equals(df) for df in causal_df_permut):
                            stop_1 = stop_1 + 1
                            if stop_1 == 3:
                                roc_scores.append(roc)
                                print()
                                print(f'Most relevant Edges have been the same for 3 consecutive iterations:')
                                print()
                                print(causal_df_unif_1)
                                print()
                                print(f'No more improvements, let\'s stop here.')
                                print()
                                break
                            else:
                                stop_1 = 0

                # save the causal_df as a pkl file alone
                output_folder = os.path.join(self.results_folder, f'metrics/estimate_{i}/')

                # Create the folder if it doesn't exist
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                with open(os.path.join(output_folder, f'causal_df_top_{self.k}_td2c_R_N5_unified.pkl'), 'wb') as f:
                    pickle.dump(causal_df_1, f)
                
                # print the resultant causal_df
                if i != self.it:
                    print()
                    print(f'Example of most relevant Edges that will be added in the next iteration:')
                    print(causal_df_1[1][0])
                    print()
                else:
                    print()
                    print(f'Example of most relevant Edges for the last iteration:')
                    print(causal_df_1[1][0])
                    print()
                    print('End of iterations.')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------

            return roc_scores, causal_df_1
        
        elif response in ['no', 'n', 'No', 'N']:
            print()
            print("Wise choice! Change the parameters and try again.")
            return
        
        else:
            print()
            print("ERROR:")
            return

    def final_iteration(self, best_edges): # to modify

        if best_edges.shape[0] > 10:
            best_edges = best_edges.nlargest(10, 'counts')

        causal_df_unified = best_edges

        # use causal_df_unified to run an ioterative TD2C with and Arbitrary Decreasing mode

        # setting
        stop_1 = 0
        stop_2 = 0
        roc_scores = []
        causal_df_mid = []
        causal_df_unified_list = []
        causal_df_unified_list.append(causal_df_unified)
        roc_0 = 0
        np.random.seed(self.SEED)
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        forind = best_edges.shape[0]

        for i in range(forind): # this loop goes from 0 to forind-1

            print()
            print(f'----------------------------  Estimation {i}  ----------------------------')
            print()

            # Iterative TD2C ####################################################
            input_folder = self.data_folder
            output_folder = self.descr_folder + f'estimate_{i}/'        

            # Descriptors Generation ############################################
            # List of files to process
            to_process = []

            # Filtering the files to process
            for file in sorted(os.listdir(input_folder)):
                gen_process_number = int(file.split('_')[0][1:])
                n_variables = int(file.split('_')[1][1:])
                max_neighborhood_size = int(file.split('_')[2][2:])
                noise_std = float(file.split('_')[3][1:-4])

                if noise_std != self.noise_std_filter or max_neighborhood_size != self.max_neighborhood_size_filter:
                    continue

                to_process.append(file)
            
            # Create output folder if it does not exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            print('Making Descriptors...')

            # Process each file and create new DAGs based on causal paths
            for file in tqdm(to_process):
                gen_process_number = int(file.split('_')[0][1:])
                n_variables = int(file.split('_')[1][1:])
                max_neighborhood_size = int(file.split('_')[2][2:])
                noise_std = float(file.split('_')[3][1:-4])

                dataloader = DataLoader(n_variables=n_variables, maxlags=self.maxlags)
                dataloader.from_pickle(input_folder + file)

                d2c = D2C(
                        observations=dataloader.get_observations(),
                        dags=dataloader.get_dags(),
                        couples_to_consider_per_dag= self.COUPLES_TO_CONSIDER_PER_DAG,
                        MB_size= self.MB_SIZE,
                        n_variables=n_variables,
                        maxlags= self.maxlags,
                        seed= self.SEED,
                        n_jobs= self.N_JOBS,
                        full=True,
                        quantiles=True,
                        normalize=True,
                        cmi='original',
                        mb_estimator= 'iterative',
                        top_vars=self.top_vars,
                        causal_df=causal_df_unified_list[i],
                    )

                d2c.initialize()  # Initializes the D2C object
                descriptors_df = d2c.get_descriptors_df()  # Computes the descriptors

                # Save the descriptors along with new DAGs if needed
                descriptors_df.insert(0, 'process_id', gen_process_number)
                descriptors_df.insert(2, 'n_variables', n_variables)
                descriptors_df.insert(3, 'max_neighborhood_size', max_neighborhood_size)
                descriptors_df.insert(4, 'noise_std', noise_std)

                descriptors_df.to_pickle(output_folder + f'Estimate_{i}_P{gen_process_number}_N{n_variables}_Nj{max_neighborhood_size}_n{noise_std}_MB{self.MB_SIZE}.pkl')

            # Set Classifier #################################################################################
            data_root = self.data_folder

            to_dos = []

            # This loop gets a list of all the files to be processed
            for testing_file in sorted(os.listdir(data_root)):
                if testing_file.endswith('.pkl'):
                    gen_process_number = int(testing_file.split('_')[0][1:])
                    n_variables = int(testing_file.split('_')[1][1:])
                    max_neighborhood_size = int(testing_file.split('_')[2][2:])
                    noise_std = float(testing_file.split('_')[3][1:-4])
                    
                    if noise_std != 0.01: # if the noise is different we skip the file
                        continue

                    if max_neighborhood_size != 2: # if the max_neighborhood_size is different we skip the file
                        continue

                    to_dos.append(testing_file) # we add the file to the list (to_dos) to be processed

            # sort to_dos by number of variables
            to_dos_5_variables = [file for file in to_dos if int(file.split('_')[1][1:]) == 5]
            # to_dos_10_variables = [file for file in to_dos if int(file.split('_')[1][1:]) == 10]
            # to_dos_25_variables = [file for file in to_dos if int(file.split('_')[1][1:]) == 25]

            # we create a dictionary with the lists of files to be processed
            todos = {'5': to_dos_5_variables} # , '10': to_dos_10_variables, '25': to_dos_25_variables

            # we create a dictionary to store the results
            dfs = []
            descriptors_root = self.descr_folder + f'estimate_{i}/'

            # Create the folder if it doesn't exist
            if not os.path.exists(descriptors_root):
                os.makedirs(descriptors_root)

            # Re-save pickle files with protocol 4
            for testing_file in sorted(os.listdir(descriptors_root)):
                if testing_file.endswith('.pkl'):
                    file_path = os.path.join(descriptors_root, testing_file)
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Re-save with protocol 4
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f, protocol=4)

            # This loop gets the descriptors for the files to be processed
            for testing_file in sorted(os.listdir(descriptors_root)):
                if testing_file.endswith('.pkl'):
                    df = pd.read_pickle(os.path.join(descriptors_root, testing_file))
                    if isinstance(df, pd.DataFrame):
                        dfs.append(df)

            # we concatenate the descriptors
            descriptors_training = pd.concat(dfs, axis=0).reset_index(drop=True)

            # Classifier & Evaluation Metrics #################################################################

            print('Classification & Evaluation Metrics')

            for n_vars, todo in todos.items():

                m1 = f'Estimate_{i}_rocs_process'
                # m2 = f'Estimate_{i}_precision_process'
                # m3 = f'Estimate_{i}_recall_process'
                # m4 = f'Estimate_{i}_f1_process'

                globals()[m1] = {}
                # globals()[m2] = {}
                # globals()[m3] = {}
                # globals()[m4] = {}
                causal_df_1 = {}

                for testing_file in tqdm(todo):
                    gen_process_number = int(testing_file.split('_')[0][1:])
                    n_variables = int(testing_file.split('_')[1][1:])
                    max_neighborhood_size = int(testing_file.split('_')[2][2:])
                    noise_std = float(testing_file.split('_')[3][1:-4])

                    # split training and testing data
                    training_data = descriptors_training.loc[descriptors_training['process_id'] != gen_process_number]
                    X_train = training_data.drop(columns=['process_id', 'graph_id', 'n_variables', 'max_neighborhood_size','noise_std', 'edge_source', 'edge_dest', 'is_causal',])
                    y_train = training_data['is_causal']

                    testing_data = descriptors_training.loc[(descriptors_training['process_id'] == gen_process_number) & (descriptors_training['n_variables'] == n_variables) & (descriptors_training['max_neighborhood_size'] == max_neighborhood_size) & (descriptors_training['noise_std'] == noise_std)]

                    model = BalancedRandomForestClassifier(n_estimators=100, random_state=0, n_jobs=50, max_depth=None, sampling_strategy='auto', replacement=True, bootstrap=False)
                    # model = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=50, max_depth=10)

                    model = model.fit(X_train, y_train)

                    rocs = {}
                    # precisions = {}
                    # recalls = {}
                    # f1s = {}
                    causal_dfs = {}
                    for graph_id in range(40):
                        #load testing descriptors
                        test_df = testing_data.loc[testing_data['graph_id'] == graph_id]
                        test_df = test_df.sort_values(by=['edge_source','edge_dest']).reset_index(drop=True) # sort for coherence

                        X_test = test_df.drop(columns=['process_id', 'graph_id', 'n_variables', 'max_neighborhood_size','noise_std', 'edge_source', 'edge_dest', 'is_causal',])
                        y_test = test_df['is_causal']

                        y_pred_proba = model.predict_proba(X_test)[:,1]
                        y_pred = model.predict(X_test)

                        roc = roc_auc_score(y_test, y_pred_proba)
                        # precision = precision_score(y_test, y_pred)
                        # recall = recall_score(y_test, y_pred)
                        # f1 = f1_score(y_test, y_pred)
                        
                        rocs[graph_id] = roc
                        # precisions[graph_id] = precision
                        # recalls[graph_id] = recall
                        # f1s[graph_id] = f1
                        
                        # add to causal_df test_df, y_pred_proba and y_pred
                        causal_dfs[graph_id] = test_df
                        causal_dfs[graph_id]['y_pred_proba'] = y_pred_proba
                        causal_dfs[graph_id]['y_pred'] = y_pred

                    causal_df_1[gen_process_number] = causal_dfs
                    globals()[m1][gen_process_number] = rocs
                    # globals()[m2][gen_process_number] = precisions
                    # globals()[m3][gen_process_number] = recalls
                    # globals()[m4][gen_process_number] = f1s

            # pickle everything
            output_folder = self.results_folder + f'journals/estimate_{i}/'

            # Create the folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            with open(os.path.join(output_folder, f'journal_results_td2c_R_N5.pkl'), 'wb') as f:
                everything = (globals()[m1], causal_df_1) #, globals()[m2], globals()[m3], globals()[m4]
                pickle.dump(everything, f)

            # Load results #####################################################################################
            input_folder = self.results_folder + f'journals/estimate_{i}/'

            with open(os.path.join(input_folder, f'journal_results_td2c_R_N5.pkl'), 'rb') as f:
                TD2C_1_rocs_process, causal_df_1 = pickle.load(f) # , TD2C_1_precision_process, TD2C_1_recall_process, TD2C_1_f1_process


            # STOPPING CRITERIA 2: Using ROC-AUC score
            roc = pd.DataFrame(TD2C_1_rocs_process).mean().mean()
            roc_scores.append(roc)
        
            print()
            print(f'ROC-AUC score: {roc}')
            print()

            if i == 0:
                if roc < 0.5:
                    print('ROC-AUC is too low, let\'s stop here.')
                    break
            elif i > 0:
                if roc <= roc_0:
                    stop_2 = stop_2 + 1
                    if stop_2 == 5:
                        print()
                        print('Estimations are not improving, let\'s stop here.')
                        print()
                        break
                else:
                    stop_2 = 0
                
                if roc <= roc_0 - 0.1:
                    print()
                    print(f'ROC-AUC score: {roc}')
                    print()
                    print('Estimations are not improving, let\'s stop here.')
                    print()
                    break

            roc_0 = roc

            # Reshape causal_df #################################################################################
            # keep only rows for top k y_pred_proba
            for process_id, process_data in causal_df_1.items():
                for graph_id, graph_data in process_data.items():
                    causal_df_1[process_id][graph_id] = graph_data.nlargest(self.k, 'y_pred_proba')

            # for each causal_df keep only process_id, graph_id, edge_source, edge_dest and y_pred_proba
            for process_id, process_data in causal_df_1.items():
                for graph_id, graph_data in process_data.items():
                    causal_df_1[process_id][graph_id] = graph_data[['process_id', 'graph_id', 'edge_source', 'edge_dest', 'y_pred_proba']]
                    causal_df_1[process_id][graph_id].reset_index(drop=True, inplace=True)

            # save the causal_df as a pkl file alone
            output_folder = self.results_folder + f'metrics/estimate_{i}/'

            # Create the folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            with open(os.path.join(output_folder, f'causal_df_top_{self.k}_td2c_R_N5.pkl'), 'wb') as f:
                pickle.dump(causal_df_1, f)

            # Unify causal_df #################################################################################
            input_folder = self.results_folder + f'metrics/estimate_{i}/'

            with open(os.path.join(input_folder, f'causal_df_top_{self.k}_td2c_R_N5.pkl'), 'rb') as f:
                causal_df_1 = pickle.load(f)

            # create a dataframe with all the causal_df
            dfs = []
            for process_id, process_data in causal_df_1.items():
                for graph_id, graph_data in process_data.items():
                    dfs.append(graph_data)

            causal_df_unif_1 = pd.concat(dfs, axis=0).reset_index(drop=True)

            # sort in ascending order by process_id, graph_id, edge_source and edge_dest
            causal_df_unif_1.sort_values(by=['process_id', 'graph_id', 'edge_source', 'edge_dest'], inplace=True)

            # unique of causal_df_unif for couples of edge_source and edge_dest
            causal_df_unif_1 = causal_df_unif_1.drop_duplicates(subset=['edge_source', 'edge_dest'])

            causal_df_mid.append(causal_df_unif_1)


            # method: Adaptive - Decreasing

            if i == 0:
                previous_size = forind
                causal_df_unif_1 = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
            else:
                # Adjust size based on the comparison of ROC scores
                if roc_scores[i] < roc_scores[i-1]:
                    previous_size = previous_size - 1 # this is going to remove the least relevant edge, i.e. the one with the lowest y_pred_proba

                # Ensure size boundaries
                previous_size = max(1, min(previous_size, 10))

                # Select the top rows according to the adjusted size
                causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')

                # index reset
                causal_df_unif_provv.reset_index(drop=True, inplace=True)

                # check if causal_df_unif_1 is equal to any previous element in causal_df_unified list of dataframes and add 1 to its size 
                # if it is until it is different
                while any(causal_df_unif_provv.equals(df) for df in causal_df_unified):
                    previous_size += 1
                    causal_df_unif_provv = causal_df_unif_1.nlargest(previous_size, 'y_pred_proba')
                    causal_df_unif_provv.reset_index(drop=True, inplace=True)

                causal_df_unif_1 = causal_df_unif_provv


            # reset index
            causal_df_unif_1.reset_index(drop=True, inplace=True)
            # add to list of results
            causal_df_unified_list.append(causal_df_unif_1)

            # # STOPPING CRITERIA 1: if causal df is the same as the previous one for 3 consecutive iterations
            if any(causal_df_unif_1.equals(df) for df in causal_df_unified):
                stop_1 = stop_1 + 1
                if stop_1 == 3:
                    roc_scores.append(roc)
                    print()
                    print(f'Most relevant Edges have been the same for 3 consecutive iterations:')
                    print()
                    print(causal_df_unif_1)
                    print()
                    print(f'No more improvements, let\'s stop here.')
                    print()
                    break
                else:
                    stop_1 = 0

            # save the causal_df as a pkl file alone
            output_folder = self.results_folder + f'metrics/estimate_{i}/'

            # Create the folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            with open(os.path.join(output_folder, f'causal_df_top_{self.k}_td2c_R_N5_unified.pkl'), 'wb') as f:
                pickle.dump(causal_df_unif_1, f)
            
            # print the resultant causal_df
            if i != forind-1:
                print()
                print(f'Most relevant Edges that will be added in the next iteration:')
                print(causal_df_unif_1)
                print()
            else:
                print()
                print(f'Most relevant Edges:')
                print(causal_df_unif_1)
                print()
                print('End of iterations.')

        # final_roc is the highest in the roc_scores
        final_roc = max(roc_scores)
        # final_causal_df is the one corresponding with the final_roc
        final_causal_df = causal_df_unified_list[roc_scores.index(final_roc)]

        return final_roc, final_causal_df

    def best_edges(self, roc_scores, causal_df_1, roc_scores_df):
        
        # select best roc_scores as the ones > roc_score[0]
        best_roc_scores = [roc for roc in roc_scores if roc > roc_scores[0]]
        # select best causal_df_1 as the ones with roc in best_roc_scores
        # best_causal_df_1 = [causal_df_1[i-1] for i, roc in enumerate(roc_scores) if roc in best_roc_scores]
        # print best_roc_scores enumered by the iteration
        best_causal_df_1 = [pd.DataFrame(causal_df_1[i-1]) for i, roc in enumerate(roc_scores) if roc in best_roc_scores]

        
        print()
        print()
        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")
        print("                          FINAL RESULTS                            ")
        print("°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°")

        # select the edges that are present in at least 2 of the best causal_df_1
        if len(best_roc_scores) == 0:
            print('No improvements have been made.')
            return None
        elif len(best_roc_scores) == 1:
            best_edges = pd.concat(best_causal_df_1, axis=0).reset_index(drop=True)
            print('The only improvement happened with these edges: ')
            print(best_causal_df_1[0])
            print()
            print('To make a relevant conclusion, we need at least 2 improvements.')
            return best_edges
        else:
            print()
            print(f'Best ROC-AUC scores: these iterations have improved the results of the first estimate with {self.method} method.')
            print(roc_scores_df[roc_scores_df['roc_score'].isin(best_roc_scores)])
            print()
            best_edges = pd.concat(best_causal_df_1, axis=0).reset_index(drop=True)
            best_edges = best_edges[best_edges.duplicated(subset=['edge_source', 'edge_dest'], keep=False)]
            # count the number of times each edge appears
            best_edges = best_edges.groupby(['edge_source', 'edge_dest']).size().reset_index(name='counts')
            # sort by counts
            best_edges.sort_values(by='counts', ascending=False, inplace=True)
            # reset index
            best_edges.reset_index(drop=True, inplace=True)
            # make best_edges a DataFrame

            # print best_edges
            print()
            print('Best Edges: these edges have been present in at least 2 of the best iterations\' causal dataframes.')
            print(best_edges)
            print()

            # save the best_edges as a csv file
            output_folder = self.results_folder + 'metrics/best_edges/'

            # Create the folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            best_edges.to_csv(output_folder + f'best_edges_TD2C_{self.method}_{len(roc_scores)}_iterations_{self.k}_top_vars_{self.COUPLES_TO_CONSIDER_PER_DAG}_couples_per_dag.csv', index=False)

            return best_edges

    def plot_results(self, roc_scores, strategy):

            if self.treshold == None or self.method == None or self.k == None or self.it == None or self.size_causal_df == None or self.COUPLES_TO_CONSIDER_PER_DAG == None or roc_scores == None:
                print('Please run iterative_td2c() function first')
                return
            else:
                print()
                print('Resultant ROC-AUC scores Plot:')
                print()

                plt.figure(figsize=(12, 6))
                plt.plot(range(0, len(roc_scores)), roc_scores, marker='o', linestyle='-', label='ROC-AUC score')

                # Calculate cumulative average
                cumulative_avg = np.cumsum(roc_scores) / np.arange(1, len(roc_scores) + 1)
                plt.plot(range(0, len(roc_scores)), cumulative_avg, marker='x', linestyle='--', label='Cumulative Average')

                if self.treshold == False:
                    plt.title(f'ROC-AUC scores for Iterative {self.method} ({len(roc_scores)} iterations and {self.size_causal_df} top vars) with Regression MI (5 vars processes) ({self.COUPLES_TO_CONSIDER_PER_DAG} couples per dag)')
                else:
                    plt.title(f'ROC-AUC scores for Iterative {self.method} ({len(roc_scores)} iterations and {self.k} top vars) with Regression MI (5 vars processes) ({self.COUPLES_TO_CONSIDER_PER_DAG} couples per dag)')

                plt.xlabel('Iterations')
                plt.ylabel('ROC-AUC score')
                plt.grid()
                plt.legend()
                plt.tight_layout()

                # save the plot in folder
                output_folder = '/home/jpalombarini/td2c/notebooks/contributions/td2c_extesions/results/Regression/try_complete_function/plots/'
                if self.treshold == False:
                    plt.savefig(output_folder + f'ROC_AUC_scores_TD2C_{self.method}_{len(roc_scores)}_iterations_{self.size_causal_df}_top_vars_{self.COUPLES_TO_CONSIDER_PER_DAG}_couples_per_dag_{strategy}.pdf')
                else:
                    plt.savefig(output_folder + f'ROC_AUC_scores_TD2C_{self.method}_{len(roc_scores)}_iterations_{self.k}_top_vars_{self.COUPLES_TO_CONSIDER_PER_DAG}_couples_per_dag_{strategy}.pdf')
                plt.show()

    def df_scores(self, roc_scores, strategy):

            if roc_scores == None or self.method == None or self.it == None or self.COUPLES_TO_CONSIDER_PER_DAG == None or self.size_causal_df == None:
                print('Please run iterative_td2c() function first')
                return
            else:
                print()
                print('Resultant ROC-AUC scores:')
                print()
                roc_scores_df = pd.DataFrame(roc_scores, columns=['roc_score'])
                roc_scores_df['iteration'] = range(0, len(roc_scores))

                # save the df in a csv file
                output_folder = '/home/jpalombarini/td2c/notebooks/contributions/td2c_extesions/results/Regression/try_complete_function/metrics/'
                if self.treshold == False:
                    roc_scores_df.to_csv(output_folder + f'roc_scores_TD2C_{self.method}_{len(roc_scores)}_iterations_{self.size_causal_df}_top_vars_{self.COUPLES_TO_CONSIDER_PER_DAG}_couples_per_dag_{strategy}.csv', index=False)
                else:
                    roc_scores_df.to_csv(output_folder + f'roc_scores_TD2C_{self.method}_{len(roc_scores)}_iterations_{self.k}_top_vars_{self.COUPLES_TO_CONSIDER_PER_DAG}_couples_per_dag_{strategy}.csv', index=False)
                print(roc_scores_df)
            
            return roc_scores_df

    def final_run(self, best_edges):

        if best_edges is None:
            print()
            print('Try with a different strategy or change the parameters.')
            return None, None
        
        elif best_edges.shape[0] == 1 or len(best_edges) == 1:
            print()
            print('No further improvements can be made. Try with a different strategy or change the parameters.')
            print()
            return None, None
        else:
            print()
            print('°°°°°°°°°°°°°°°°°')
            print(' Final Iteration ')
            print('°°°°°°°°°°°°°°°°°')
            print('We use the Adaptive - Subtractive method to find the best pool of edges among the best ones.')

            final_roc, final_causal_df = self.final_iteration(best_edges)
            
            print()
            print('°°°°°°°°°°°°°°°°°°°°°°°')
            print(' Most improved results ')
            print('°°°°°°°°°°°°°°°°°°°°°°°')
            print()
            print('ROC-AUC score:')
            print(final_roc)
            print()
            print('Causal DataFrame:')
            print(final_causal_df)

        return final_roc, final_causal_df

    def finale_estimate(self, final_causal_df, strategy):

        # which index of final_causal_df
        if final_causal_df is None:
            print()
            print('End of the process.')
            print()
        elif final_causal_df is None:
            print()
            print('The estimate with highest ROC-AUC score possible, using the best edges possible, is the first one.')
            print()
            print('End of the process.')
            print
            return
        else:
            print('°°°°°°°°°°°°°°°°°°°°°°°°°°')
            print('         FINAL STEP       ')
            print('°°°°°°°°°°°°°°°°°°°°°°°°°°')
            print()
            print('The estimate with highest ROC-AUC score possible, using the best edges possible, is the following:')
            print()

            causal_df_unified = []
            causal_df_unified.append(final_causal_df)

            # use causal_df_unified to run an ioterative TD2C with and Arbitrary Decreasing mode

            # setting
            np.random.seed(self.SEED)
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

            
            # Iterative TD2C ####################################################
            input_folder = self.data_folder
            output_folder = self.descr_folder + 'final_estimate/'   

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
     

            # Descriptors Generation ############################################
            # List of files to process
            to_process = []

            # Filtering the files to process
            for file in sorted(os.listdir(input_folder)):
                gen_process_number = int(file.split('_')[0][1:])
                n_variables = int(file.split('_')[1][1:])
                max_neighborhood_size = int(file.split('_')[2][2:])
                noise_std = float(file.split('_')[3][1:-4])

                if noise_std != self.noise_std_filter or max_neighborhood_size != self.max_neighborhood_size_filter:
                    continue

                to_process.append(file)

            print('Making Descriptors...')

            # Process each file and create new DAGs based on causal paths
            for file in tqdm(to_process):
                gen_process_number = int(file.split('_')[0][1:])
                n_variables = int(file.split('_')[1][1:])
                max_neighborhood_size = int(file.split('_')[2][2:])
                noise_std = float(file.split('_')[3][1:-4])

                dataloader = DataLoader(n_variables=n_variables, maxlags=self.maxlags)
                dataloader.from_pickle(input_folder + file)

                d2c = D2C(
                        observations=dataloader.get_observations(),
                        dags=dataloader.get_dags(),
                        couples_to_consider_per_dag= self.COUPLES_TO_CONSIDER_PER_DAG,
                        MB_size= self.MB_SIZE,
                        n_variables=n_variables,
                        maxlags= self.maxlags,
                        seed= self.SEED,
                        n_jobs= self.N_JOBS,
                        full=True,
                        quantiles=True,
                        normalize=True,
                        cmi='original',
                        mb_estimator= 'iterative',
                        top_vars=self.top_vars,
                        causal_df=causal_df_unified[0]
                    )

                d2c.initialize()  # Initializes the D2C object
                descriptors_df = d2c.get_descriptors_df()  # Computes the descriptors

                # Save the descriptors along with new DAGs if needed
                descriptors_df.insert(0, 'process_id', gen_process_number)
                descriptors_df.insert(2, 'n_variables', n_variables)
                descriptors_df.insert(3, 'max_neighborhood_size', max_neighborhood_size)
                descriptors_df.insert(4, 'noise_std', noise_std)

                descriptors_df.to_pickle(output_folder + f'Final_estimate_P{gen_process_number}_N{n_variables}_Nj{max_neighborhood_size}_n{noise_std}_MB{self.MB_SIZE}.pkl')

            # Set Classifier #################################################################################
            data_root = self.data_folder

            to_dos = []

            # This loop gets a list of all the files to be processed
            for testing_file in sorted(os.listdir(data_root)):
                if testing_file.endswith('.pkl'):
                    gen_process_number = int(testing_file.split('_')[0][1:])
                    n_variables = int(testing_file.split('_')[1][1:])
                    max_neighborhood_size = int(testing_file.split('_')[2][2:])
                    noise_std = float(testing_file.split('_')[3][1:-4])
                    
                    if noise_std != 0.01: # if the noise is different we skip the file
                        continue

                    if max_neighborhood_size != 2: # if the max_neighborhood_size is different we skip the file
                        continue

                    to_dos.append(testing_file) # we add the file to the list (to_dos) to be processed

            # sort to_dos by number of variables
            to_dos_5_variables = [file for file in to_dos if int(file.split('_')[1][1:]) == 5]
            # to_dos_10_variables = [file for file in to_dos if int(file.split('_')[1][1:]) == 10]
            # to_dos_25_variables = [file for file in to_dos if int(file.split('_')[1][1:]) == 25]

            # we create a dictionary with the lists of files to be processed
            todos = {'5': to_dos_5_variables} # , '10': to_dos_10_variables, '25': to_dos_25_variables

            # we create a dictionary to store the results
            dfs = []
            descriptors_root = self.descr_folder + 'final_estimate/'

            # Create the folder if it doesn't exist
            if not os.path.exists(descriptors_root):
                os.makedirs(descriptors_root)

            # Re-save pickle files with protocol 4
            for testing_file in sorted(os.listdir(descriptors_root)):
                if testing_file.endswith('.pkl'):
                    file_path = os.path.join(descriptors_root, testing_file)
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Re-save with protocol 4
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f, protocol=4)

            # This loop gets the descriptors for the files to be processed
            for testing_file in sorted(os.listdir(descriptors_root)):
                if testing_file.endswith('.pkl'):
                    df = pd.read_pickle(os.path.join(descriptors_root, testing_file))
                    if isinstance(df, pd.DataFrame):
                        dfs.append(df)

            # we concatenate the descriptors
            descriptors_training = pd.concat(dfs, axis=0).reset_index(drop=True)

            # Classifier & Evaluation Metrics #################################################################

            print('Classification & Evaluation Metrics')

            for n_vars, todo in todos.items():

                m1 = f'Final_stimate_rocs_process'
                # m2 = f'Estimate_{i}_precision_process'
                # m3 = f'Estimate_{i}_recall_process'
                # m4 = f'Estimate_{i}_f1_process'

                globals()[m1] = {}
                # globals()[m2] = {}
                # globals()[m3] = {}
                # globals()[m4] = {}
                causal_df_1 = {}

                for testing_file in tqdm(todo):
                    gen_process_number = int(testing_file.split('_')[0][1:])
                    n_variables = int(testing_file.split('_')[1][1:])
                    max_neighborhood_size = int(testing_file.split('_')[2][2:])
                    noise_std = float(testing_file.split('_')[3][1:-4])

                    # split training and testing data
                    training_data = descriptors_training.loc[descriptors_training['process_id'] != gen_process_number]
                    X_train = training_data.drop(columns=['process_id', 'graph_id', 'n_variables', 'max_neighborhood_size','noise_std', 'edge_source', 'edge_dest', 'is_causal',])
                    y_train = training_data['is_causal']

                    testing_data = descriptors_training.loc[(descriptors_training['process_id'] == gen_process_number) & (descriptors_training['n_variables'] == n_variables) & (descriptors_training['max_neighborhood_size'] == max_neighborhood_size) & (descriptors_training['noise_std'] == noise_std)]

                    model = BalancedRandomForestClassifier(n_estimators=100, random_state=0, n_jobs=50, max_depth=None, sampling_strategy='auto', replacement=True, bootstrap=False)
                    # model = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=50, max_depth=10)

                    model = model.fit(X_train, y_train)

                    rocs = {}
                    # precisions = {}
                    # recalls = {}
                    # f1s = {}
                    causal_dfs = {}
                    for graph_id in range(40):
                        #load testing descriptors
                        test_df = testing_data.loc[testing_data['graph_id'] == graph_id]
                        test_df = test_df.sort_values(by=['edge_source','edge_dest']).reset_index(drop=True) # sort for coherence

                        X_test = test_df.drop(columns=['process_id', 'graph_id', 'n_variables', 'max_neighborhood_size','noise_std', 'edge_source', 'edge_dest', 'is_causal',])
                        y_test = test_df['is_causal']

                        y_pred_proba = model.predict_proba(X_test)[:,1]
                        y_pred = model.predict(X_test)

                        roc = roc_auc_score(y_test, y_pred_proba)
                        # precision = precision_score(y_test, y_pred)
                        # recall = recall_score(y_test, y_pred)
                        # f1 = f1_score(y_test, y_pred)
                        
                        rocs[graph_id] = roc
                        # precisions[graph_id] = precision
                        # recalls[graph_id] = recall
                        # f1s[graph_id] = f1
                        
                        # add to causal_df test_df, y_pred_proba and y_pred
                        causal_dfs[graph_id] = test_df
                        causal_dfs[graph_id]['y_pred_proba'] = y_pred_proba
                        causal_dfs[graph_id]['y_pred'] = y_pred

                    causal_df_1[gen_process_number] = causal_dfs
                    globals()[m1][gen_process_number] = rocs
                    # globals()[m2][gen_process_number] = precisions
                    # globals()[m3][gen_process_number] = recalls
                    # globals()[m4][gen_process_number] = f1s

            # pickle everything
            output_folder = self.results_folder + 'journals/final_estimate/'

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Create the folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            with open(os.path.join(output_folder, 'journal_results_td2c_R_N5.pkl'), 'wb') as f:
                everything = (globals()[m1], causal_df_1) #, globals()[m2], globals()[m3], globals()[m4]
                pickle.dump(everything, f)

            # Load results #####################################################################################
            input_folder = self.results_folder + 'journals/final_estimate/'

            with open(os.path.join(input_folder, 'journal_results_td2c_R_N5.pkl'), 'rb') as f:
                TD2C_1_rocs_process, causal_df_1 = pickle.load(f) # , TD2C_1_precision_process, TD2C_1_recall_process, TD2C_1_f1_process


            # STOPPING CRITERIA 2: Using ROC-AUC score
            roc_avg = pd.DataFrame(TD2C_1_rocs_process).mean().mean()
            roc = pd.DataFrame(TD2C_1_rocs_process)

            # Combine data for boxplot
            combined_data = []

            for col in roc.columns:
                combined_data.append(roc[col])

            # Create labels for x-axis
            labels = []
            for col in roc.columns:
                labels.append(f'{col} TD2C_Reg')

            # Plotting
            plt.figure(figsize=(12, 6))
            box = plt.boxplot(combined_data, patch_artist=True)

            # Color customization
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, i in zip(box['boxes'], range(len(box['boxes']))):
                patch.set_facecolor(colors[i % 3])


            plt.xticks(range(1, len(labels) + 1), labels, rotation=-90)
            plt.title('Final boxplot of ROC-AUC scores for Iterative TD2C, with Regression to estimate MI (5 variables processes)')
            plt.xlabel('Processes')
            plt.ylabel('ROC-AUC score')
            plt.tight_layout()
            plt.show()

            # save the plot in folder
            output_folder = self.results_folder + 'plots/final_estimate/'
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            plt.savefig(output_folder + f'FINAL_ROC_AUC_scores_TD2C_{self.method}_iterations_{self.size_causal_df}_top_vars_{self.COUPLES_TO_CONSIDER_PER_DAG}_couples_per_dag_{strategy}.pdf')

            print()
            print(f'FINAL ROC-AUC: {roc_avg}')
            print()
            print('End of the process.')
            print



    def main(self):

        # Check parameters:
        self.param_check()

        self.start()

        # Give info one the iteration
        strategy = self.strategy()

        # Start the iteration?
        response = self.response()

        # run the iteration
        roc_scores, causal_df_1 = self.iterative_td2c(response)

        # plot results
        self.plot_results(roc_scores, strategy)

        # print roc scores
        roc_scores_df = self.df_scores(roc_scores, strategy)

        # best edges
        best_edges = self.best_edges(roc_scores, causal_df_1, roc_scores_df)

        # final run
        final_roc, final_causal_df = self.final_run(best_edges)

        # final estimate
        self.finale_estimate(final_causal_df, strategy)
