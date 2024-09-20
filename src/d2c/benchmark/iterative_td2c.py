
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

    def __init__(self, method = 'ts', k = 1, it = 6, top_vars = 3, N_JOBS = 50, noise_std_filter = 0.01,
                COUPLES_TO_CONSIDER_PER_DAG = -1, maxlags = 5, SEED = 42, MB_SIZE = 2,
                max_neighborhood_size_filter = 2, data_folder = 'home/data/', descr_folder = 'home/descr/', 
                results_folder = 'home/results/', strategy = 'Classic'):
        
        self.method = method
        self.k = k
        self.it = it
        self.top_vars = top_vars
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
        self.strategy = strategy
        np.random.seed(self.SEED)

    def main(self):

        # Check parameters:
        self.param_check()

        self.start()

        # Start the iteration?
        response = self.response()
        
        if response in ['yes', 'y', 'Yes', 'Y']:
            # run the iteration
            print()
            print("Ok! Let's start the iteration.")
            print()
            roc_scores, causal_dfs = self.iteration()
            print('roc_scores:')
            print(roc_scores)
        
        elif response in ['no', 'n', 'No', 'N']:
            print()
            print("Wise choice! Change the parameters and try again.")
            return
        
        else:
            print()
            print("ERROR: Answer not recognized. Please try again.")
            return

        # plot results
        self.plot_results(roc_scores)

        # print roc scores
        self.df_scores(roc_scores)

        # plot ground truth with the causal_df to compare
        self.plot_ground_truth(roc_scores, causal_dfs)



    def param_check(self):
        # verify parameters
        if self.method == None or self.k == None or self.it == None or self.top_vars == None or self.data_folder == None or self.descr_folder == None or self.results_folder == None or self.COUPLES_TO_CONSIDER_PER_DAG == None or self.maxlags == None or self.SEED == None or self.MB_SIZE == None or self.max_neighborhood_size_filter == None or self.noise_std_filter == None or self.N_JOBS == None or self.strategy == None:
            print('Please provide the correct parameters')
            return
        if self.method not in ['ts', 'original', 'ts_rank', 'ts_rank_2', 'ts_rank_3', 'ts_rank_4', 'ts_rank_5', 'ts_rank_6', 'ts_rank_7', 'ts_past', 'ts_rank_no_count']:
            print('Please provide the correct method')
            return
        if self.k < 1:
            print('Please provide a value greater than 0 for k')
            return
        if self.k < 10:
            print('We suggest you to keep k higher than 10 for a better performance')
            return
        if self.it < 1:
            print('Please provide a value greater than 0 for it')
            return
        if self.top_vars < 1:
            print('Please provide a value greater than 0 for top_vars')
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
        if self.strategy not in ["Classic", "More-Less", "More", "Tail", "Adding", "Subtracting", "Balancing1", "Balancing2", "Increasing", "Decreasing", "Random", "Treshold", "try"]:
            print('Please provide the correct strategy: Classic, More-Less, More, Tail, Adding, Subtracting, Balancing1, Balancing2, Increasing, Decreasing, Random, Treshold', 'try')
            return

    def start(self):

        # Description of the iteration
        print()
        print(f'Iterative TD2C - Method: {self.method} - Max iterations: {self.it} - Variables to keep per DAG: {self.k} - Top Variables: {self.top_vars}')
        print()
    
        # Computational time estimation

    def response(self):
        # Start the iteration?
        if self.it < 6:
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

        return response

#####
    def iteration(self):

        # SETTINGS ______________________________________________________________________________________________
        
        si = self.k
        th = 0.8
        stop_1 = 0
        roc_scores = []
        causal_dfs = {}
        np.random.seed(self.SEED)
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        for i in range(0,self.it+1):

            # Initialization _____________________________________________________________________________________

            input_folder, output_folder = self.initialization(i=i)

            # Descriptors Generation _____________________________________________________________________________

            self.descriptors_generation(i=i, input_folder=input_folder, output_folder=output_folder)

            # Set Classifier _____________________________________________________________________________________

            descriptors_training, todos = self.set_classifier(i)

            # Classifier & Evaluation Metrics ____________________________________________________________________

            print('Classification & Evaluation Metrics')

            self.classifier(i, descriptors_training, todos)

            # Load results (roc-auc score and causal_df) _________________________________________________________

            roc, roc_scores, causal_df = self.load_results(i, roc_scores)

            self.stopping_criteria_1(i, roc, roc_scores, stop_1)

            # Reshape causal_df __________________________________________________________________________________

            causal_df, causal_dfs = self.reshape_causal_df(i, causal_df, causal_dfs, roc_scores, si, th)

            # Save causal_df _____________________________________________________________________________________

            self.save_causal_df(i, causal_df)

            # Print Best edges found in causal_df ________________________________________________________________
            
            self.print_best_edges(causal_df, i)

        # Save causal_dfs _____________________________________________________________________________________
        
        self.save_causal_dfs(i, causal_dfs)

        return roc_scores, causal_dfs


    def initialization(self, i):
        print()
        print(f'----------------------------  Estimation {i}  ----------------------------')
        print()

        
        input_folder = self.data_folder
        output_folder = self.descr_folder + f'estimate_{i}/'
        
        # Create output folder if it does not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        return input_folder, output_folder

    def descriptors_generation(self, i, input_folder, output_folder):
        # List of files to process
        to_process = []

        # Filtering the files to process
        for file in sorted(os.listdir(input_folder)):
            gen_process_number = int(file.split('_')[0][1:])
            n_variables = int(file.split('_')[1][1:])
            max_neighborhood_size = int(file.split('_')[2][2:])
            noise_std = float(file.split('_')[3][1:-4])
            # if gen_process_number < 10:
            #     continue

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

            # First iteration: classic TD2C
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

            # i > 0 iterations: uses the causal_df from the previous iteration and the strategy to adjust the causal_df for the next iteration
            else:
                d2c = D2C(
                    observations = dataloader.get_observations(),
                    dags = dataloader.get_dags(),
                    couples_to_consider_per_dag = self.COUPLES_TO_CONSIDER_PER_DAG,
                    MB_size = self.MB_SIZE,
                    n_variables =n_variables,
                    maxlags = self.maxlags,
                    seed = self.SEED,
                    n_jobs = self.N_JOBS,
                    full = True,
                    quantiles = True,
                    normalize = True,
                    cmi='original',
                    mb_estimator = 'iterative',
                    top_vars = self.top_vars,
                    causal_df_list =  dataloader.from_pickle_causal_df(os.path.join(self.results_folder, f'causal_dfs/single/', f'it_{i-1}_causal_df_top_{self.k}_td2c_R_N5.pkl'))
                )

            d2c.initialize()  # Initializes the D2C object
            descriptors_df = d2c.get_descriptors_df()  # Computes the descriptors

            # Save the descriptors along with new DAGs if needed
            descriptors_df.insert(0, 'process_id', gen_process_number)
            descriptors_df.insert(2, 'n_variables', n_variables)
            descriptors_df.insert(3, 'max_neighborhood_size', max_neighborhood_size)
            descriptors_df.insert(4, 'noise_std', noise_std)

            descriptors_df.to_pickle(output_folder + f'Estimate_{i}_P{gen_process_number}_N{n_variables}_Nj{max_neighborhood_size}_n{noise_std}_MB{self.MB_SIZE}.pkl')

        return descriptors_df

    def set_classifier(self, i):

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

        # sort to_dos by number of variables (IT'S POSSIBLE TO CHOSE BETWEEN 5, 10, 25 VARIABLES)
        to_dos_5_variables = [file for file in to_dos if int(file.split('_')[1][1:]) == 5]

        # we create a dictionary with the lists of files to be processed
        todos = {'5': to_dos_5_variables} # , '10': to_dos_10_variables, '25': to_dos_25_variables

        # we create a dictionary to store the results
        dfs = []
        descriptors_root = self.descr_folder + f'estimate_{i}/'

        # Create the folder if it doesn't exist
        if not os.path.exists(descriptors_root):
            os.makedirs(descriptors_root)

        # Re-save with protocol 4
        for testing_file in sorted(os.listdir(descriptors_root)):
            if testing_file.endswith('.pkl'):
                file_path = os.path.join(descriptors_root, testing_file)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Re-save with protocol 4
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f, protocol=4)

        # This loop gets a list of all the files to be processed
        for testing_file in sorted(os.listdir(descriptors_root)):
            if testing_file.endswith('.pkl'):
                df = pd.read_pickle(os.path.join(descriptors_root, testing_file))
                if isinstance(df, pd.DataFrame):
                    dfs.append(df)

        # we concatenate the descriptors
        descriptors_training = pd.concat(dfs, axis=0).reset_index(drop=True)

        return descriptors_training, todos
    
    def classifier(self, i, descriptors_training, todos):

        for n_vars, todo in todos.items():

            # WE KEEP ONLY THE ROC-AUC SCORE FOR SYMPLICITY
            m1 = f'Estimate_{i}_rocs_process'
            # m2 = f'Estimate_{i}_precision_process'
            # m3 = f'Estimate_{i}_recall_process'
            # m4 = f'Estimate_{i}_f1_process'

            globals()[m1] = {}
            # globals()[m2] = {}
            # globals()[m3] = {}
            # globals()[m4] = {}
            causal_df = {}

            for testing_file in tqdm(todo):
                gen_process_number = int(testing_file.split('_')[0][1:])
                n_variables = int(testing_file.split('_')[1][1:])
                max_neighborhood_size = int(testing_file.split('_')[2][2:])
                noise_std = float(testing_file.split('_')[3][1:-4])

                # split training and testing data
                training_data = descriptors_training.loc[descriptors_training['process_id'] != gen_process_number]
                X_train = training_data.drop(columns=['process_id', 'graph_id', 'n_variables', 'max_neighborhood_size','noise_std', 'edge_source', 'edge_dest', 'is_causal'])
                y_train = training_data['is_causal']

                testing_data = descriptors_training.loc[(descriptors_training['process_id'] == gen_process_number) & (descriptors_training['n_variables'] == n_variables) & (descriptors_training['max_neighborhood_size'] == max_neighborhood_size) & (descriptors_training['noise_std'] == noise_std)]

                model = BalancedRandomForestClassifier(n_estimators=100, random_state=0, n_jobs=self.N_JOBS, max_depth=None, sampling_strategy='auto', replacement=True, bootstrap=False)
                # model = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=50, max_depth=10)

                model = model.fit(X_train, y_train)

                rocs = {}
                # precisions = {}
                # recalls = {}
                # f1s = {}
                causal_df_small = {}
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
                    causal_df_small[graph_id] = test_df
                    causal_df_small[graph_id]['y_pred_proba'] = y_pred_proba
                    causal_df_small[graph_id]['y_pred'] = y_pred

                causal_df[gen_process_number] = causal_df_small
                globals()[m1][gen_process_number] = rocs
                # globals()[m2][gen_process_number] = precisions
                # globals()[m3][gen_process_number] = recalls
                # globals()[m4][gen_process_number] = f1s

        # pickle everything
        output_folder = self.results_folder + f'estimates/estimate_{i}/'

        # Create the folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with open(os.path.join(output_folder, f'journal_results_td2c_R_N5.pkl'), 'wb') as f:
            everything = (globals()[m1], causal_df) #, globals()[m2], globals()[m3], globals()[m4]
            pickle.dump(everything, f)

        return globals()[m1], causal_df

    def load_results(self, i, roc_scores):

        input_folder = self.results_folder + f'estimates/estimate_{i}/'

        with open(os.path.join(input_folder, f'journal_results_td2c_R_N5.pkl'), 'rb') as f:
            roc_score, causal_df = pickle.load(f) # , TD2C_1_precision_process, TD2C_1_recall_process, TD2C_1_f1_process


        # Add roc score to the list of roc scores and print it 
        roc = pd.DataFrame(roc_score).mean().mean()
        roc_scores.append(roc)
        
        return roc, roc_scores, causal_df
    
    def stopping_criteria_1(self, i, roc, roc_scores, stop_1):
        
        # STOPPING CRITERIA 1: Using ROC-AUC score
        if self.strategy != "Random":
            print()
            print(f'ROC-AUC score: {round(roc, 4)}')
            print()
        
            if i == 0:
                if roc < 0.5:
                    print('ROC-AUC is too low, let\'s stop here.')
                    return
            elif i > 0:
                if roc <= roc_scores[0]:
                    stop_1 = stop_1 + 1
                    if stop_1 == 5:
                        print()
                        print('Estimations are not improving, let\'s stop here.')
                        print()
                        return

                else:
                    stop_1 = 0
                
                if roc <= roc_scores[0] - 0.1:
                    print()
                    print(f'ROC-AUC score: {round(roc, 4)}')
                    print()
                    print('Estimations are not improving, let\'s stop here.')
                    print()
                    return
        else:
            print()
            print(f'ROC-AUC score: {round(roc, 4)}')
            print()

    def reshape_causal_df(self, i, causal_df, causal_dfs, roc_scores, si, th):
  
        if i > 0 and roc_scores[i] == roc_scores[i-1]:
            si += 1

        if i > 0 and roc_scores[i] == roc_scores[i-1]:
            th = th - 0.1
            th = max(th, 0.5)

        for process_id, process_data in causal_df.items():
            for graph_id, graph_data in process_data.items():

                # keep only top k y_pred_proba
                graph_data = graph_data[graph_data['y_pred_proba'] > th]

                graph_data = graph_data.nlargest(si, 'y_pred_proba')
                graph_data = graph_data[['process_id', 'graph_id', 'edge_source', 'edge_dest', 'y_pred_proba']]
                graph_data.reset_index(drop=True, inplace=True)
                # assign the processed graph_data back to causal_df_1
                causal_df[process_id][graph_id] = graph_data

        print(f'Threshold: {th}')

        if i == 0:
            causal_dfs[i] = causal_df

            return causal_df, causal_dfs
        
        else:
            # set of 'edge_source'-'edge_dest' in causal_df
            edges_now = set()
            for process_id, process_data in causal_df.items():
                for graph_id, graph_data in process_data.items():
                    for index, row in graph_data.iterrows():
                        edges_now.add((row['process_id'], row['graph_id'], row['edge_source'], row['edge_dest'], row['y_pred_proba']))
                        
            # set of 'edge_source'-'edge_dest' in causal_df[i-1]
            edges_old = set()
            for process_id, process_data in causal_dfs[i-1].items():
                for graph_id, graph_data in process_data.items():
                    for index, row in graph_data.iterrows():
                        edges_old.add((row['process_id'], row['graph_id'], row['edge_source'], row['edge_dest'], row['y_pred_proba']))

            # find the difference between the two sets
            edges_diff = edges_now - edges_old
            
            causal_df = causal_dfs[i-1]
        
            # add the difference to causal_df
            for edge in edges_diff:
                process_id = edge[0]
                graph_id = edge[1]
                edge_source = edge[2]
                edge_dest = edge[3]
                y_pred_proba = edge[4]
                # causal_df[process_id][graph_id] = causal_df[process_id][graph_id].append({'process_id': process_id, 'graph_id': graph_id, 'edge_source': edge_source, 'edge_dest': edge_dest, 'y_pred_proba': y_pred_proba}, ignore_index=True)
                causal_df[process_id][graph_id] = pd.concat([causal_df[process_id][graph_id],pd.DataFrame([{'process_id': process_id, 'graph_id': graph_id, 'edge_source': edge_source, 'edge_dest': edge_dest, 'y_pred_proba': y_pred_proba}])])
                causal_df[process_id][graph_id] = causal_df[process_id][graph_id].drop_duplicates(subset=['process_id', 'graph_id', 'edge_source', 'edge_dest'], keep='first')
                causal_df[process_id][graph_id].reset_index(drop=True, inplace=True)

            causal_dfs[i] = causal_df

            return causal_df, causal_dfs

    def save_causal_df(self, i, causal_df):

        # output folder
        output_folder = os.path.join(self.results_folder, f'causal_dfs/single/')

        # Create the folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save the causal_df and causal_dfs in a pickle file
        with open(os.path.join(output_folder, f'it_{i}_causal_df_top_{self.k}_td2c_R_N5.pkl'), 'wb') as f:
            pickle.dump(causal_df, f)

    def print_best_edges(self, causal_df, i):

        ################ Unify causal_df #################

        # create a dataframe with all the causal_df
        dfs = []
        causal_unif = pd.DataFrame()
        
        # create a list of dataframes with all the causal_df
        for process_id, process_data in causal_df.items():
            for graph_id, graph_data in process_data.items():
                dfs.append(graph_data)

        # concatenate all the dataframes in the list in one dataframe
        causal_unif = pd.concat(dfs, axis=0).reset_index(drop=True)
        # sort in ascending order by process_id, graph_id, edge_source and edge_dest
        causal_unif.sort_values(by=['process_id', 'graph_id', 'edge_source', 'edge_dest', 'y_pred_proba'], ascending=[True, True, True, True, False], inplace=True)
        # keep top 5
        causal_unif = causal_unif.nlargest(5, 'y_pred_proba')
        # reset index
        causal_unif.reset_index(drop=True, inplace=True)

        
        ################ Print Results #################

        if self.strategy == "Random":
            print()
            print(f'Example of most relevant Edges that will be added in the next iteration:')
            print(causal_df)
            print()

        else:
            # Find the causal_df with the highest size
            biggest_causal_df = None
            shape = 0

            for process_id, process_data in causal_df.items():
                for graph_id, graph_data in process_data.items():
                    if graph_data.shape[0] > shape:
                        biggest_causal_df = causal_df[process_id][graph_id]
                        shape = graph_data.shape[0]

            # Print
            if i != self.it:
                print()
                print(f'Example of most relevant Edges that will be added in the next iteration:')
                print(biggest_causal_df)
                print()
                print("Oerall most relevant Edges found in the last iteration:")
                print(causal_unif)
                print()
            else:
                print()
                print(f'Example of most relevant Edges for the last iteration:')
                print(biggest_causal_df)
                print()
                print("Oerall most relevant Edges found in the last iteration:")
                print(causal_unif)
                print()
                print('End of iterations.')
#####

    def save_causal_dfs(self, i, causal_dfs):
            
            # output folder
            output_folder = os.path.join(self.results_folder, f'causal_dfs/dictionary/')
    
            # Create the folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
    
            # Save the causal_df and causal_dfs in a pickle file
            with open(os.path.join(output_folder, f'it_{i}_causal_dfs_top_{self.k}_td2c_R_N5.pkl'), 'wb') as f:
                pickle.dump(causal_dfs, f)
    
    def plot_results(self, roc_scores):

        if  self.method == None or self.k == None or self.it == None or self.COUPLES_TO_CONSIDER_PER_DAG == None or roc_scores == None or self.strategy == None:
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
            plt.title(f'ROC-AUC scores for Iterative {self.method} ({len(roc_scores)} iterations and {self.k} top vars) with Regression MI (5 vars processes) ({self.COUPLES_TO_CONSIDER_PER_DAG} couples per dag)')
            plt.xlabel('Iterations')
            plt.ylabel('ROC-AUC score')
            plt.grid()
            plt.legend()
            plt.tight_layout()

            # save the plot in folder
            output_folder = os.path.join(self.results_folder, 'plots/')
            plt.savefig(output_folder + f'ROC_AUC_scores_TD2C_{self.method}_{len(roc_scores)}_iterations_{self.k}_top_vars_{self.COUPLES_TO_CONSIDER_PER_DAG}_couples_per_dag_{self.strategy}.pdf')
            plt.show()

    def df_scores(self, roc_scores):

        if roc_scores == None or self.method == None or self.it == None or self.COUPLES_TO_CONSIDER_PER_DAG == None or self.k == None or self.strategy == None:
            print('Please run iterative_td2c() function first')
            return
        else:
            print()
            print('Resultant ROC-AUC scores:')
            print()
            roc_scores_df = pd.DataFrame(roc_scores, columns=['roc_score'])
            roc_scores_df['iteration'] = range(0, len(roc_scores))

            # save the df in a csv file
            output_folder = os.path.join(self.results_folder, 'metrics/')
            roc_scores_df.to_csv(output_folder + f'roc_scores_TD2C_{self.method}_{len(roc_scores)}_iterations_{self.k}_top_vars_{self.COUPLES_TO_CONSIDER_PER_DAG}_couples_per_dag_{self.strategy}.csv', index=False)
            print(roc_scores_df)
        
        return roc_scores_df
    
    def plot_ground_truth(self, roc_scores, causal_dfs):
        return