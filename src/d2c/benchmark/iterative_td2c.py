
# PACKAGE IMPORTS #################################################################################
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import pickle 
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_auc_score

from d2c.data_generation.builder import TSBuilder
from d2c.descriptors_generation import D2C, DataLoader
from d2c.benchmark import D2CWrapper


# PARAMETERS #################################################################################

N_JOBS = 40 # number of jobs to run in parallel. For D2C, parallelism is implemented at the observation level: each observation from a single file is processed in parallel
SEED = 42 # random seed for reproducibility
MB_SIZE = 2 # size to consider when estimating the markov blanket. This is only useful if the MB is actually estimated
COUPLES_TO_CONSIDER_PER_DAG = -1 # edges that are considered in total to compute descriptors, for each TS. This can speed up the process. If set to -1, all possible edges are considered
maxlags = 5 # maximum lags to consider when considering variable couples
noise_std_filter = 0.01  # Example noise standard deviation to filter
max_neighborhood_size_filter = 2  # Example filter for neighborhood size


# ITERATIVE TD2C FUNCTION #################################################################################
def iterative_td2c(method = 'ts', k = 1, it = 3, top_vars = 3, treshold = True, treshold_value = 0.9, size_causal_df = 5,
                   data_folder = 'home/data/', descr_folder = 'home/descr/', results_folder = 'home/results/'):
    """
    # This function requires data already generated and stored in the data folder

    # Methods:
        # 'ts' = for classic TD2C
        # 'original' = for original D2C
        # 'ts_rank' = for TD2C with ranking
        # 'ts_rank_2' = for TD2C with ranking 2
        # 'ts_rank_3' = for TD2C with ranking 3
        # 'ts_rank_4' = for TD2C with ranking 4
        # 'ts_rank_5' = for TD2C with ranking 5
        # 'ts_rank_6' = for TD2C with ranking 6
        # 'ts_rank_7' = for TD2C with ranking 7
        # 'ts_past' = for TD2C with past and future nodes
        # 'ts_rank_no_count' = for TD2C with ranking with no contemporaneous nodes

    Parameters:
    # k is the number of top variables to keep at each iteration for each DAG (keep = 1 if treshold = False)
    # it is the limit for the number of iterations to perform
    # top_vars is the number of top variables to keep in case of TD2C Ranking
    # treshold is a boolean to determine if we want to keep in the causal df the variables with a pred.proba higher than treshold_value
    # treshold_value is the value to keep the variables in the causal df
    # size_causal_df is the number of variables to keep in the causal_df in case of treshold = False
    # data_folder is the folder where the data is stored
    # descr_folder is the folder where the descriptors are stored
    # results_folder is the folder where the results are stored

    Stopping Criteria:
     1. if average ROC-AUC score does not improve or is the same as the previous iteration for 3 consecutive iterations
     2. if the first iteration has an average ROC-AUC score lower than 0.5
     3. if the average ROC-AUC score is more than 0.2 points lower than the first iteration
     4. if causal df is the same as the previous one for 3 consecutive iterations

    Output:
     1. Plot of average ROC-AUC scores (saved in results folder as pdf file)
     2. Average ROC-AUC scores for each iteration (saved in results folder as csv file)
    """
    
    iter_df = pd.DataFrame
    stop_1 = 0
    stop_2 = 0
    roc_scores = []

    print()
    print(f'Iterative TD2C - Method: {method} - Max iterations: {it} - Variables to keep per DAG: {k} - Top Variables: {top_vars} - Treshold: {treshold} - Size of Causal DF: {size_causal_df}')
    print()
    if COUPLES_TO_CONSIDER_PER_DAG == -1 and size_causal_df == 5 and treshold == False:
        print('Using all couples for each DAG')
        print(f'This iteration will take approximately {8.5*it} minutes')
        print()
    elif COUPLES_TO_CONSIDER_PER_DAG == -1 and treshold == True:
        print(f'Using all couples for each DAG and a pred.proba higher than {treshold_value}')
        print(f'This iteration will take approximately {10.5*it} minutes')
        print()
    elif COUPLES_TO_CONSIDER_PER_DAG != -1 and size_causal_df == 5:
        print(f'Using the top {COUPLES_TO_CONSIDER_PER_DAG} couples for each DAG')
        print(f'This iteration will take approximately {4*it} minutes')
        print()
    elif COUPLES_TO_CONSIDER_PER_DAG != -1 and size_causal_df == 1:
        print(f'Using the top {COUPLES_TO_CONSIDER_PER_DAG} couples for each DAG')
        print(f'This iteration will take approximately {3.5*it} minutes')
        print()

    print("Do you want to continue with the rest of the function? (y/n): ")

    response = input("Do you want to continue with the rest of the function? (y/n): ").strip().lower()

    if response in ['yes', 'y', 'Yes', 'Y']:
        print()
        print("Ok! Let's start the iteration.")
        print()

        # Estimation For Cycle
        for i in range(1,it+1):

            print()
            print(f'----------------------------  Estimation {i}  ----------------------------')
            print()

            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            input_folder = data_folder
            
            output_folder = descr_folder + f'estimate_{i}/'
            
            # Descriptors Generation #############################################################################
            # List of files to process
            to_process = []

            # Filtering the files to process
            for file in sorted(os.listdir(input_folder)):
                gen_process_number = int(file.split('_')[0][1:])
                n_variables = int(file.split('_')[1][1:])
                max_neighborhood_size = int(file.split('_')[2][2:])
                noise_std = float(file.split('_')[3][1:-4])

                if noise_std != noise_std_filter or max_neighborhood_size != max_neighborhood_size_filter:
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

                dataloader = DataLoader(n_variables=n_variables, maxlags=maxlags)
                dataloader.from_pickle(input_folder + file)

                if i  == 1:
                    d2c = D2C(
                        observations=dataloader.get_observations(),
                        dags=dataloader.get_dags(),
                        couples_to_consider_per_dag=COUPLES_TO_CONSIDER_PER_DAG,
                        MB_size=MB_SIZE,
                        n_variables=n_variables,
                        maxlags=maxlags,
                        seed=SEED,
                        n_jobs=N_JOBS,
                        full=True,
                        quantiles=True,
                        normalize=True,
                        cmi='original',
                        mb_estimator=method,
                        top_vars=top_vars
                    )
                
                else:
                    d2c = D2C(
                        observations=dataloader.get_observations(),
                        dags=dataloader.get_dags(),
                        couples_to_consider_per_dag=COUPLES_TO_CONSIDER_PER_DAG,
                        MB_size=MB_SIZE,
                        n_variables=n_variables,
                        maxlags=maxlags,
                        seed=SEED,
                        n_jobs=N_JOBS,
                        full=True,
                        quantiles=True,
                        normalize=True,
                        cmi='original',
                        mb_estimator= 'iterative',
                        top_vars=top_vars,
                        causal_df=iter_df
                    )

                d2c.initialize()  # Initializes the D2C object
                descriptors_df = d2c.get_descriptors_df()  # Computes the descriptors

                # Save the descriptors along with new DAGs if needed
                descriptors_df.insert(0, 'process_id', gen_process_number)
                descriptors_df.insert(2, 'n_variables', n_variables)
                descriptors_df.insert(3, 'max_neighborhood_size', max_neighborhood_size)
                descriptors_df.insert(4, 'noise_std', noise_std)

                descriptors_df.to_pickle(output_folder + f'Estimate_{i}_P{gen_process_number}_N{n_variables}_Nj{max_neighborhood_size}_n{noise_std}_MB{MB_SIZE}.pkl')

            # Set Classifier #################################################################################
            data_root = data_folder

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
            descriptors_root = descr_folder + f'estimate_{i}/'

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
            output_folder = results_folder + f'journals/estimate_{i}/'

            # Create the folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            with open(os.path.join(output_folder, f'journal_results_td2c_R_N5.pkl'), 'wb') as f:
                everything = (globals()[m1], causal_df_1) #, globals()[m2], globals()[m3], globals()[m4]
                pickle.dump(everything, f)

            # Load results #####################################################################################
            input_folder = results_folder + f'journals/estimate_{i}/'

            with open(os.path.join(input_folder, f'journal_results_td2c_R_N5.pkl'), 'rb') as f:
                TD2C_1_rocs_process, causal_df_1 = pickle.load(f) # , TD2C_1_precision_process, TD2C_1_recall_process, TD2C_1_f1_process


            # STOPPING CRITERIA 2: Using ROC-AUC score
            roc = pd.DataFrame(TD2C_1_rocs_process).mean().mean()

            if i == 1:
                if roc > 0.5:
                    roc_first = roc
                else:
                    print('ROC-AUC is too low, let\'s stop here.')
                    break
            elif i > 1:
                if roc <= roc_0:
                    stop_2 = stop_2 + 1
                    if stop_2 == 3:
                        print()
                        print('Estimation are not improving, let\'s stop here.')
                        print()
                        break
                else:
                    stop_2 = 0
                
                if roc <= roc_first-0.2:
                    print()
                    print('Estimation are not improving, let\'s stop here.')
                    print()
                    break
            
            print()
            print(f'ROC-AUC score: {roc}')
            print()
            roc_scores.append(roc)
            roc_0 = roc

            # Reshape causal_df #################################################################################
            # keep only rows for top k y_pred_proba
            for process_id, process_data in causal_df_1.items():
                for graph_id, graph_data in process_data.items():
                    causal_df_1[process_id][graph_id] = graph_data.nlargest(k, 'y_pred_proba')

            # for each causal_df keep only process_id, graph_id, edge_source, edge_dest and y_pred_proba
            for process_id, process_data in causal_df_1.items():
                for graph_id, graph_data in process_data.items():
                    causal_df_1[process_id][graph_id] = graph_data[['process_id', 'graph_id', 'edge_source', 'edge_dest', 'y_pred_proba']]
                    causal_df_1[process_id][graph_id].reset_index(drop=True, inplace=True)

            # save the causal_df as a pkl file alone
            output_folder = results_folder + f'metrics/estimate_{i}/'

            # Create the folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            with open(os.path.join(output_folder, f'causal_df_top_{k}_td2c_R_N5.pkl'), 'wb') as f:
                pickle.dump(causal_df_1, f)

            # Unify causal_df #################################################################################
            input_folder = results_folder + f'metrics/estimate_{i}/'

            with open(os.path.join(input_folder, f'causal_df_top_{k}_td2c_R_N5.pkl'), 'rb') as f:
                causal_df_1 = pickle.load(f)

            # create a dataframe with all the causal_df
            dfs = []
            for process_id, process_data in causal_df_1.items():
                for graph_id, graph_data in process_data.items():
                    dfs.append(graph_data)

            causal_df_unif_1 = pd.concat(dfs, axis=0).reset_index(drop=True)

            # sort in ascending order by process_id, graph_id, edge_source and edge_dest
            causal_df_unif_1.sort_values(by=['process_id', 'graph_id', 'edge_source', 'edge_dest'], inplace=True)

            # # keep couples edge-dest,edge_source present for more than one dag (TO TRY)
            # causal_df_unif_1 = causal_df_unif_1[causal_df_unif_1.duplicated(subset=['edge_source', 'edge_dest'], keep=False)]

            # unique of causal_df_unif for couples of edge_source and edge_dest
            causal_df_unif_1 = causal_df_unif_1.drop_duplicates(subset=['edge_source', 'edge_dest'])

            # KEEP VARIABLES IN CAUSAL_DF FOR A TRESHOLD OR TOP N 
            if treshold == True:
                # drop rows with y_pred_proba < 0.7 (not necessary given the next step)
                causal_df_unif_1 = causal_df_unif_1[causal_df_unif_1['y_pred_proba'] >= treshold_value]
                if causal_df_unif_1.shape[0] > 1:
                    causal_df_unif_1 = causal_df_unif_1.nlargest(10, 'y_pred_proba')

            else:
                # if n row > 5, keep only the top 5 rows with highest y_pred_proba
                if causal_df_unif_1.shape[0] > 1:
                    causal_df_unif_1 = causal_df_unif_1.nlargest(size_causal_df, 'y_pred_proba')

            # index reset
            causal_df_unif_1.reset_index(drop=True, inplace=True)

            # STOPPING CRITERIA 1: if causal df is the same as the previous one for 2 consecutive iterations
            if causal_df_unif_1.equals(iter_df):
                stop_1 = stop_1 + 1
                if stop_1 == 2:
                    print()
                    print(f'No new edges to add in the next iteration')
                    print()
                    break
            else:
                stop_1 = 0

            # save the causal_df as a pkl file alone
            output_folder = results_folder + f'metrics/estimate_{i}/'

            # Create the folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            with open(os.path.join(output_folder, f'causal_df_top_{k}_td2c_R_N5_unified.pkl'), 'wb') as f:
                pickle.dump(causal_df_unif_1, f)

            iter_df = causal_df_unif_1

            print()
            print(f'Most relevant Edges that will be added in the next iteration:')
            print(causal_df_unif_1)
            print()

        # PLOT RESULTS #################################################################################
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, it+1), roc_scores, marker='o')
        if treshold == False:
            size = size_causal_df
            plt.title(f'ROC-AUC scores for Iterative {method} ({it} iterations and {size} top vars) with Regression MI (5 vars processes) ({COUPLES_TO_CONSIDER_PER_DAG} couples per dag)')
        else:
            size = k
            plt.title(f'ROC-AUC scores for Iterative {method} ({it} iterations and {size} top vars) with Regression MI (5 vars processes) ({COUPLES_TO_CONSIDER_PER_DAG} couples per dag)')
        plt.xlabel('Iterations')
        plt.ylabel('ROC-AUC score')
        plt.grid()
        plt.tight_layout()

        # save the plot in folder
        output_folder = '/home/jpalombarini/td2c/notebooks/contributions/td2c_extesions/results/Regression/try_complete_function/plots/'
        plt.savefig(output_folder + f'ROC_AUC_scores_TD2C_{method}_{it}_iterations_{size}_top_vars_{COUPLES_TO_CONSIDER_PER_DAG}_couples_per_dag.pdf')

        roc_scores_df = pd.DataFrame(roc_scores, columns=['roc_score'])
        roc_scores_df['iteration'] = range(1, it+1)

        # save the df in a csv file
        output_folder = '/home/jpalombarini/td2c/notebooks/contributions/td2c_extesions/results/Regression/try_complete_function/metrics/'
        roc_scores_df.to_csv(output_folder + f'roc_scores_TD2C_{method}_{it}_iterations_{size}_top_vars_{COUPLES_TO_CONSIDER_PER_DAG}_couples_per_dag.csv', index=False)

        plt.show()
    
    else:
        print()
        print("Wise choice! Change the parameters and try again.")
        return