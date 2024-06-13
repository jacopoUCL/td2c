"""
This script can be used to iterate over the datasets of a particular experiment.
Below you import your function "my_method" stored in the module causeme_my_method.

Importantly, you need to first register your method on CauseMe.
Then CauseMe will return a hash code that you use below to identify which method
you used. Of course, we cannot check how you generated your results, but we can
validate a result if you upload code. Users can filter the Ranking table to only
show validated results.
"""
import numpy as np
import pandas as pd
import json
import zipfile
import bz2
import time
import os

from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

import sys
sys.path.append("..")
sys.path.append("../d2c/")
from descriptors.d2c import D2C
from descriptors.utils import create_lagged_multiple_ts

from benchmark import BenchmarkRunner, VAR, VARLiNGAM, PCMCI, Granger, DYNOTEARS

def competitor_method(data, method, maxlags=1, n_variables =3):
    data = [pd.DataFrame(data)]
    if method == 'granger':
        competitor = Granger(data, maxlags=maxlags, n_jobs=1, ground_truth=None).run()
    elif method == 'pcmci':
        competitor =PCMCI(data, maxlags=maxlags, n_jobs=1, ground_truth=None).run()
    elif method == 'var':
        competitor = VAR(data, maxlags=maxlags, n_jobs=1, ground_truth=None).run()
    elif method == 'varlingam':
        competitor = VARLiNGAM(data, maxlags=maxlags, n_jobs=1, ground_truth=None).run()
    elif method == 'dynotears':
        competitor = DYNOTEARS(data, maxlags=maxlags, n_jobs=1, ground_truth=None).run()
    
    competitor.build_causeme_matrices(n_variables=n_variables)
    causal_matrices, p_value_matrices, lag_matrices = competitor.get_causeme_matrices()
    val_matrix = causal_matrices[0]
    p_matrix = p_value_matrices[0]
    lag_matrix = lag_matrices[0]

    return val_matrix, p_matrix, lag_matrix

# Your method must be called 'my_method'
# Describe all parameters (except for 'data') in the method registration on CauseMe
def my_method(data, clf=RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0), maxlags=1, n_variables=3, correct_pvalues=True):

    ##TODO: what is correct_pvalues? 

    # Input data is of shape (time, variables)
    T, N = data.shape

    data_df = pd.DataFrame(data)

    lagged_data = create_lagged_multiple_ts([data_df], maxlags) #TODO: how could we not have this?


    d2c_test = D2C(None,lagged_data,maxlags=maxlags,n_variables=n_variables)
    X_test = d2c_test.compute_descriptors_no_dags()
    

    test_df = pd.DataFrame(X_test)
    # test_df = test_df.drop(['graph_id', 'edge_source', 'edge_dest'], axis=1)
    test_df = test_df.drop(['graph_id','edge_source','edge_dest'], axis=1)
    
    
    y_pred = clf.predict_proba(test_df)[:,1]
    returned = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_pred, columns=['is_causal'])], axis=1)
    of_interest = returned[['edge_source','edge_dest','is_causal']]
    
    #we need to select the best lag (the one with highest score)

    extended_val_matrix = np.zeros((n_variables * (maxlags + 1), n_variables), dtype='float32')
    
    for index, row in of_interest.iterrows():
        source =int(row['edge_source'])
        dest = int(row['edge_dest'])
        weight = row['is_causal']
        extended_val_matrix[source, dest] = weight


    val_matrix = np.zeros((N, N), dtype='float32')
    lag_matrix = np.zeros((N, N), dtype='float32')

    for i in range(n_variables):
        for j in range(n_variables):
            # Extract all values related to the causal influence of variable i on j across all lags
            values = extended_val_matrix[i::n_variables, j] #rows from i to the end, with step n_variables
            # Update the causal_matrix with the maximum value found
            val_matrix[i, j] = np.max(values)
            # Update the lag_matrix with the lag corresponding to the maximum value found
            lag_matrix[i, j] = np.argmax(values)

    thresholded_val_matrix = val_matrix.copy()
    thresholded_val_matrix[thresholded_val_matrix < 0.5] = 0
    thresholded_val_matrix[thresholded_val_matrix >= 0.5] = 1

    return val_matrix, 1 - thresholded_val_matrix, lag_matrix #as pvalues we return 1 - val_matrix



def process_zip_file(name, method, clf, maxlags=1, n_variables=3):
    print("\rRun on {}".format(name), end='', flush=True)
    data = np.loadtxt('experiments/'+name[:-9]+'/'+name)
    
    # Runtimes for your own assessment
    start_time = time.time()

    # Run your method (adapt parameters if needed)
    if method == 'd2cpy':
        val_matrix, p_matrix, lag_matrix = my_method(data, clf, maxlags,n_variables, correct_pvalues=True)
    else:
        val_matrix, p_matrix, lag_matrix = competitor_method(data, method, maxlags, n_variables)
    runtime = time.time() - start_time

    # Convert the matrices to the required format and return
    score = val_matrix.flatten()
    pvalue = p_matrix.flatten() if p_matrix is not None else None
    lag = lag_matrix.flatten() if lag_matrix is not None else None

    return score, pvalue, lag, runtime


if __name__ == '__main__':
    

    shas = {"granger":"d7e314aaffaf428b912f046eebf77a19",
            "pcmci":"8cdc476ef289424c9389a83d1a0a16c3",
            "var":"5892502d5b3a441eb0dabdcf04053627",
            "varlingam":"cf7dff649c80409381d6c757ae12b76e",
            "dynotears":"4a4ebee6f75e4c29aba7a8f13ac66d39",
            "d2cpy":"0931a3e645e3436b89c56f5e1274dcb7"
            }

    for method_key in shas.keys():
        
        method_name = method_key
        method_sha = shas[method_name]

        if method_name == 'pcmci':
            n_jobs = 1
        else: 
            n_jobs = 50



        if method_name == 'd2cpy':
            training_data = pd.read_pickle('2024.01.20.pkl')
            X_train = training_data.drop(['graph_id', 'edge_source', 'edge_dest', 'is_causal'], axis=1)
            y_train = training_data['is_causal']

            clf = BalancedRandomForestClassifier(n_estimators=20, max_depth=10, n_jobs=1, random_state=0,sampling_strategy='all', replacement=True)
            clf.fit(X_train, y_train)


        for file in os.listdir('experiments/'):
            if not file.endswith('.zip'):
                continue

            # Setup a python dictionary to store method hash, parameter values, and results
            results = {}
            results['method_sha'] = method_sha

            maxlags = 5 #general 
            n_variables = int(file.split('N-')[1].split('_')[0])


            results['parameter_values'] = "maxlags=%d" % maxlags
            results['model'] = file.split('_N-')[0]

            experimental_setup = file.split(results['model'])[1].split('.zip')[0][1:] #remove the first underscore


            results['experiment'] = results['model'] + '_' + experimental_setup

            # Adjust save name if needed
            save_name = '{}_{}_{}'.format(method_name,
                                        results['parameter_values'],
                                        results['experiment'])

            # Setup directories (adjust to your needs)
            experiment_folder = './experiments/'
            results_folder = './results/'
            unzip_folder = results['experiment']


            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            if not os.path.exists(experiment_folder):
                os.makedirs(experiment_folder)

            if not os.path.exists(unzip_folder):
                os.makedirs(unzip_folder)

            experiment_zip = experiment_folder+'%s.zip' % results['experiment']
            results_file = results_folder+'%s.json.bz2' % (save_name)

            #################################################

            # Start of script
            scores = []
            pvalues = []
            lags = []
            runtimes = []

            print("Load data")
            # This will hold results from all processes
            results_from_mp = []

            with zipfile.ZipFile(experiment_zip, "r") as zip_ref:
                #unzip the files and make a list
                zip_ref.extractall("experiments/"+unzip_folder)
                names = sorted(zip_ref.namelist())

            if method_name == 'd2cpy':
                args_list = [(name, method_name, clf, maxlags, n_variables) for name in names]
            else:
                args_list = [(name, method_name, None, maxlags, n_variables) for name in names]
            # Create a pool of worker processes
            if n_jobs > 1:
                with Pool(processes=n_jobs) as pool:
                    results_from_mp = pool.starmap(process_zip_file, args_list)
            else:
                for args in args_list:
                    results_from_mp.append(process_zip_file(*args))

            # Extract the results to the original lists
            scores, pvalues, lags, runtimes = [], [], [], []
            for result in results_from_mp:
                score, pvalue, lag, runtime = result
                scores.append(score)
                if pvalue is not None: pvalues.append(pvalue)
                if lag is not None: lags.append(lag)
                runtimes.append(runtime)

            # Store arrays as lists for json
            results['scores'] = np.array(scores).tolist()
            if len(pvalues) > 0: results['pvalues'] = np.array(pvalues).tolist()
            if len(lags) > 0: results['lags'] = np.array(lags).tolist()
            results['runtimes'] = np.array(runtimes).tolist()

            # Save data
            print('Writing results ...')
            results_json = bytes(json.dumps(results), encoding='latin1')
            with bz2.BZ2File(results_file, 'w') as mybz2:
                mybz2.write(results_json)
