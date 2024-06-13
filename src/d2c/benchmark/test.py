import pandas as pd
import time

import sys
sys.path.append('../d2c')

from src.benchmark.d2c_wrapper import D2C
from src.benchmark.dynotears import DYNOTEARS
from src.benchmark.granger import Granger
from src.benchmark.pcmci import PCMCI
from src.benchmark.var import VAR
from src.benchmark.varlingam import VARLiNGAM

import seaborn as sns
import matplotlib.pyplot as plt

#suppress FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning) #TODO:  y_pred contains classes not in y_true

class BenchmarkRunner:
    def __init__(self, data, ground_truth, benchmarks, name, maxlags=3, n_jobs=1):
        self.data = data
        self.ground_truth = ground_truth
        self.benchmarks = benchmarks  # List of benchmark objects
        self.name = name
        self.maxlags = maxlags
        self.n_jobs = n_jobs
        self.results = []

        self.test_couples = None #used to store the subset of pairs of variables to test ! 

    def run_all(self):
        for benchmark in self.benchmarks:
            print(f"\nRunning {benchmark.__class__.__name__}")
            start_time = time.time()
            benchmark.run()
            if benchmark.__class__.__name__ == 'D2C':
                self.test_couples = benchmark.get_causal_dfs()
            else:
                benchmark.filter_causal_dfs(self.test_couples)
            elapsed_time = time.time() - start_time
            print(f"\n{benchmark.__class__.__name__} time: {round(elapsed_time, 2)} seconds")
            self.results.append(benchmark.evaluate())

    def save_results(self, path='../data/3vars/'):
        # Combine and save all results
        df_all_eval = pd.DataFrame(columns=['Model', 'Metric', 'Score'])
        for result in self.results:
            df_all_eval = pd.concat([df_all_eval, pd.DataFrame(result, columns=['Model', 'Metric', 'Score'])])

        df_all_eval.to_csv(path+f'{self.name}_scores.csv', index=False)

    def plot_results(self, path='../data/3vars/'):
        # Load results if not already loaded
        df_scores = pd.read_csv(path+f'{self.name}_scores.csv')

        sns.set_style("whitegrid")
        sns.set_palette("muted")  # Set the Seaborn
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Metric', y='Score', hue='Model', data=df_scores)
        plt.title("Comparison of Methods Across Different Metrics")
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        #bbox
        plt.tight_layout()

        plt.savefig(path+f'{self.name}_scores.png')
        # plt.show()

