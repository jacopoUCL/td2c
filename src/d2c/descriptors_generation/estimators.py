"""
This module contains a MarkovBlanketEstimator and a MutualInformationEstimator. 
"""
import numpy as np

from cachetools import cached, Cache
from cachetools.keys import hashkey

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr

from sklearn.base import BaseEstimator, RegressorMixin
import time


###################################################################################################################################
   # MB ESTIMATORS: BASED ON FEATURE RANKING, LAGGED VALUES AND SOME OTHER MB MIXING STRATEGIES           
################################################################################################################################### 

class MarkovBlanketEstimator:
    def __init__(self, size=5, n_variables=5, maxlags=5, verbose=True, top_vars=3):
        """
        Initializes the Markov Blanket Estimator with specified parameters.
        
        Parameters:
        - verbose (bool): Whether to print detailed logs.
        - size (int): The desired size of the Markov Blanket.
        - n_variables (int): The number of variables in the dataset.
        - maxlags (int): The maximum number of lags to consider in the time series analysis.
        """
        self.verbose = verbose
        self.size = size
        self.n_variables = n_variables
        self.maxlags = maxlags
        self.top_vars = top_vars

    def column_based_correlation(self, X, Y):
        """
        Computes Pearson correlation coefficients between each column in X and the vector Y.
        
        Parameters:
        - X (numpy.ndarray): The feature matrix.
        - Y (numpy.ndarray): The target vector.
        
        Returns:
        - numpy.ndarray: An array of correlation coefficients.
        """
        correlations = np.array([pearsonr(X[:, i], Y)[0] for i in range(X.shape[1])])
        return correlations

    def rank_features(self, X, Y, regr=False):
        """
        Ranks features in X based on their correlation or regression coefficient with Y.
        
        Parameters:
        - X (numpy.ndarray): The feature matrix.
        - Y (numpy.ndarray): The target vector.
        - nmax (int): The maximum number of features to rank (default is self.nmax).
        - regr (bool): Whether to use regression coefficients instead of correlations.
        
        Returns:
        - numpy.ndarray: Indices of the top-ranked features.
        """
        print(f"Shape of X: {X.shape}, Shape of Y: {Y.shape}")

        if regr:
            model = RidgeCV()
            model.fit(X, Y)
            importances = np.abs(model.coef_)
            ranked_indices = np.argsort(importances)[::-1]
        else:
            correlations = self.column_based_correlation(X, Y)
            ranked_indices = np.argsort(np.abs(correlations))[::-1]
        
        return ranked_indices

# MB estimation for d2c
    def estimate(self, dataset, node):
        """
        Estimates the Markov Blanket for a given node using feature ranking.
        
        Parameters:
        - dataset (numpy.ndarray): The dataset containing all variables.
        - node (int): The index of the target node for which to estimate the Markov Blanket.
        - size (int): The desired size of the Markov Blanket.
        
        Returns:
        - numpy.ndarray: Indices of the variables in the estimated Markov Blanket.
        """
        n = dataset.shape[1]
        candidates_positions = np.array(list(set(range(n)) - {node}))
        Y = dataset[:, node]
        
        # Exclude the target node from the dataset for ranking
        X = dataset[:, candidates_positions]
        
        order = self.rank_features(X, Y, regr=False) # rank features based on correlation, this is not done in 
                                                     # estimte_time_series, used in td2c
        sorted_ind = candidates_positions[order]
        
        return sorted_ind[:self.size]
    
# MB estimation for td2c
    def estimate_time_series(self, dataset, node):
        '''
        The idea is to leverage the fact that we are in a temporal context and we know that x_t-1 is in the markov blanked of x_t. As well as x_t+1. 
        Based on the position in the observations dataset; and the number of dimensions and maxlags, we can reconstruct the time series and estimate the markov blanket.
        This is how the dataset is structured:
        |X1,X2,X3|X1_t-1,X2_t-1,X3_t-1|X1_t-2,X2_t-2,X3_t-2|
        in fact, it's just (node + n_variables) and (node - n_variables) when they exist!
        '''
        print('Estimating MB for node', node)
        mb = np.array([])
        if node + self.n_variables < dataset.shape[1]:
            mb = np.append(mb, node + self.n_variables)
        if node - self.n_variables >= 0:
            mb = np.append(mb, node - self.n_variables)
        print('Markov Blanket:', mb)
        # make mb type int
        return mb.astype(int)

# MB estimation for td2c + ranking 
    def estimate_time_series_ranking(self, dataset, node):
        '''
        This method estimates the Markov Blanket for a given node doing the following steps:
        - select the first passed and future instant of the node for each candidate variable, as in estimate_time_series
        - performs a feature ranking as in estimate, giving the possibility to keep 1, 2 or 3 most important variables
        '''
        print('Estimating MB for node', node)
        mb = np.array([])
        if node + self.n_variables < dataset.shape[1]:
            mb = np.append(mb, node + self.n_variables)
        if node - self.n_variables >= 0:
            mb = np.append(mb, node - self.n_variables)
        print('Markov Blanket:', mb)
        # make mb type int
        mb = mb.astype(int)
        # rank the features
        n = dataset.shape[1] # here we declare n as the number of columns in the dataset
        candidates_positions = np.array(list(set(range(n)) - set(mb) - {node})) # here we are excluding the target node and the markov blanket from the candidates
        Y = dataset[:, node] # here we are using the target node as Y

        # Exclude the target node and the TD2C MB from the dataset for ranking
        X = dataset[:, candidates_positions] # here we are using the candidates as X

        order = self.rank_features(X, Y, regr=False) # rank features based on correlation between X and Y
        sorted_ind = candidates_positions[order] # get the sorted indices of the candidates
        # get the top top_vars most important variables
        top_variables = sorted_ind[:self.top_vars] # get the top_vars most important variables
        return np.append(mb, top_variables) # return the markov blanket and the top_vars most important variables
    

# MB estimation for td2c with extended ancestors (X_t-i and X_t+i, for i > 1)
    def estimate_time_series_extended(self, dataset, node):
        '''

        '''
        print('Estimating MB for node', node)
        mb = np.array([])
        if node + self.n_variables < dataset.shape[1]:
            mb = np.append(mb, node + self.n_variables)
        if node - self.n_variables >= 0:
            mb = np.append(mb, node - self.n_variables)
        print('Markov Blanket:', mb)
        # make mb type int
        mb = mb.astype(int)
        # extend the markov blanket
        for i in range(2, self.maxlags + 1):
            if node + i * self.n_variables < dataset.shape[1]:
                mb = np.append(mb, node + i * self.n_variables)
            if node - i * self.n_variables >= 0:
                mb = np.append(mb, node - i * self.n_variables)
        return mb

cache = Cache(maxsize=1024)  # Define cache size

def custom_hashkey(*args, **kwargs):
    """
    Custom hashkey function for caching based on the data and shape of the input arguments.
    """
    return hashkey(*(
        (arg.data.tobytes(), arg.shape) if isinstance(arg, np.ndarray) else arg
        for arg in args
    ), **kwargs)

@cached(cache, key=custom_hashkey)

def mse(X, y, cv):
    """
    Calculates the mean squared error (MSE) based on the prediction of Y from X using the specified regression model.
    The MSE is a proxy for the conditional entropy of Y given X. Higher MSE means higher uncertainty, therefore higher entropy.
    
    
    Parameters:
    - X (numpy.ndarray): The feature matrix.
    - Y (numpy.ndarray): The target vector.
    - cv (int): The number of cross-validation folds to use.

    Returns:
    - float: The MSE of the prediction.
    """
    X = X[:, np.newaxis] if X.ndim == 1 else X
    y = y[:, np.newaxis] if y.ndim == 1 else y

    # model = self.get_regression_model()
    neg_mean_squared_error_folds = cross_val_score(Ridge(alpha=1e-3), X, y, scoring='neg_mean_squared_error', cv=cv)
    return max(1e-3, -np.mean(neg_mean_squared_error_folds)) #we set 0.001 as a lower bound



###################################################################################################################################
   # MI ESTIMATORS: KNNCMI (td2c), THE ONES BASED ON ERROR ESTIMATION (LOWESS, RIDGE REG AND RF) (d2c) AND SOME MIXING STRATEGIES          
###################################################################################################################################  

# PROXY CLASS FOR MUTUAL INFORMATION ESTIMATION BY ESTIMATING THE ERROR IN PREDICTING Y FROM X (AND X2) AND KNNCMI
class MutualInformationEstimator: 
    def __init__(self, proxy='Ridge', proxy_params=None, k=3):
        """
        Initializes the Mutual Information Estimator with specified regression proxy and parameters.
        
        Parameters:
        - proxy (str): The name of the regression model to use ('Ridge' by default).
        - proxy_params (dict): Parameters for the regression model.
        """
        self.proxy = proxy
        self.proxy_params = proxy_params or {}
        self.k = k

    def get_regression_model(self):
        """
        Initializes the regression model based on the specified proxy and proxy parameters.
        
        Returns:
        - model: The regression model instance.
        """
        if self.proxy == 'Ridge':
            alpha = self.proxy_params.get('alpha', 1e-3)
            model = Ridge(alpha=alpha)
        elif self.proxy == 'LOWESS':
            tau = self.proxy_params.get('tau', 0.5)
            model = LOWESS(tau=tau)
        elif self.proxy == 'RF':
            raise NotImplementedError("Random Forest is not yet supported as a proxy model.")
        else: #TODO: Implement other regression models here based on the proxy value.
            raise ValueError(f"Unsupported proxy model: {self.proxy}")
        return model
    
    def estimate_original(self, dataset, y_index, x1_index, x2_index = None, cv=2):
        """
        Estimates the (normalized) conditional mutual information of x1 to y given x2. 
        
        For a rough approximation, assuming Gaussian distributed errors in a linear regression model, 
        we can consider H(y|x) to be proportional to the MSE. 
        As the error in predicting y from x increases (i.e., as the MSE increases), 
        the uncertainty of y given x also increases, reflecting a higher conditional entropy.
        
        Similarly, we consider H(y) to be proportional to the variance of y.

        The assumptions are strong and may not hold in practice, but they provide a simple and fast approximation.
        TODO: further explore the validity of the assumptions.


        Formulas:
        - H(y) ≈ Var(y) 
        - H(y|x) ≈ MSE(x,y)
        - I(x1; y) ≈ (H(y) − H(y|x1))/H(y)= 1 - MSE(x1,y) / Var(y)
        - I(x1; y|x2) ≈ [(H(y|x2) − H(y|x1, x2))] / H(y|x2) = 1 - MSE([x1,x2],y) / MSE(x2, y)

        Parameters:
        - y (numpy.ndarray): The target vector.
        - x1 (numpy.ndarray): The feature matrix for the first variable.
        - x2 (numpy.ndarray): The feature matrix for the second variable (optional).
        - cv (int): The number of cross-validation folds to use.

        Returns:
        - float: The estimated conditional mutual information.
        """

        y = dataset[:, y_index]
        x1 = dataset[:, x1_index]
        x2 = None if x2_index is None else dataset[:, x2_index]

        
        if x2 is None or x2.size == 0:  #- I(x1; y) ≈ (H(y) − H(y|x1))/H(y)= 1 - MSE(x1,y) / Var(y)
            entropy_y = max(1e-3, np.var(y)) #we set 0.001 as a lower bound
            entropy_y_given_x1 = mse(x1, y, cv=cv) 
            mutual_information = 1 - entropy_y_given_x1 / entropy_y 
            return max(0, mutual_information) #if negative, it means that knowing x1 brings more uncertainty to y (conditional entropy is higher than unconditional entropy)
        else: #- I(x1; y|x2) ≈ [(H(y|x2) − H(y|x1, x2))] / H(y|x2) = 1 - MSE([x1,x2],y) / MSE(x2, y)
            if y.size == 0 or x1.size == 0:
                return 0
            x1_2d = x1 if x1.ndim > 1 else x1[:, np.newaxis]
            x2_2d = x2 if x2.ndim > 1 else x2[:, np.newaxis]

            x1x2 = np.hstack((x1_2d, x2_2d)) 
            entropy_y_given_x2 = mse(x2, y, cv=cv) 
            entropy_y_given_x1_x2 = mse(x1x2, y, cv=cv) # how much information x1 and x2 together have about y
            mutual_information = 1 - entropy_y_given_x1_x2 / entropy_y_given_x2
            return max(0, mutual_information)
        
    def estimate_knn_cmi(self, dataset, y_index, x1_index, x2_index = None):
        """
        Estimates the conditional mutual information of x1 to y given x2 using the KNNCMI algorithm.
        """
        import knncmi
        import pandas as pd
        dataset = pd.DataFrame(dataset)
        #make columns strings
        dataset.columns = [str(i) for i in dataset.columns]

        # if x2_index is list and is empty, set it to None
        if x2_index is not None and isinstance(x2_index, list) and len(x2_index) == 0:
            x2_index = None

        # print(dataset)
        # print(dataset.columns)
        # print(y_index)
        # print(dataset.columns[y_index])
        y_name = [dataset.columns[y_index]]
        x1_name = [dataset.columns[x1_index]]
        x2_name = None if x2_index is None else list(dataset.columns[x2_index])
      
        # print(y_name)
        # print(x1_name)
        # print(x2_name)
        if x2_name is None:  
            return knncmi.cmi(y_name, x1_name, [], self.k, dataset)

        else: 
            return knncmi.cmi(y_name, x1_name, x2_name, self.k, dataset)

# LOWESS
class LOWESS(BaseEstimator, RegressorMixin):
    def __init__(self, tau):
        self.tau = tau
        self.X_ = None
        self.y_ = None
        self.theta_ = None

    def wm(self, point, X):
        """
        Calculates the weight matrix for the given point and dataset X.

        Parameters:
        - point (numpy.ndarray): The point for which to calculate the weights.
        - X (numpy.ndarray): The dataset for which to calculate the weights.

        Returns:
        - numpy.ndarray: The weight matrix.
        """
        # Calculate the squared differences in a vectorized way
        # point is reshaped to (1, -1) for broadcasting to match the shape of X
        differences = X - point.reshape(1, -1)
        squared_distances = np.sum(differences ** 2, axis=1)

        # Calculate the weights
        tau_squared = -2 * self.tau * self.tau
        weights = np.exp(squared_distances / tau_squared)

        # Create a diagonal matrix from the weights
        weight_matrix = np.diag(weights)

        return weight_matrix

    def fit(self, X, y):
        """
        Fits the LOWESS model to the data.
        """
        # Fit the model to the data
        self.X_ = np.append(X, np.ones(X.shape[0]).reshape(X.shape[0],1), axis=1)
        self.y_ = np.array(y).reshape(-1, 1)
        return self

    def predict(self, X):
        """
        Predicts the target variable for the input data.   
        """

        #allocate array of size X.shape[0]
        preds = np.empty(X.shape[0])
        X_ = np.append(X, np.ones(X.shape[0]).reshape(X.shape[0],1), axis=1)

        start = time.time()

        for i in range(X.shape[0]):
            point_ = X_[i] 
            w = self.wm(point_, self.X_)
            self.theta_ =  np.linalg.pinv(self.X_.T@(w @ self.X_))@self.X_.T@(w @ self.y_)
            pred = np.dot(point_, self.theta_)
            preds[i] = pred

        return preds.reshape(-1, 1)


# MERGING KNN-CMI AND REGRESSION METHODS:
# to potentially achieve more accurate or robust mutual information estimates.
# KNN-CMI is a non-parametric method that does not assume any specific functional form of the relationship between variables. 
# It works well for both linear and non-linear dependencies. However, its flexibility might lead to less precise estimates when the 
# underlying relationship is simple and linear.
# Regression methods, especially linear regression (like Ridge regression), are naturally suited to linear relationships. 
# They model the dependency between variables directly as a linear function, leading to more precise and interpretable estimates 
# of mutual information in such cases.

    # Ensemble Approach:
        # Combine Outputs: Calculate mutual information using both the KNN-CMI and regression methods separately, then combine the 
        #   results. This could involve averaging the estimates, or applying a weighted average depending on how 
        #   confident you are in each method for the specific dataset.
        # Adaptive Weighting: Assign different weights to the KNN-CMI and regression-based estimates depending on the 
        #   characteristics of the data (e.g., linearity, noise level, dimensionality).
        # Cross-Validation: You could use cross-validation to determine the optimal combination or weighting of the two methods. 
        #   For instance, if you notice that one method consistently performs better on certain data types, you 
        #   could prioritize that method in those cases.

    # Hybrid Method:
        # Feature Engineering: Use KNN-CMI as a non-parametric method to capture complex dependencies and use regression-based 
        #   methods to model linear or near-linear relationships. Combine the strengths of both by integrating 
        #   outputs or using KNN to preprocess the data before applying regression.
    
    # Sequential Approach:
        # Preprocessing: Use KNN-CMI to identify potential non-linear relationships and select relevant features. Then,
        #   apply regression-based methods to estimate mutual information on the reduced feature set.
        # Model Selection: Use KNN-CMI to guide the selection of appropriate regression models or hyperparameters based on the
        #   complexity of the relationships in the data.

    

# OTHER POSSIBLE METHODS FOR MI ESTIMATION TO IMPLEMENT:
    # 1. Parametric Methods
        # Gaussian Estimator: Assumes that the data follows a multivariate Gaussian distribution. Mutual information is then calculated based on the covariance matrix of the data.
        # Kernel Density Estimation (KDE): Uses kernel density estimates to approximate the probability densities of the variables and then computes mutual information from these densities.
    # 2. Non-Parametric Methods
        # Histogram-Based Estimators: Discretizes continuous data into bins and estimates mutual information based on the resulting histograms.
        # Maximal Information Coefficient (MIC): MIC is designed to capture a wide range of associations between variables, including linear and non-linear relationships.
    # 3. Entropy-Based Methods
        # Plug-in Estimator: Directly estimates entropy by plugging in empirical distributions from the data into the entropy formula.
        # Nearest-Neighbor Estimators: Estimates entropy and mutual information based on distances to the nearest neighbors in the data space (similar to KNN-CMI but focuses on entropy estimation).
    # 4. Model-Based Methods
        # Copula-Based Methods: Uses copulas to model the dependency structure between variables separately from their marginal distributions. Mutual information is then computed based on these models.
        # Information Bottleneck Method: Uses a neural network-based approach to compress the input data while preserving relevant information. Mutual information is estimated from the network's internal representations.
    # 5. Resampling-Based Methods
        # Bootstrap-Based Estimators: Estimates mutual information by resampling the data and computing empirical distributions.
    # 6. Adaptive Partitioning Methods
        # Adaptive Partitioning: Dynamically partitions the data space based on the density of observations, leading to an adaptive grid for estimating mutual information.
    # 7. Monte Carlo Estimators
        # Monte Carlo Integration: Estimates mutual information by integrating over the joint and marginal distributions using Monte Carlo methods.