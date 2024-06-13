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


class MarkovBlanketEstimator:
    def __init__(self, size=5, verbose=True):
        """
        Initializes the Markov Blanket Estimator with specified parameters.
        
        Parameters:
        - nmax (int): The maximum number of features to consider in the Markov Blanket.
        - verbose (bool): Whether to print detailed logs.
        """
        self.verbose = verbose
        self.size = size

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

        if regr:
            model = RidgeCV()
            model.fit(X, Y)
            importances = np.abs(model.coef_)
            ranked_indices = np.argsort(importances)[::-1]
        else:
            correlations = self.column_based_correlation(X, Y)
            ranked_indices = np.argsort(np.abs(correlations))[::-1]
        
        return ranked_indices

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
        
        order = self.rank_features(X, Y, regr=False)
        sorted_ind = candidates_positions[order]
        
        return sorted_ind[:self.size]
    
cache = Cache(maxsize=1024)  # Define cache size

def custom_hashkey(*args, **kwargs):
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

  

class MutualInformationEstimator: 

    def __init__(self, proxy='Ridge', proxy_params=None):
        """
        Initializes the Mutual Information Estimator with specified regression proxy and parameters.
        
        Parameters:
        - proxy (str): The name of the regression model to use ('Ridge' by default).
        - proxy_params (dict): Parameters for the regression model.
        """
        self.proxy = proxy
        self.proxy_params = proxy_params or {}

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
    
    def estimate(self, y, x1, x2=None, cv=2):
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
        


class LOWESS(BaseEstimator, RegressorMixin):
    def __init__(self, tau):
        self.tau = tau
        self.X_ = None
        self.y_ = None
        self.theta_ = None

    def wm(self, point, X):
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
        # Fit the model to the data
        self.X_ = np.append(X, np.ones(X.shape[0]).reshape(X.shape[0],1), axis=1)
        self.y_ = np.array(y).reshape(-1, 1)
        return self

    def predict(self, X):
        # Predict using the fitted model

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
