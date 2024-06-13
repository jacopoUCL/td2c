import math
import networkx as nx

def mean_over_indices(Y_t, indices):
    """
    Calculate the mean of the elements in Y_t corresponding to the indices.
    """
    return sum(Y_t[i] for i in indices) / len(indices)

def I(condition):
    """
    Indicator function.
    """
    return 1 if condition else 0

def sign(x):
    """
    Sign function.
    """
    return -1 if x < 0 else (1 if x > 0 else 0)


def add_edges(G, T, N, N_j, time_from):
        """
        Allows for a modular creations of DAGs, 
        by simply specifying from which past time lags we should point to the current time.
        """

        #We always include all nodes, despite the existance of edges.
        for t in range(T+1):
            for j in range(N):
                G.add_node(f"Y[{t}][{j}]")

        for t in range(T):
            for j in range(N):
                for nj in N_j[j]:
                    for element in time_from:
                        if t >= element:
                            G.add_edge(f"Y[{t-element}][{nj}]", f"Y[{t+1}][{j}]")

        return G
class ModelRegistry:
    def __init__(self):
        self.registry = {}

    def register(self, model_id):
        def inner_wrapper(wrapped_class):
            if model_id in self.registry:
                print(f"Model ID {model_id} already exists. Overwriting.")
            self.registry[model_id] = wrapped_class
            return wrapped_class
        return inner_wrapper

    def get_model(self, model_id):
        model = self.registry.get(model_id)
        if model is None:
            raise ValueError(f"Model ID {model_id} does not exist.")
        return model

# Instantiate the registry
model_registry = ModelRegistry()

class BaseModel:
    def __init__(self):
        self.time_from = [None] #from t -> t+1

    @staticmethod
    def update(Y, t, j, N_j, W):
        pass

    def build_dag(self,T, N_j, N):
        return add_edges(nx.DiGraph(),T,N,N_j,self.time_from) 
    
    def get_maximum_time_lag(self):
        return max(self.time_from)

@model_registry.register(model_id=1)
class Model1(BaseModel):
    """
    Y_t+1[j] = -0.4 * (3 - (Y_bar_t[N_j])**2) / (1 + (Y_bar_t[N_j])**2) + 0.6 * (3 - (Y_bar_t-1[N_j] - 0.5)**3) / (1 + (Y_bar_t-1[N_j] - 0.5)**4) + W_t+1[j]
    """

    def __init__(self):
        self.time_from = [0,1] #from t -> t+1 and from t-1 -> t+1
        
    
    @staticmethod  
    def update(Y, t, j, N_j, W):
        Y_bar_t = mean_over_indices(Y[t], N_j)
        Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)

        term1 = -0.4 * (3 - Y_bar_t**2) / (1 + Y_bar_t**2)
        term2 = 0.6 * (3 - (Y_bar_t_minus_1 - 0.5)**3) / (1 + (Y_bar_t_minus_1 - 0.5)**4)

        return term1 + term2 + W[t][j]

@model_registry.register(model_id=2)
class Model2(BaseModel):
    """
    Y_t+1[j] = (0.4 - 2 * exp(-50 * Y_bar_t-1[N_j]**2)) * Y_bar_t-1[N_j] + (0.5 - 0.5 * exp(-50 * Y_bar_t-2[N_j]**2)) * Y_bar_t-2[N_j] + W_t+1[j]
    """


    def __init__(self):
        self.time_from = [1,2] #from t-1 -> t+1 and from t-2 -> t+1
        

    @staticmethod 
    def update(Y, t, j, N_j, W):
        Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)
        Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)

        term1 = (0.4 - 2 * math.exp(-50 * Y_bar_t_minus_1**2)) * Y_bar_t_minus_1
        term2 = (0.5 - 0.5 * math.exp(-50 * Y_bar_t_minus_2**2)) * Y_bar_t_minus_2

        return term1 + term2 + W[t][j]
    
@model_registry.register(model_id=3)
class Model3(BaseModel):
    """
    Y_t+1[j] = 1.5 * sin(pi / 2 * Y_bar_t-1[N_j]) - sin(pi / 2 * Y_bar_t-2[N_j]) + W_t+1[j]
    """

    def __init__(self):
        self.time_from = [1,2] #from t-1 -> t+1 and from t-2 -> t+1
        

    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)
        Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)

        term1 = 1.5 * math.sin(math.pi / 2 * Y_bar_t_minus_1)
        term2 = -math.sin(math.pi / 2 * Y_bar_t_minus_2)

        return term1 + term2 + W[t][j]

@model_registry.register(model_id=4)
class Model4(BaseModel):
    """
    Y_t+1[j] = 2 * exp(-0.1 * Y_bar_t[N_j]**2) * Y_bar_t[N_j] - exp(-0.1 * Y_bar_t-1[N_j]**2) * Y_bar_t-1[N_j] + W_t+1[j]
    """


    def __init__(self):
        self.time_from = [0,1] #from t -> t+1 and from t-1 -> t+1
        

    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t = mean_over_indices(Y[t], N_j)
        Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)

        term1 = 2 * math.exp(-0.1 * Y_bar_t**2) * Y_bar_t
        term2 = -math.exp(-0.1 * Y_bar_t_minus_1**2) * Y_bar_t_minus_1

        return term1 + term2 + W[t][j]

@model_registry.register(model_id=5)
class Model5(BaseModel):
    """
    Y_t+1[j] = -2 * Y_bar_t[N_j] * I(Y_bar_t[N_j] < 0) + 0.4 * Y_bar_t[N_j] * I(Y_bar_t[N_j] < 0) + W_t+1[j]
    """

    def __init__(self):
        self.time_from = [0] #from t -> t+1 
        

    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t = mean_over_indices(Y[t], N_j)

        term1 = -2 * Y_bar_t * I(Y_bar_t < 0)
        term2 = 0.4 * Y_bar_t * I(Y_bar_t < 0)

        return term1 + term2 + W[t][j]

@model_registry.register(model_id=6)
class Model6(BaseModel):
    """
    Y_t+1[j] = 0.8 * log(1 + 3 * Y_bar_t[N_j]**2) - 0.6 * log(1 + 3 * Y_bar_t-2[N_j]**2) + W_t+1[j]
    """

    def __init__(self):
        self.time_from = [0,2] #from t -> t+1, and from t-2 -> t+1
        

    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t = mean_over_indices(Y[t], N_j)
        Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)

        term1 = 0.8 * math.log(1 + 3 * Y_bar_t**2)
        term2 = -0.6 * math.log(1 + 3 * Y_bar_t_minus_2**2)

        return term1 + term2 + W[t][j]

@model_registry.register(model_id=7)
class Model7(BaseModel):
    """
    Y_t+1[j] = (0.4 - 2 * cos(40 * Y_bar_t-2[N_j]) * exp(-30 * Y_bar_t-2[N_j]**2)) * Y_bar_t-2[N_j] + (0.5 - 0.5 * exp(-50 * Y_bar_t-1[N_j]**2)) * Y_bar_t-1[N_j] + W_t+1[j]
    """

    def __init__(self):
        self.time_from = [1,2] #from t-1 -> t+1, and from t-2 -> t+1
        


    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)
        Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)

        term1_part1 = 0.4 - 2 * math.cos(40 * Y_bar_t_minus_2) * math.exp(-30 * Y_bar_t_minus_2**2)
        term1 = term1_part1 * Y_bar_t_minus_2
        term2 = (0.5 - 0.5 * math.exp(-50 * Y_bar_t_minus_1**2)) * Y_bar_t_minus_1

        return term1 + term2 + W[t][j]

@model_registry.register(model_id=8)
class Model8(BaseModel):
    """
    Y_t+1[j] = (0.5 - 1.1 * exp(-50 * Y_bar_t[N_j]**2)) * Y_bar_t[N_j] + (0.3 - 0.5 * exp(-50 * Y_bar_t-2[N_j]**2)) * Y_bar_t-2[N_j] + W_t+1[j]
    """

    def __init__(self):
        self.time_from = [0,2] #from t -> t+1 and from t-2 -> t+1
        


    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t = mean_over_indices(Y[t], N_j)
        Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)

        term1 = (0.5 - 1.1 * math.exp(-50 * Y_bar_t**2)) * Y_bar_t
        term2 = (0.3 - 0.5 * math.exp(-50 * Y_bar_t_minus_2**2)) * Y_bar_t_minus_2

        return term1 + term2 + W[t][j]

@model_registry.register(model_id=9)
class Model9(BaseModel):
    """
    Y_t+1[j] = 0.3 * Y_bar_t[N_j] + 0.6 * Y_bar_t-1[N_j] + (0.1 - 0.9 * Y_bar_t[N_j] + 0.8 * Y_bar_t-1[N_j]) / (1 + exp(-10 * Y_bar_t[N_j])) + W_t+1[j]
    """

    def __init__(self):
        self.time_from = [0,1] #from t -> t+1, and from t-1 -> t+1
        

    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t = mean_over_indices(Y[t], N_j)
        Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)

        term1 = 0.3 * Y_bar_t
        term2 = 0.6 * Y_bar_t_minus_1
        term3_numerator = 0.1 - 0.9 * Y_bar_t + 0.8 * Y_bar_t_minus_1
        term3_denominator = 1 + math.exp(-10 * Y_bar_t)
        term3 = term3_numerator / term3_denominator

        return term1 + term2 + term3 + W[t][j]

@model_registry.register(model_id=10)
class Model10(BaseModel):
    """
    Y_t+1[j] = sign(Y_bar_t[N_j]) + W_t+1[j]
    """

    def __init__(self):
        self.time_from = [0] #from t -> t+1
        

    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t = mean_over_indices(Y[t], N_j)
        return sign(Y_bar_t) + W[t][j]

@model_registry.register(model_id=11)    
class Model11(BaseModel):
    """
    Y_t+1[j] = 0.8 * Y_bar_t[N_j] - (0.8 * Y_bar_t[N_j]) / (1 + exp(-10 * Y_bar_t[N_j])) + W_t+1[j]
    """


    def __init__(self):
        self.time_from = [0] #from t -> t+1
        

    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t = mean_over_indices(Y[t], N_j)
        term1 = 0.8 * Y_bar_t
        term2_denominator = 1 + math.exp(-10 * Y_bar_t)
        term2 = -0.8 * Y_bar_t / term2_denominator

        return term1 + term2 + W[t][j]

@model_registry.register(model_id=12)
class Model12(BaseModel):
    """
    Y_t+1[j] = 0.3 * Y_bar_t[N_j] + 0.6 * Y_bar_t-1[N_j] + (0.1 - 0.9 * Y_bar_t[N_j] + 0.8 * Y_bar_t-1[N_j]) / (1 + exp(-10 * Y_bar_t[N_j])) + W_t+1[j]
    """

    def __init__(self):
        self.time_from = [0,1] #from t -> t+1 and from t-1 -> t+1
        


    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t = mean_over_indices(Y[t], N_j)
        Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)

        term1 = 0.3 * Y_bar_t
        term2 = 0.6 * Y_bar_t_minus_1
        term3_numerator = 0.1 - 0.9 * Y_bar_t + 0.8 * Y_bar_t_minus_1
        term3_denominator = 1 + math.exp(-10 * Y_bar_t)
        term3 = term3_numerator / term3_denominator

        return term1 + term2 + term3 + W[t][j]
    
@model_registry.register(model_id=13)
class Model13(BaseModel):
    """
    Y_t+1[j] = 0.38 * Y_bar_t[N_j] * (1 - Y_bar_t-1[N_j]) + W_t+1[j]
    """

    def __init__(self):
        self.time_from = [0,1] #from t -> t+1 and from t-1 -> t+1
        

    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t = mean_over_indices(Y[t], N_j)
        Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)

        term1 = 0.38 * Y_bar_t * (1 - Y_bar_t_minus_1)

        return term1 + W[t][j]
    
@model_registry.register(model_id=14)
class Model14(BaseModel):
    """
    Y_t+1[j] = -0.5 * Y_bar_t[N_j] if Y_bar_t[N_j] < 1 else 0.4 * Y_bar_t[N_j]
    """

    def __init__(self):
        self.time_from = [0] #from t -> t+1
        
    
    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t = mean_over_indices(Y[t], N_j)

        if Y_bar_t < 1:
            return -0.5 * Y_bar_t + W[t][j]
        else:
            return 0.4 * Y_bar_t + W[t][j]

@model_registry.register(model_id=15)    
class Model15(BaseModel):
    """
    Y_t+1[j] = 0.9 * Y_bar_t[N_j] + W_t+1[j] if abs(Y_bar_t[N_j]) < 1 else -0.3 * Y_bar_t[N_j] + W_t+1[j]
    """

    def __init__(self):
        self.time_from = [0] #from t -> t+1
        
    
    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t = mean_over_indices(Y[t], N_j)

        if abs(Y_bar_t) < 1:
            return 0.9 * Y_bar_t + W[t][j]
        else:
            return -0.3 * Y_bar_t + W[t][j]

@model_registry.register(model_id=16)    
class Model16(BaseModel):
    """
    Y_t+1[j] = -0.5 * Y_bar_t[N_j] + W_t+1[j] if x_t == 1 else 0.4 * Y_bar_t[N_j] + W_t+1[j]
    x_t+1 = 1 - x_t, x_0 = 1

    The exogenous variable x_t is a simple switch. 
    We implement it by checking the parity of t.
    """
    def __init__(self):
        self.time_from = [0] #from t -> t+1
        
    
    @staticmethod
    def update(Y, t, j, N_j, W):
        Y_bar_t = mean_over_indices(Y[t], N_j)

        if t % 2 == 0:
            return -0.5 * Y_bar_t + W[t][j]
        else:
            return 0.4 * Y_bar_t + W[t][j]

@model_registry.register(model_id=17)    
class Model17(BaseModel):
    """
    Y_t+1[j] = sqrt(0.000019 + 0.846 * (Y_bar_t[N_j]**2 + 0.3 * Y_bar_t-1[N_j]**2 + 0.2 * Y_bar_t-2[N_j]**2 + 0.1 * Y_bar_t-3[N_j]**2)) * W_t+1[j]
    """

    def __init__(self):
        self.time_from = [0,1,2,3] #from t -> t+1 and from t-1 -> t+1
        

    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t = mean_over_indices(Y[t], N_j)
        Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)
        Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)
        Y_bar_t_minus_3 = mean_over_indices(Y[t-3], N_j)

        squared_sum = (
            Y_bar_t**2 + 
            0.3 * Y_bar_t_minus_1**2 + 
            0.2 * Y_bar_t_minus_2**2 + 
            0.1 * Y_bar_t_minus_3**2
        )

        coefficient = math.sqrt(0.000019 + 0.846 * squared_sum)

        return coefficient * W[t][j]

@model_registry.register(model_id=18)    
class Model18(BaseModel):
    """
    Linear case 1
    Y_t+1[j] = 0.9 * Y_bar_t[N_j] + W_t+1[j]
    """

    def __init__(self):
        self.time_from = [0] #from t -> t+1
        

    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t = mean_over_indices(Y[t], N_j)
        return 0.9 * Y_bar_t + W[t][j]

@model_registry.register(model_id=19)    
class Model19(BaseModel):
    """
    Linear case 2
    Y_t+1[j] = 0.4 * Y_bar_t-1[N_j] + 0.6 * Y_bar_t-2[N_j] + W_t+1[j]
    """

    def __init__(self):
        self.time_from = [1,2] #from t-1 -> t+1, and from t-2 -> t+1
        

    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t_minus_1 = mean_over_indices(Y[t-1], N_j)
        Y_bar_t_minus_2 = mean_over_indices(Y[t-2], N_j)
        return 0.4 * Y_bar_t_minus_1 + 0.6 * Y_bar_t_minus_2 + W[t][j]

@model_registry.register(model_id=20)
class Model20(BaseModel):
    """
    Linear case 3
    Y_t+1[j] = 0.5 * Y_bar_t-3[N_j] + W_t+1[j]
    """

    def __init__(self):
        self.time_from = [3] #from t-3 -> t+1
        

    @staticmethod
    def update( Y, t, j, N_j, W):
        Y_bar_t_minus_3 = mean_over_indices(Y[t-3], N_j)
        return 0.5 * Y_bar_t_minus_3 + W[t][j]