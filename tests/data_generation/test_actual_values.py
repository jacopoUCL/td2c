import numpy as np
import math
import pytest
from d2c.data_generation.builder import TSBuilder  
from d2c.data_generation.models import model_registry  

@pytest.fixture
def initialization():
    observations_per_time_series=10 
    n_variables=3 
    time_series_per_process=1 
    max_neighborhood_size=2
    seed=42 
    return observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed 

def test_observations_from_model_1(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 1
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()

    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        assert value == -0.4 * (3 - Y_bar_t**2) / (1 + Y_bar_t**2) + 0.6 * (3 - (Y_bar_t_minus_1 - 0.5)**3) / (1 + (Y_bar_t_minus_1 - 0.5)**4)

def test_observations_from_model_2(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 2
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()

    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])
                        assert value == (0.4 - 2 * math.exp(-50 * Y_bar_t_minus_1**2)) * Y_bar_t_minus_1 + (0.5 - 0.5 * math.exp(-50 * Y_bar_t_minus_2**2)) * Y_bar_t_minus_2

def test_observations_from_model_3(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 3
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()

    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])
                        assert value == 1.5 * math.sin(math.pi / 2 * Y_bar_t_minus_1) -math.sin(math.pi / 2 * Y_bar_t_minus_2)

def test_observations_from_model_4(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 4
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()

    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])
                        assert value == 2 * math.exp(-0.1 * Y_bar_t**2) * Y_bar_t -math.exp(-0.1 * Y_bar_t_minus_1**2) * Y_bar_t_minus_1 

# Model 5 (almost) never converges, therefore we skip the test (and likely skip the usage)
# def test_observations_from_model_5(initialization):
#     observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
#     model = 5
#     np.random.seed(42)
#     ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=0, noise_std=0, max_attempts=1000)
#     ts_builder.build()
#     neighbors = ts_builder.get_neighbors()
#     observations = ts_builder.get_generated_observations()
#     model = model_registry.get_model(model)
#     starting_time = model().get_maximum_time_lag()

#     for model_id, ts_data in observations.items():
#         for ts_index, data in ts_data.items():
#             for row_idx, row in enumerate(data):
#                 if row_idx > starting_time:
#                     for variable_idx, value in enumerate(row):
#                         variable_neighbors = neighbors[model_id][ts_index][variable_idx]
#                         Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
#                         Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
#                         Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])
#                         assert -2 * Y_bar_t * (Y_bar_t < 0) + 0.4 * Y_bar_t * (Y_bar_t < 0)
  
  
def test_observations_from_model_6(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 6
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()

    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])
                        assert value == 0.8 * math.log(1 + 3 * Y_bar_t**2) -0.6 * math.log(1 + 3 * Y_bar_t_minus_2**2)
  
  
def test_observations_from_model_7(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 7
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()

    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])
                        assert value ==  (0.4 - 2 * math.cos(40 * Y_bar_t_minus_2) * math.exp(-30 * Y_bar_t_minus_2**2))* Y_bar_t_minus_2 + (0.5 - 0.5 * math.exp(-50 * Y_bar_t_minus_1**2)) * Y_bar_t_minus_1


def test_observations_from_model_8(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 8
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()

    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])
                        assert value ==  (0.5 - 1.1 * math.exp(-50 * Y_bar_t**2)) * Y_bar_t + (0.3 - 0.5 * math.exp(-50 * Y_bar_t_minus_2**2)) * Y_bar_t_minus_2


def test_observations_from_model_9(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 9
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()

    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])
                        term1 = 0.3 * Y_bar_t
                        term2 = 0.6 * Y_bar_t_minus_1
                        term3_numerator = 0.1 - 0.9 * Y_bar_t + 0.8 * Y_bar_t_minus_1
                        term3_denominator = 1 + math.exp(-10 * Y_bar_t)
                        term3 = term3_numerator / term3_denominator

                        assert value ==  term1 + term2 + term3


def test_observations_from_model_10(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 10
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()
    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])

                        assert value == np.sign(Y_bar_t)


def test_observations_from_model_11(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 11
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()
    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])

                        term1 = 0.8 * Y_bar_t
                        term2_denominator = 1 + math.exp(-10 * Y_bar_t)
                        term2 = -0.8 * Y_bar_t / term2_denominator

                        assert value == term1 + term2

def test_observations_from_model_12(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 12
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()
    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])

                        term1 = 0.3 * Y_bar_t
                        term2 = 0.6 * Y_bar_t_minus_1
                        term3_numerator = 0.1 - 0.9 * Y_bar_t + 0.8 * Y_bar_t_minus_1
                        term3_denominator = 1 + math.exp(-10 * Y_bar_t)
                        term3 = term3_numerator / term3_denominator

                        assert value == term1 + term2 + term3

def test_observations_from_model_13(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 13
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()
    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])

                        assert value == 0.38 * Y_bar_t * (1 - Y_bar_t_minus_1)



def test_observations_from_model_14(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 14
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()
    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])

                        if Y_bar_t < 1:
                            assert value == -0.5 * Y_bar_t 
                        else:
                            assert value == 0.4 * Y_bar_t 


def test_observations_from_model_15(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 15
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()
    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])

                        if abs(Y_bar_t) < 1:
                            assert value == 0.9 * Y_bar_t
                        else:
                            assert value == -0.3 * Y_bar_t


def test_observations_from_model_16(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 16
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()
    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])

                        if (row_idx - 1) % 2 == 0:
                            assert value == -0.5 * Y_bar_t
                        else:
                            assert value == 0.4 * Y_bar_t 

# Model 17 does almost never converge, so we will skip this test
# def test_observations_from_model_17(initialization):
#     observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
#     model = 17
#     np.random.seed(42)
#     ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
#     ts_builder.build()
#     neighbors = ts_builder.get_neighbors()
#     observations = ts_builder.get_generated_observations()
#     model = model_registry.get_model(model)
#     starting_time = model().get_maximum_time_lag()
#     for model_id, ts_data in observations.items():
#         for ts_index, data in ts_data.items():
#             for row_idx, row in enumerate(data):
#                 if row_idx > starting_time:
#                     for variable_idx, value in enumerate(row):
#                         variable_neighbors = neighbors[model_id][ts_index][variable_idx]
#                         Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
#                         Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
#                         Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])
#                         Y_bar_t_minus_3 = np.mean(observations[model_id][ts_index][row_idx - 4][variable_neighbors])

#                         squared_sum = (
#                             Y_bar_t**2 + 
#                             0.3 * Y_bar_t_minus_1**2 + 
#                             0.2 * Y_bar_t_minus_2**2 + 
#                             0.1 * Y_bar_t_minus_3**2
#                         )

#                         assert value == math.sqrt(0.000019 + 0.846 * squared_sum)

def test_observations_from_model_18(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 18
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()
    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1]
                        [variable_neighbors])
                        

                        assert value == 0.9 * Y_bar_t

def test_observations_from_model_19(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 19
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()
    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        Y_bar_t = np.mean(observations[model_id][ts_index][row_idx - 1][variable_neighbors])
                        Y_bar_t_minus_1 = np.mean(observations[model_id][ts_index][row_idx - 2][variable_neighbors])
                        Y_bar_t_minus_2 = np.mean(observations[model_id][ts_index][row_idx - 3][variable_neighbors])

                        assert value == 0.4 * Y_bar_t_minus_1 + 0.6 * Y_bar_t_minus_2


def test_observations_from_model_20(initialization):
    observations_per_time_series, n_variables, time_series_per_process, max_neighborhood_size, seed  = initialization
    model = 20
    np.random.seed(42)
    ts_builder = TSBuilder(observations_per_time_series=observations_per_time_series, n_variables=n_variables, time_series_per_process=time_series_per_process, processes_to_use=[model], max_neighborhood_size=max_neighborhood_size, seed=seed, noise_std=0)
    ts_builder.build()
    neighbors = ts_builder.get_neighbors()
    observations = ts_builder.get_generated_observations()
    model = model_registry.get_model(model)
    starting_time = model().get_maximum_time_lag()
    for model_id, ts_data in observations.items():
        for ts_index, data in ts_data.items():
            for row_idx, row in enumerate(data):
                if row_idx > starting_time:
                    for variable_idx, value in enumerate(row):
                        variable_neighbors = neighbors[model_id][ts_index][variable_idx]
                        
                        Y_bar_t_minus_3 = np.mean(observations[model_id][ts_index][row_idx - 4][variable_neighbors])

                        assert value == 0.5 * Y_bar_t_minus_3