import pytest
import math
import sys
import networkx as nx
from d2c.data_generation.models import model_registry



@pytest.fixture
def test_data():
    Y = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 3 variables, 3 time steps
    t = 2  # at time t, to compute t+1
    j = 0  # for variable j
    N_j = [0, 1]  # neighbourhood of j
    W = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]  # noise
    return Y, t, j, N_j, W


def test_model1_update(test_data):
    result = model_registry.get_model(1).update(*test_data)

    # Define the expected result
    expected_result = (
        -0.4 * (3 - ((7 + 8) / 2) ** 2) / (1 + ((7 + 8) / 2) ** 2)
        + 0.6 * (3 - ((4 + 5) / 2 - 0.5) ** 3) / (1 + ((4 + 5) / 2 - 0.5) ** 4)
        + 0.7
    )

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model2_update(test_data):
    # Call the update method
    result = model_registry.get_model(2).update(*test_data)

    # Define the expected result
    expected_result = (
        (0.4 - 2 * math.exp(-50 * ((4 + 5) / 2) ** 2)) * ((4 + 5) / 2)
        + (0.5 - 0.5 * math.exp(-50 * ((1 + 2) / 2) ** 2)) * ((1 + 2) / 2)
        + 0.7
    )

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model3_update(test_data):
    # Call the update method
    result = model_registry.get_model(3).update(*test_data)

    # Define the expected result
    expected_result = (
        1.5 * math.sin(math.pi / 2 * ((4 + 5) / 2))
        - math.sin(math.pi / 2 * ((1 + 2) / 2))
        + 0.7
    )

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model4_update(test_data):
    # Call the update method
    result = model_registry.get_model(4).update(*test_data)

    # Define the expected result
    expected_result = (
        2 * math.exp(-0.1 * ((7 + 8) / 2) ** 2) * ((7 + 8) / 2)
        - math.exp(-0.1 * ((4 + 5) / 2) ** 2) * ((4 + 5) / 2)
        + 0.7
    )

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model5_update(test_data):
    # Call the update method
    result = model_registry.get_model(5).update(*test_data)

    # Define the expected result
    expected_result = -2 * ((7 + 8) / 2) * (0) + 0.4 * ((7 + 8) / 2) * (0) + 0.7

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model6_update(test_data):
    # Call the update method
    result = model_registry.get_model(6).update(*test_data)

    # Define the expected result
    expected_result = (
        0.8 * math.log(1 + 3 * ((7 + 8) / 2) ** 2)
        - 0.6 * math.log(1 + 3 * ((1 + 2) / 2) ** 2)
        + 0.7
    )

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model7_update(test_data):
    # Call the update method
    result = model_registry.get_model(7).update(*test_data)

    # Define the expected result
    expected_result = (
        (0.4 - 2 * math.cos(40 * ((1 + 2) / 2)) * math.exp(-30 * ((1 + 2) / 2) ** 2))
        * ((1 + 2) / 2)
        + (0.5 - 0.5 * math.exp(-50 * ((4 + 5) / 2) ** 2)) * ((4 + 5) / 2)
        + 0.7
    )

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model8_update(test_data):
    # Call the update method
    result = model_registry.get_model(8).update(*test_data)

    # Define the expected result
    expected_result = (
        (0.5 - 1.1 * math.exp(-50 * ((7 + 8) / 2) ** 2)) * ((7 + 8) / 2)
        + (0.3 - 0.5 * math.exp(-50 * ((1 + 2) / 2) ** 2)) * ((1 + 2) / 2)
        + 0.7
    )

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model9_update(test_data):
    # Call the update method
    result = model_registry.get_model(9).update(*test_data)

    # Define the expected result
    expected_result = (
        0.3 * ((7 + 8) / 2)
        + 0.6 * ((4 + 5) / 2)
        + (0.1 - 0.9 * ((7 + 8) / 2) + 0.8 * ((4 + 5) / 2))
        / (1 + math.exp(-10 * ((7 + 8) / 2)))
        + 0.7
    )

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model10_update(test_data):
    # Call the update method
    result = model_registry.get_model(10).update(*test_data)

    # Define the expected result
    expected_result = 1 + 0.7

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model11_update(test_data):
    # Call the update method
    result = model_registry.get_model(11).update(*test_data)

    # Define the expected result
    expected_result = (
        0.8 * ((7 + 8) / 2)
        - (0.8 * ((7 + 8) / 2)) / (1 + math.exp(-10 * ((7 + 8) / 2)))
        + 0.7
    )

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model12_update(test_data):
    # Call the update method
    result = model_registry.get_model(12).update(*test_data)

    # Define the expected result
    expected_result = (
        0.3 * ((7 + 8) / 2)
        + 0.6 * ((4 + 5) / 2)
        + (0.1 - 0.9 * ((7 + 8) / 2) + 0.8 * ((4 + 5) / 2))
        / (1 + math.exp(-10 * ((7 + 8) / 2)))
        + 0.7
    )

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model13_update(test_data):
    # Call the update method
    result = model_registry.get_model(13).update(*test_data)

    # Define the expected result
    expected_result = 0.38 * ((7 + 8) / 2) * (1 - ((4 + 5) / 2)) + 0.7

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model14_update(test_data):
    # Call the update method
    result = model_registry.get_model(14).update(*test_data)

    # Define the expected result
    expected_result = 0.4 * ((7 + 8) / 2) + 0.7

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model15_update(test_data):
    # Call the update method
    result = model_registry.get_model(15).update(*test_data)

    # Define the expected result
    expected_result = -0.3 * ((7 + 8) / 2) + 0.7

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model16_update(test_data):
    # Call the update method
    result = model_registry.get_model(16).update(*test_data)

    expected_result = -0.5 * ((7 + 8) / 2) + 0.7

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model17_update(test_data):
    # Y[t-3] = [7, 8, 9]
    # Call the update method
    result = model_registry.get_model(17).update(*test_data)

    expected_result = (
        math.sqrt(
            0.000019
            + 0.846
            * (
                ((7 + 8) / 2) ** 2
                + 0.3 * ((4 + 5) / 2) ** 2
                + 0.2 * ((1 + 2) / 2) ** 2
                + 0.1 * ((7 + 8) / 2) ** 2
            )
        )
        * 0.7
    )

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model18_update(test_data):
    # Call the update method
    result = model_registry.get_model(18).update(*test_data)

    expected_result = 0.9 * ((7 + 8) / 2) + 0.7

    # Assert that the result matches the expected result
    assert result == expected_result


def test_model19_update(test_data):
    # Call the update method
    result = model_registry.get_model(19).update(*test_data)

    expected_result = 0.4 * ((4 + 5) / 2) + 0.6 * ((1 + 2) / 2) + 0.7
    # Assert that the result matches the expected result
    assert result == expected_result


def test_model20_update(test_data):
    # Y[t-3] = [7, 8, 9]
    result = model_registry.get_model(20).update(*test_data)
    expected_result = 0.5 * ((7 + 8) / 2) + 0.7
    # Assert that the result matches the expected result
    assert result == expected_result
import pytest
import math
import sys
sys.path.append("../..")
from d2c.data_generation.models import model_registry
from d2c.data_generation.models import add_edges


@pytest.fixture
def test_data():
    Y = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 3 variables, 3 time steps
    t = 2  # at time t, to compute t+1
    j = 0  # for variable j
    N_j = [0, 1]  # neighbourhood of j
    W = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]  # noise
    return Y, t, j, N_j, W


def test_model1_update(test_data):
    result = model_registry.get_model(1).update(*test_data)

    # Define the expected result
    expected_result = (
        -0.4 * (3 - ((7 + 8) / 2) ** 2) / (1 + ((7 + 8) / 2) ** 2)
        + 0.6 * (3 - ((4 + 5) / 2 - 0.5) ** 3) / (1 + ((4 + 5) / 2 - 0.5) ** 4)
        + 0.7
    )

    # Assert that the result matches the expected result
    assert result == expected_result


# Add a new test function to test the add_edges function
def test_add_edges():
    G = nx.DiGraph()
    T = 3
    N = 3
    N_j = [[0, 1], [1, 2], [2]]
    time_from = [1, 2]

    # Call the add_edges function
    add_edges(G, T, N, N_j, time_from)

    # Assert that the nodes and edges are added correctly
    assert len(G.nodes) == (T + 1) * N
    assert len(G.edges) == 15


