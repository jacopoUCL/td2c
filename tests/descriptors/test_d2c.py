# test_d2c.py
import pytest
import networkx as nx
import pandas as pd
import numpy as np
from d2c.descriptors import D2C

# Fixture for a simple DAG and observations
@pytest.fixture
def simple_dag_and_data():
    maxlags = 3
    n_variables = 3
    dag = nx.DiGraph()
    dag.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11)])
    observations = pd.DataFrame(np.random.randn(100, n_variables*(maxlags+1)))
    return dag, observations

# Test initialization of the D2C class
def test_d2c_initialization(simple_dag_and_data):
    dag, observations = simple_dag_and_data
    d2c_instance = D2C(dags=[dag], observations=[observations], couples_to_consider_per_dag=10, MB_size=2)
    assert d2c_instance is not None
    assert d2c_instance.n_variables == 3
    # Add more assertions as needed

# Test the compute_descriptors_with_dag method
def test_compute_descriptors_with_dag(simple_dag_and_data):
    dag, observations = simple_dag_and_data
    d2c_instance = D2C(dags=[dag], observations=[observations], couples_to_consider_per_dag=4, MB_size=2)
    d2c_instance.initialize()  # Assuming this computes the descriptors
    assert d2c_instance.x_y is not None
    # Further assertions to validate the structure and data of the descriptors
