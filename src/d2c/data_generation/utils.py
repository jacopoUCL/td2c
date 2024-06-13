def get_causal_dfs(dag, n_variables, maxlags):

    """
    Start from a DAG and return a pandas dataframe with the causal relationships in the DAG.
    The dataframe has a MultiIndex with the source and target variables.
    The values are 1 if the edge is causal, 0 otherwise.
    """

    import pandas as pd

    pairs = [(source, effect) for source in range(n_variables, n_variables * maxlags + n_variables) for effect in range(n_variables)]
    multi_index = pd.MultiIndex.from_tuples(pairs, names=['source', 'target'])
    causal_dataframe = pd.DataFrame(index=multi_index, columns=['is_causal'])

    causal_dataframe['is_causal'] = 0

    # print(causal_dataframe)
    
    for parent_node, child_node in dag.edges:
        child_variable = int(child_node.split("_")[0])
        child_lag = int(child_node.split("-")[1])
        corresponding_value_child = child_lag * n_variables + child_variable
        if corresponding_value_child < n_variables:
            parent_variable = int(parent_node.split("_")[0])
            parent_lag = int(parent_node.split("-")[1])
            corresponding_value_parent = parent_lag * n_variables + parent_variable
            causal_dataframe.loc[(corresponding_value_parent, child_variable), 'is_causal'] = 1
            

    return causal_dataframe


def custom_layout(G, n_variables, t_lag):
    """
    Create a custom layout for the graph where nodes with the same identifier
    are aligned in the same column, regardless of their connections.
    """
    pos = {}
    width = 1.0 / (n_variables - 1)
    height = 1.0 / (t_lag - 1)

    for node in G.nodes():
        #if node is integer
        if isinstance(node, int):
            i, t = node%n_variables, node//n_variables
        elif '_t-' in node:
            i, t = map(int, node.split('_t-'))
        else:
            i, t = int(node), 0
        pos[node] = (i * width, t * height)

    # Scale and center the positions
    pos = {node: (x * 10, y * 3) for node, (x, y) in pos.items()}
    return pos


def show_DAG(G, n_variables, t_lag):
    """
    Plot the DAG using the custom layout.
    """
    
    import networkx as nx
    import matplotlib.pyplot as plt
    # Using the custom layout for plotting
    plt.figure(figsize=(10, 6))
    pos_custom = custom_layout(G, n_variables, t_lag)
    nx.draw(G, pos_custom, with_labels=True, node_size=1000, node_color="lightpink", font_size=10, arrowsize=10)
    plt.title("Time Series DAG with Custom Layout")
    plt.show()

