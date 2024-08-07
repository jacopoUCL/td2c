**TD2C (Time-Dependency to Causality)** is a library for time series causal discovery. It focuses on computing asymmetric conditional mutual information terms, known as descriptors, within the Markov blankets of variable pairs.

To get started with TD2C, follow these steps:

1. Create a new conda environment:
    ```
    conda create --name td2c
    ```

2. Activate the conda environment:
    ```
    conda activate td2c
    ```

3. Install pip in the environment:
    ```
    conda install pip
    ```

4. Verify that pip is installed:
    ```
    which pip
    ```

    If it's not installed, add it to the system's path.
    ```
    export PATH="/home/yourusername/miniconda3/envs/your_env_name/bin:$PATH"
    ```
    

5. Install TD2C using pip:
    ```
    pip install .
    ```


## How to update the documentation? 
1. Make changes to `docs/` content on the `main` branch
2. re-build the book with `jupyter-book build docs/` 
3. from the `docs/` folder run `ghp-import -n -p -f _build/html` to push the newly built HTML to the `gh-pages` branch
4. 