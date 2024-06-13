from setuptools import setup, find_packages

setup(
    name="d2c",
    version="0.1",
    packages=find_packages(where='src'), 
    package_dir={'': 'src'}, 
    install_requires=[
        'jupyter-book',
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'statsmodels',
        'tigramite',
        'lingam',
        'imblearn',
        'cachetools',
    ],
)