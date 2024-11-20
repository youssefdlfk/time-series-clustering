from dataclasses import dataclass


@dataclass
class ClusteringConfig:
    """Configuration parameters for time series clustering"""
    # Length of time series
    ts_length: int
    # Name of the csv data file
    csv_name: str
    # Maximum number of iterations for training of clustering models
    max_iter: int
    # Convergence threshold (tolerance)
    tol: float
    # Constraint type (sakoe-chiba, itakura, none)
    global_constraint: any
    # Constraining radius for sako-chiba (radius, none)
    constraint_radius: any
    # Constraining slope for itakura (slope, none)
    constraint_slope: any
    # Percentage of data points removed in the perturbed clusters for stability measures
    perc_col_del: float
    # Range of the number of clusters to explore for validation
    k1: int
    k2: int
    # Down sampling factor of time series to manage memory usage for DTW
    down_sample_factor: int
    # Random seed for reproducibility
    random_seed: int = 42
