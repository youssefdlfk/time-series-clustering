"""
Configuration module for the Time Series clustering project.

Stores all configurable hyperparameters, file paths, and constants utilized across modules. Adjust these parameters
depending on the desired clustering and available data.
"""


from dataclasses import dataclass


@dataclass
class ClusteringConfig:
    """Configuration parameters for time series clustering"""
    # Length of time series (up sampling)
    ts_length: int = 2050
    # Name of the csv raw data file
    csv_name: str = "dataRaw_AnswerUpdated_processed.csv"
    # Maximum number of iterations for training of clustering models
    max_iter: int = 10
    # Convergence threshold (tolerance)
    tol: float = 1e-5
    # Constraint type (sakoe-chiba, itakura, none)
    global_constraint: any = None
    # Constraining radius for sako-chiba (radius, none)
    constraint_radius: any = None
    # Constraining slope for itakura (slope, none)
    constraint_slope: any = None
    # Percentage of data points removed in the perturbed clusters for stability measures
    perc_col_del: float = 0.1
    # Range of the number of clusters to explore for validation
    k1: int = 2
    k2: int = 3
    # Number of best algo to save and plot
    topk: int = 3
    # Down sampling factor of time series to manage memory usage for DTW
    down_sample_factor: int = 700
    # Random seed for reproducibility
    random_seed: int = 42
    # Output files
    labels_output_file: str = 'cluster_labels'
    score_matrix_output_file: str = 'score_matrix'
    rank_matrix_output_file: str = 'rank_matrix'
    dist_to_ref_output_file: str = 'dist_to_ref'
    cluster_centers_output_file: str = 'cluster_centers'


default_config = ClusteringConfig()
