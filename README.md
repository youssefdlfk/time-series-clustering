# Time Series Clustering and Validation Framework

A robust Python framework designed to cluster and evaluate time series data efficiently and flexibly using advanced clustering methods and validation indices.

## 📌 Overview

This project clusters time series data using:

- **Clustering algorithms**:
  - **K-Means**
  - **K-Shape**

- **Similarity measures**:
  - **Euclidean Distance**
  - **Dynamic Time Warping (DTW)**
  - **Cross-Correlation**

It evaluates clustering quality with several internal validation indices:

- **Silhouette Index**
- **Dunn Index**
- **Davies-Bouldin Index**
- **Calinski-Harabasz Index**
- **Average Proportion of Non-overlap (APN)**
- **Average Distance (AD)**
- **Hartigan Index**

The best combination of clustering algorithm and the number of clusters is determined by measuring the proximity to a reference ranking vector using the **Spearman footrule distance**.

Originally developed for analyzing psychological experiment data (Insight phenomena), this framework generalizes easily to other time series datasets.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Libraries specified in `requirements.txt` (notably `tslearn`, `numpy`, `pandas`, `scikit-learn`, and `matplotlib`)

### Installation

Clone the repository:

```bash
git clone <your_repository_url>
cd <your_repository_folder>
