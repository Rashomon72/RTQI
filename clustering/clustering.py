import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

def hierarchical_clustering_with_rtqi(csv_path, new_data):
    """
    Perform hierarchical clustering and RTQI prediction.
    
    Parameters:
    - csv_path (str): Path to the dataset CSV file.
    - new_data (list): Input data for RTQI prediction.
    
    Returns:
    - dict: A dictionary containing clustering results, silhouette score, cluster-wise means, 
      and predicted RTQI for the new data.
    """
    # Load and preprocess the dataset
    df = pd.read_csv(csv_path).dropna()
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    # Adjust specific columns to negative values as per requirement
    columns_to_negate = [1, 3, 4]  # Column indices to negate
    df_scaled.iloc[:, columns_to_negate] *= -1

    # Perform hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=10)
    hierarchical_labels = hierarchical.fit_predict(df_scaled)
    df_scaled['Cluster'] = hierarchical_labels

    # Calculate silhouette score
    silhouette = silhouette_score(df_scaled.iloc[:, :-1], hierarchical_labels)

    # Compute cluster-wise means
    cluster_means = (
        df_scaled.groupby('Cluster')
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={'Cluster': 'Cluster_Mean'})
    )

    # Define RTQI mapping
    rtqi_mapping = {0: 9, 1: 6, 2: 10, 3: 5, 4: 4, 5: 2, 6: 1, 7: 8, 8: 7, 9: 3}
    df_scaled['RTQI'] = df_scaled['Cluster'].map(rtqi_mapping)

    # Predict RTQI for new data
    new_data_scaled = scaler.transform([new_data])
    new_data_scaled[0, columns_to_negate] *= -1  # Negate specific columns
    all_data = np.vstack([df_scaled.iloc[:, :-2], new_data_scaled])  # Combine data for clustering
    predicted_cluster = hierarchical.fit_predict(all_data)[-1]
    predicted_rtqi = rtqi_mapping.get(predicted_cluster, "Unknown Cluster")

    # Return results
    return {
        "silhouette_score": silhouette,
        "cluster_means": cluster_means,
        "predicted_cluster": predicted_cluster,
        "predicted_rtqi": predicted_rtqi
    }
