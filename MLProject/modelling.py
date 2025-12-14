import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data preprocessing
data = pd.read_csv("bankdataset_preprocessing.csv")
X = data.values

# Set experiment (aman, tidak bikin konflik)
mlflow.set_experiment("CI_KMeans_Experiment")

# Train model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_
score = silhouette_score(X, labels)

# Manual logging (WAJIB untuk skilled/advanced)
mlflow.log_param("n_clusters", 3)
mlflow.log_metric("silhouette_score", score)

# Log model artifact
mlflow.sklearn.log_model(kmeans, "kmeans_model")

print("Training completed")
