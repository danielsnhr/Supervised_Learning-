# Iris Dataset - Model Comparison
from sklearn import datasets, metrics
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM (Linear)": SVC(kernel="linear"),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Neural Net": MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42),
}

results = {}

# Train and evaluate supervised models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.3f}")
    
    # Cross-validation scores
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name} CV Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Classification report for SVM as example
    if name == "SVM (Linear)":
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("-" * 50)

# KMeans clustering
print("\nKMeans Clustering Evaluation:")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)
cluster_labels = kmeans.labels_
ari = metrics.adjusted_rand_score(y, cluster_labels)
results["KMeans (ARI)"] = ari
print(f"KMeans Adjusted Rand Index: {ari:.3f}")

# PCA visualization of clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(12, 5))

# True labels
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('True Iris Species')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# KMeans clusters
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.title('KMeans Clusters')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

plt.tight_layout()
plt.savefig('kmeans_clusters.png', dpi=300, bbox_inches='tight')
print("KMeans cluster visualization saved as 'kmeans_clusters.png'")

# Results summary
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Score"])
print("\nModel Comparison:")
print(results_df)

# Plot results
plt.figure(figsize=(8, 4))
plt.bar(results_df["Model"], results_df["Score"], color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Score (Accuracy / ARI)")
plt.title("Iris Model Performance")
plt.tight_layout()
plt.savefig('iris_results.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'iris_results.png'")