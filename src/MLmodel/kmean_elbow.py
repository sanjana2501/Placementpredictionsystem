import pandas as pd
from matplotlib import pyplot as plt

from src.MLmodel import kmean
from src.data.load_data import load_data
from src.data.preprocess import (
    split_X_data,
    identify_features,
    handle_missing_values,
    standardize,
    one_hot_encode_data,
    ordinal_encode_data
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
def find_optimal_k(X):
    wcss=[]
    for k in range(1,11):
        model = kmean(n_clusters=k, init="k-means++",n_init=10,max_iter=300,random_state=42)
        model.fit(X)
        wcss.append(model.inertia_)
    print("wcss values")
    for k,value in zip(range(1,11),wcss):
        print(k,value)
    plt.figure(figsize=(8,6))
    plt.plot(range(1,11),wcss,marker="o")
    plt.xlabel("k")
    plt.ylabel("wcss")
    plt.title("elbow method for optimal k")
    plt.xticks(range(1,11))
    plt.grid(True)
    plt.show()
    return wcss

def create_model(k):
    model = kmean(n_clusters=k, init="k-means++",n_init=10,max_iter=300,random_state=42)
    return model
def train_model(model,X):
    labels = model.fit_predict(X)
    print("\n Means trained succesfully")
    return model,labels
def evaluate_model(model,X,labels):
    print("\nInertia (WCSS):")
    print(model.inertia_)
    print("\nPrint inertia to converge:")
    print(model.n_iter_)
    silhouette = silhouette_score(X, labels)
    print("\nSilhouette score:")
    print(silhouette)
    return silhouette
def display_clusters(X,labels,model):
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,0],X[:,1],c=labels,cmap="virdis")
    plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],marker="X",s=200,label="Centroids",cmap="viridis")
    plt.title("K-Means Clusters")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.legend()
    plt.show()

def main()
    df = load_data()
    print(df.shape)
    X=split_X_data(df,drop_colums=["StudentId","PlacementStatus","Salary package","PlacementStatus"])
    print(X.shape)
    numerical_features,categorical_features(
        identify_features(X)
    )
    print(numerical_features)

    one_hot_features=[
        "Gender",
        "City",
        "Stream",
        "Specialization"
    ]

    ordinal_features=[
        "CollegeTier",
        "CGPA_Tier"
    ]
    X,_,imputer