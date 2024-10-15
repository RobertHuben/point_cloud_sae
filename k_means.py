import torch
import matplotlib.pyplot as plt
import time

from point_cloud_datasets import PointCloudDataset, create_lollipops_dataset, create_blobs_dataset, create_blob_grid_dataset, generate_datasets_for_saes
from utils import compute_pairwise_squared_distances, embed_point_cloud

import os
import glob

def k_nearest_neighbors(point_cloud:PointCloudDataset, num_clusters=3, plot_results=True):
    centroids=point_cloud.points[torch.randperm(len(point_cloud.points))[:num_clusters]]
    assignments=assign_to_nearest_nieghbor(point_cloud, centroids)
    stabilized=False
    steps=0
    print(f"Entropy of k-means after {steps} steps: {point_cloud.compute_entropy_of_clustering(assignments)}")
    if plot_results:
        plot_current_assignments(point_cloud, assignments, steps)
    while not stabilized:
        new_centroids=update_centroids(point_cloud, assignments, num_clusters=num_clusters)
        new_assignments=assign_to_nearest_nieghbor(point_cloud, new_centroids)
        stabilized=torch.all(new_assignments==assignments)
        assignments=new_assignments
        centroids=new_centroids
        steps+=1
        print(f"Entropy of k-means after {steps} steps: {point_cloud.compute_entropy_of_clustering(assignments)}")
        if plot_results:
            plot_current_assignments(point_cloud, assignments, steps)

    return centroids, assignments

def update_centroids(point_cloud:PointCloudDataset, assignments, num_clusters):
    centroids=[torch.mean(point_cloud.points[assignments==idx], dim=0) for idx in range(num_clusters)]
    return torch.stack(centroids)

def assign_to_nearest_nieghbor(point_cloud:PointCloudDataset, centroids):
    distances=compute_pairwise_squared_distances(point_cloud.points, centroids)
    return torch.min(distances, dim=1).indices

def k_nearest_demo(mode, num_classes, point_cloud_size, plot_results=True, seed=1):
    files_to_delete = glob.glob("k_nearest_neighbors_after_step*")
    for file_path in files_to_delete:
            os.remove(file_path)
            
    # point_cloud=create_blobs_dataset(point_cloud_size, seed=seed)
    _, point_cloud, __=generate_datasets_for_saes(mode, 1, point_cloud_size, 1, num_classes, class_weights_seed=None, points_seeds=[0,seed,0])
    t_start=time.time()
    k_nearest_neighbors(point_cloud, num_clusters=num_classes, plot_results=plot_results)
    t_end=time.time()
    print(f"{t_end-t_start} seconds")

def k_nearest_neighbor_with_embedding_demo():
    num_anchors=100
    train_size=100
    anchors=create_lollipops_dataset(num_anchors, seed=1)
    train_dataset=PointCloudDataset(create_blobs_dataset(train_size, seed=2))

    embedded_cloud=embed_point_cloud(train_dataset.points, anchors)
    x=k_nearest_neighbors(embedded_cloud)
    
    plt.scatter(x=train_dataset.points[:,0].detach().numpy(), y=train_dataset.points[:,1].detach().numpy(), c=x[1].detach().numpy())
    plt.savefig("embedded_k_nearest.png")
    plt.close()

def plot_current_assignments(point_cloud:PointCloudDataset, assignments, steps):
    x,y=point_cloud.points.T.detach().numpy()
    plt.scatter(x=x, y=y, c=assignments)
    plt.savefig(f"analysis_results/k_nearest_neighbors_after_step_{steps}.png")
    plt.close()

if __name__=="__main__":
    modes_with_specs={'basic_blobs':5, 'lollipops':5, 'blob_grid':18, 'random_blobs':10}
    mode='random_blobs'
    num_classes=modes_with_specs[mode]
    for seed in range(10):
        print(f"Running k-means on random seed {seed}")
        k_nearest_demo(mode, num_classes=num_classes, point_cloud_size=1000, plot_results=True, seed=seed)
