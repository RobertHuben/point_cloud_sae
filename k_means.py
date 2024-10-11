import torch
import matplotlib.pyplot as plt
import time

from point_cloud_datasets import PointCloudDataset, create_lollipops_dataset, create_blobs_dataset, create_blob_grid_dataset
from utils import compute_pairwise_squared_distances, embed_point_cloud


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

def k_nearest_demo(point_cloud_size, num_clusters=None, plot_results=True, seed=1):
    '''
    seed 1 fails, seed 2 works
    '''
    # point_cloud=create_blobs_dataset(point_cloud_size, seed=seed)
    point_cloud=create_blob_grid_dataset(point_cloud_size, seed=seed)
    if num_clusters==None:
        num_clusters=point_cloud.num_classes
    t_start=time.time()
    k_nearest_neighbors(point_cloud, num_clusters=num_clusters, plot_results=plot_results)
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
    for seed in range(10):
        print(f"Running k-means on random seed {seed}")
        k_nearest_demo(1000, plot_results=True, seed=seed)
