import torch
import matplotlib.pyplot as plt
import time

from point_cloud_datasets import PointCloudDataset, generate_point_cloud_blobs, generate_point_cloud_lollipops
from utils import compute_pairwise_squared_distances, embed_point_cloud


def k_nearest_neighbors(point_cloud:PointCloudDataset, num_clusters=3, plot_progress=True):
    centroids=point_cloud[torch.randperm(len(point_cloud))[:num_clusters]]
    assignments=assign_to_nearest_nieghbor(point_cloud, centroids)
    stabilized=False
    steps=0
    if plot_progress:
        plot_current_assignments(point_cloud, assignments, steps)
    while not stabilized:
        new_centroids=update_centroids(point_cloud, assignments, num_clusters=num_clusters)
        new_assignments=assign_to_nearest_nieghbor(point_cloud, new_centroids)
        stabilized=torch.all(new_assignments==assignments)
        assignments=new_assignments
        centroids=new_centroids
        steps+=1
        if plot_progress:
            plot_current_assignments(point_cloud, assignments, steps)

    return centroids, assignments

def update_centroids(point_cloud, assignments, num_clusters):
    centroids=[torch.mean(point_cloud[assignments==idx], dim=0) for idx in range(num_clusters)]
    return torch.stack(centroids)

def assign_to_nearest_nieghbor(point_cloud, centroids):
    distances=compute_pairwise_squared_distances(point_cloud, centroids)
    return torch.min(distances, dim=1).indices

def k_nearest_demo(point_cloud_size, num_clusters=5, plot_progress=True):
    point_cloud=generate_point_cloud_blobs(point_cloud_size, seed=2)
    t_start=time.time()
    k_nearest_neighbors(point_cloud, num_clusters=num_clusters, plot_progress=plot_progress)
    t_end=time.time()
    print(f"{t_end-t_start} seconds")

def k_nearest_neighbor_with_embedding_demo(seed=1):
    '''
    seed 1 fails, seed 2 works
    '''
    num_anchors=100
    train_size=100
    anchors=generate_point_cloud_lollipops(num_anchors, seed=1)
    train_dataset=PointCloudDataset(generate_point_cloud_blobs(train_size, seed=seed))

    embedded_cloud=embed_point_cloud(train_dataset.points, anchors)
    x=k_nearest_neighbors(embedded_cloud)
    
    plt.scatter(x=train_dataset.points[:,0].detach().numpy(), y=train_dataset.points[:,1].detach().numpy(), c=x[1].detach().numpy())
    plt.savefig("embedded_k_nearest.png")
    plt.close()

def plot_current_assignments(point_cloud, assignments, steps):
    plt.scatter(x=point_cloud[:,0].detach().numpy(), y=point_cloud[:, 1].detach().numpy(), c=assignments)
    plt.savefig(f"analysis_results/point_cloud_results/k_nearest_neighbors_after_step_{steps}.png")
    plt.close()

if __name__=="__main__":
    k_nearest_demo(1000, plot_progress=True, seed=2)
