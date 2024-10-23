import torch
import matplotlib.pyplot as plt
from utils import entropy_from_counts, compute_pairwise_squared_distances
from math import sqrt, ceil
import random

class PointCloudDataset(torch.utils.data.Dataset):

    def __init__(self, points:torch.tensor, true_classes:torch.tensor):
        self.points=points
        self.true_classes=true_classes
        self.num_classes=len(set(int(x) for x in true_classes))

    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, index):
        return self.points[index]

    def plot_as_scatter(self, labels=None, save_name=None, show=False, close_at_end=True):
        if labels==None:
            labels=self.true_classes
            label_term="True Class"
        else:
            label_term="Cluster"
        label_set=sorted(list(set(int(x) for x in labels)))
        if max(label_set)>10:
            colors = plt.cm.tab20(torch.linspace(0, 1, 20))
            for class_num in label_set:
                x,y=self[torch.where(labels==class_num)].T
                plt.scatter(x,y, label=f"{label_term} {class_num}", color=colors[class_num])
        else:
            for class_num in label_set:
                x,y=self[torch.where(labels==class_num)].T
                plt.scatter(x,y, label=f"{label_term} {class_num}")
        plt.title("Points in the dataset, separated by class")
        plt.legend()
        if show:
            plt.show()
        if save_name:
            plt.savefig(f"analysis_results/{save_name}.png")
        if close_at_end:
            plt.close()

    def compute_entropy_of_clustering(self, cluster_assignments):
        '''
        computes the entropy of a classification attempt. 
        cluster_ids should be an iterable of length len(self), with integer values in range(self.num_classes)
        '''
        num_clusters=len(set(int(x) for x in cluster_assignments))
        cluster_assignments_one_hot= (cluster_assignments.unsqueeze(1)==torch.arange(num_clusters).unsqueeze(0)).int()
        ground_truth_one_hot= (self.true_classes.unsqueeze(1)==torch.arange(self.num_classes).unsqueeze(0)).int()
        cluster_counts=cluster_assignments_one_hot.sum(dim=0)
        combined_class_truths=cluster_assignments_one_hot.T @ ground_truth_one_hot
        entropies=torch.tensor([entropy_from_counts(row) for row in combined_class_truths])
        weighted_entropy=(entropies*cluster_counts).sum()/(cluster_counts.sum())
        return weighted_entropy


def create_blobs_dataset(num_points, stdev=1, seed=0, class_weights=None):
    '''
    generates a (num_points, 2) shaped tensor of points ~evenly split between the group means
    '''
    if seed:
        torch.random.manual_seed(seed)
    centers=torch.tensor([[0,0], [10,0], [0,10], [10,10], [5,7]], dtype=torch.float32)
    if class_weights==None:
        class_weights=[1 for _ in centers]
    total_class_weights=sum(class_weights)
    num_points_per_class=[num_points*class_weight//total_class_weights for class_weight in class_weights]
    num_points_per_class[0]+=num_points-sum(num_points_per_class)
    all_points=[]
    true_classes=[]
    class_num=0
    for num_points_in_this_class, center in zip(num_points_per_class,centers):
        all_points.append(create_point_blob(num_points_in_this_class, center, stdev=stdev*torch.ones(1,2)))
        true_classes.append(torch.tensor([class_num for _ in range(num_points_in_this_class)]))
        class_num+=1
    blobs_dataset= PointCloudDataset(torch.concat(all_points), torch.concat(true_classes))
    return blobs_dataset

def create_blob_grid_dataset(num_points, num_blobs=18, stdev=1, seed=0, class_weights=None):
    '''
    generates a (num_points, 2) shaped tensor of points ~evenly split between the group means
    '''
    if seed:
        torch.random.manual_seed(seed)
    blob_side_length=ceil(sqrt(num_blobs))
    centers=5*torch.stack([
        torch.arange(blob_side_length).unsqueeze(0)*torch.ones((blob_side_length,1)),
        torch.arange(blob_side_length).unsqueeze(1)*torch.ones((1,blob_side_length))] 
        ).flatten(start_dim=1).T
    centers=centers[:num_blobs]
    if class_weights==None:
        class_weights=[1 for _ in centers]
    total_class_weights=sum(class_weights)
    num_points_per_class=[num_points*class_weight//total_class_weights for class_weight in class_weights]
    num_points_per_class[0]+=num_points-sum(num_points_per_class)
    all_points=[]
    true_classes=[]
    class_num=0
    for num_points_in_this_class, center in zip(num_points_per_class,centers):
        all_points.append(create_point_blob(num_points_in_this_class, center, stdev=stdev*torch.ones(1,2)))
        true_classes.append(torch.tensor([class_num for _ in range(num_points_in_this_class)]))
        class_num+=1
    blobs_dataset= PointCloudDataset(torch.concat(all_points), torch.concat(true_classes))
    return blobs_dataset

def create_lollipops_dataset(num_points, seed=0, class_weights=None):
    '''
    generates a (num_points, 2) shaped tensor of points ~evenly split between 5 groups:
    1. a blob
    2. a blob
    3. a blob
    4. a stem (such that 1+4 form a lollipop)
    5. a stem (such that 2+5 form a lollipop)
    '''
    if seed:
        torch.random.manual_seed(seed)
    centers=torch.tensor([[0,0], [5,0], [0,10]], dtype=torch.float32)
    rectangle_shapes=torch.tensor([[-.2, .2, -10,-2],[7, 15, 0.2,-0.2]])
    if class_weights==None:
        class_weights=[1 for _ in range(len(centers)+len(rectangle_shapes))]
    total_class_weights=sum(class_weights)
    num_points_per_class=[num_points*class_weight//total_class_weights for class_weight in class_weights]
    num_points_per_class[0]+=num_points-sum(num_points_per_class)
    all_points=[]
    true_classes=[]
    class_num=0
    for num_points_in_this_class, center in zip(num_points_per_class[:3],centers):
        all_points.append(create_point_blob(num_points_in_this_class, center, stdev=torch.ones(1,2)))
        true_classes.append(torch.tensor([class_num for _ in range(num_points_in_this_class)]))
        class_num+=1
    for num_points_in_this_class, rectangle_shape in zip(num_points_per_class[3:], rectangle_shapes):
        all_points.append(create_point_cloud_rectangle(num_points_in_this_class, rectangle_shape))
        true_classes.append(torch.tensor([class_num for _ in range(num_points_in_this_class)]))
        class_num+=1
    lollipops_dataset= PointCloudDataset(torch.concat(all_points), torch.concat(true_classes))
    return lollipops_dataset

def create_random_blobs_dataset(num_points, num_blobs, blobs_seed=0, points_seed=1, class_weights=None):
    torch.random.manual_seed(blobs_seed)
    centers_scale_factor=3*torch.sqrt(torch.tensor(num_blobs))
    
    centers=torch.tensor([]).reshape((0,2))
    distance_cutoff=8

    while len(centers)<num_blobs:
        prospective_center=torch.normal(torch.zeros((1,2)), centers_scale_factor*torch.tensor([1,1]))
        squared_distance_to_previous_centers=compute_pairwise_squared_distances(prospective_center, centers)
        if torch.all(squared_distance_to_previous_centers>distance_cutoff**2):
            centers=torch.concat([centers,prospective_center], dim=0)

    pre_covariances= 4*(torch.rand((num_blobs, 2, 2))-.5)
    covariances=pre_covariances.transpose(1,2)@pre_covariances+.5*torch.eye(2)

    if class_weights==None:
        class_weights=[1 for _ in centers]
    total_class_weights=sum(class_weights)
    num_points_per_class=[num_points*class_weight//total_class_weights for class_weight in class_weights]
    num_points_per_class[0]+=num_points-sum(num_points_per_class)
    all_points=[]
    true_classes=[]
    class_num=0

    torch.random.manual_seed(points_seed)

    for num_points_in_this_class, center, covariance in zip(num_points_per_class, centers, covariances):
        all_points.append(create_skew_point_blob(num_points_in_this_class, center, covariance))
        true_classes.append(torch.tensor([class_num for _ in range(num_points_in_this_class)]))
        class_num+=1
    random_blobs_dataset= PointCloudDataset(torch.concat(all_points), torch.concat(true_classes))
    return random_blobs_dataset

def create_point_blob(num_points, center, stdev):
    points=torch.normal(mean=center.tile(num_points,1), std=stdev.tile(num_points,1))
    return points

def create_skew_point_blob(num_points, center, covariance):
    dist=torch.distributions.multivariate_normal.MultivariateNormal(center, covariance_matrix=covariance.float())
    points=dist.sample((num_points,))
    # points=torch.normal(mean=center.tile(num_points,1), std=stdev.tile(num_points,1))
    return points

def create_point_cloud_rectangle(num_points, dims):
    left_x,right_x,top_y, bottom_y=dims
    stretch_factors=torch.tensor([[right_x-left_x, top_y-bottom_y]])
    shifts=torch.tensor([[left_x, bottom_y]])
    points=(torch.rand((num_points, 2))*stretch_factors)+shifts
    return points

def generate_datasets_for_saes(mode:str, num_anchors:int, train_size:int, test_size:int, num_classes:int, class_weights=None, class_weights_seed=-1, points_seeds=[1,2,3], other_random_seed=10):
    if class_weights_seed>=0:
        random.seed(class_weights_seed)
        class_weights=[random.choice([1,2,3,4]) for _ in range(num_classes)]
        print(f"Randomized class weights are: {class_weights}")
    else:
        class_weights=None

    if mode=='basic_blobs':
        anchors=create_blobs_dataset(num_anchors, seed=points_seeds[0], class_weights=class_weights).points
        train_dataset=create_blobs_dataset(train_size, seed=points_seeds[1], class_weights=class_weights)
        test_dataset=create_blobs_dataset(test_size, seed=points_seeds[2], class_weights=class_weights)
    elif mode=='lollipops':
        anchors=create_lollipops_dataset(num_anchors, seed=points_seeds[0], class_weights=class_weights).points
        train_dataset=create_lollipops_dataset(train_size, seed=points_seeds[1], class_weights=class_weights)
        test_dataset=create_lollipops_dataset(test_size, seed=points_seeds[2], class_weights=class_weights)
    elif mode=='blob_grid':
        anchors=create_blob_grid_dataset(num_anchors, seed=points_seeds[0], class_weights=class_weights).points
        train_dataset=create_blob_grid_dataset(train_size, seed=points_seeds[1], class_weights=class_weights)
        test_dataset=create_blob_grid_dataset(test_size, seed=points_seeds[2], class_weights=class_weights)
    elif mode=='random_blobs':
        anchors=create_random_blobs_dataset(num_anchors, num_blobs=num_classes, blobs_seed=other_random_seed, points_seed=points_seeds[0], class_weights=class_weights).points
        train_dataset=create_random_blobs_dataset(train_size, num_blobs=num_classes, blobs_seed=other_random_seed, points_seed=points_seeds[1], class_weights=class_weights)
        test_dataset=create_random_blobs_dataset(test_size, num_blobs=num_classes, blobs_seed=other_random_seed, points_seed=points_seeds[2], class_weights=class_weights)
    return anchors, train_dataset, test_dataset


def graph_all_types():
    modes_with_specs={'basic_blobs':5, 'blob_grid':18, 'random_blobs':10,'lollipops':5, }
    for n, mode in enumerate(modes_with_specs):
        ax=plt.subplot(2,2,n+1)
        num_classes=modes_with_specs[mode]
        _, point_cloud, __=generate_datasets_for_saes(mode, 1, 1000, 1, num_classes, class_weights_seed=None, points_seeds=[0,1,0])
        point_cloud.plot_as_scatter(save_name="true_clusters_plot")
        ax.get_legend().remove()
        plt.title(mode)
    plt.suptitle("Datasets used, with true clusters shown")
    plt.tight_layout()
    plt.savefig("analysis_results/true_clusters_plot.png")

if __name__=="__main__":
    graph_all_types()
