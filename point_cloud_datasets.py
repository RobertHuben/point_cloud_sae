import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class PointCloudDataset(Dataset):

    def __init__(self, points:torch.tensor, true_classes:torch.tensor):
        self.points=points
        self.true_classes=true_classes
        self.num_classes=len(set(int(x) for x in true_classes))

    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, index):
        return self.points[index]

    def show_as_scatter(self):
        for class_num in range(self.num_classes):
            x,y=self[torch.where(self.true_classes==class_num)].T
            plt.scatter(x,y, label=f"Class {class_num}")
        plt.title("Points in the dataset, separated by class")
        plt.legend()
        plt.show()


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
    rectangle_shapes=torch.tensor([[-.2, .2, -10,0],[5, 15, 0.2,-0.2]])
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

def create_point_blob(num_points, center, stdev):
    points=torch.normal(mean=center.tile(num_points,1), std=stdev.tile(num_points,1))
    return points

def create_point_cloud_rectangle(num_points, dims):
    left_x,right_x,top_y, bottom_y=dims
    stretch_factors=torch.tensor([[right_x-left_x, top_y-bottom_y]])
    shifts=torch.tensor([[left_x, bottom_y]])
    points=(torch.rand((num_points, 2))*stretch_factors)+shifts
    return points