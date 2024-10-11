import torch
from math import log

def compute_pairwise_squared_distances(points:torch.tensor, anchors:torch.tensor):
    '''
    turns (N,D) tensor of points and (M,D) tensor of anchor points into (M,N) list of pairwise squared distances
    '''
    differences=points.unsqueeze(1)-anchors.unsqueeze(0)
    squared_distances= (differences**2).sum(dim=2)
    return squared_distances

def embed_point_cloud(points:torch.tensor, anchors:torch.tensor, eps=1e-5, scale_factor=1):
    '''
    embeds (N,D) tensor of points as (M,N) tensor, based on (M,D) tensor of anchor points
    the (i,j)th entry of the (M,N) tensor is exp(-2*d^2/sigma) where d is the distance from the ith point to the jth anchor
    and sigma is the variance of all such distances
    '''
    squared_distances=compute_pairwise_squared_distances(points, anchors)
    variance=torch.std(anchors, dim=0).norm()**2
    return torch.exp(-1*scale_factor*squared_distances/(variance+eps))

def entropy_from_counts(counts):
    total_items=sum(counts)
    probabilities=[x/total_items for x in counts]
    return -1*sum([prob*log(prob) for prob in probabilities if prob>0])