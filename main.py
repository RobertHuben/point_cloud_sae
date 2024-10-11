import torch
import random
import time

from point_cloud_sae import TopkSAE, SAEAnthropic
from point_cloud_datasets import create_blobs_dataset, create_lollipops_dataset, create_blob_grid_dataset 
from analysis import graph_point_cloud_results, graph_point_cloud_results_unified, graph_reconstruction_errors

if __name__=="__main__":
    modes_with_specs={'blobs':5, 'lollipops':5, 'blob_grid':18}
    mode='blob_grid'
    num_classes=modes_with_specs[mode]

    num_anchors=100
    train_size=100
    test_size=1000
    num_epochs=1000000//train_size
    cloud_scale_factor=10

    custom_number_of_features=None
    if custom_number_of_features==None:
        num_features=num_classes
    else:
        num_features=custom_number_of_features

    randomize_class_weights=False
    if randomize_class_weights:
        class_weights=[random.choice([1,2]) for _ in range(num_classes)]
        print(f"Randomized class weights are: {class_weights}")
    else:
        class_weights=None

    if mode=='blobs':
        anchors=create_blobs_dataset(num_anchors, seed=1, class_weights=class_weights).points
        train_dataset=create_blobs_dataset(train_size, seed=2, class_weights=class_weights)
        test_dataset=create_blobs_dataset(test_size, seed=3, class_weights=class_weights)
    elif mode=='lollipops':
        anchors=create_lollipops_dataset(num_anchors, seed=1, class_weights=class_weights).points
        train_dataset=create_lollipops_dataset(train_size, seed=2, class_weights=class_weights)
        test_dataset=create_lollipops_dataset(test_size, seed=3, class_weights=class_weights)
    elif mode=='blob_grid':
        anchors=create_blob_grid_dataset(num_anchors, seed=1, class_weights=class_weights).points
        train_dataset=create_blob_grid_dataset(train_size, seed=2, class_weights=class_weights)
        test_dataset=create_blob_grid_dataset(test_size, seed=3, class_weights=class_weights)

    for seed in [10]:
        torch.random.manual_seed(seed)
        # sae=SAEAnthropic(anchors, num_features=num_features, l1_sparsity_coefficient=2, cloud_scale_factor=cloud_scale_factor)
        sae=TopkSAE(anchors, num_features=num_features, k=1, use_relu=True, l1_sparsity_coefficient=0, cloud_scale_factor=cloud_scale_factor)
        sae.train_sae(train_dataset=train_dataset, eval_dataset=test_dataset, learning_rate=1e-4, num_epochs=num_epochs, report_every_n_data=100000, reinitialize_every_n_data=20000, batch_size=100)
    torch.save(sae, "trained_saes/point_cloud_sae.pkl")
    graph_point_cloud_results(test_dataset, anchors=anchors)
    graph_reconstruction_errors(test_dataset)
    graph_point_cloud_results_unified(test_dataset, anchors=anchors, anchor_details='encoder', save_name='all_features_encoder')
    graph_point_cloud_results_unified(test_dataset, anchors=anchors, anchor_details='decoder', save_name='all_features_decoder')